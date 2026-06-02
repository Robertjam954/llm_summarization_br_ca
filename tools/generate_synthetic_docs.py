"""
generate_synthetic_docs.py
──────────────────────────────────────────────────────────────────────────────
Synthetic breast-imaging + pathology source-document generator.

Adapts the Microsoft LLM Data Creation approach
(EMNLP'23 "Making Large Language Models Better Data Creators") to produce
realistic de-identified clinical reports that exercise all 13 extraction
features used in this project.

Settings (mirrors the MS repo):
  naive    – format-only seed; generates N docs with no further guidance
  diverse  – varied clinical profiles (cancer subtype, modality mix, laterality)
  similar  – close variations of a specific seed document
  tree     – iterative: generated docs become seeds for the next round

Each output pair:
  <out_dir>/<id>_document.txt    – synthetic OCR-style clinical report
  <out_dir>/<id>_features.json   – ground-truth feature values (all 13 features)

Usage examples:
  # Naive: generate 20 docs from any .txt file in source_dir
  python tools/generate_synthetic_docs.py \\
      --source_dir C:/path/to/ocr_texts \\
      --setting naive --n 20 --output_dir data/synthetic/naive

  # Diverse: no seeds needed (Claude varies the clinical profile itself)
  python tools/generate_synthetic_docs.py \\
      --setting diverse --n 30 --output_dir data/synthetic/diverse

  # Similar: generate 10 close variations of one seed file
  python tools/generate_synthetic_docs.py \\
      --seed_file C:/path/to/ocr_texts/patient_001.txt \\
      --setting similar --n 10 --output_dir data/synthetic/similar

  # Tree: iterative (round 1 uses real seeds; later rounds use synthetic ones)
  python tools/generate_synthetic_docs.py \\
      --source_dir C:/path/to/ocr_texts \\
      --setting tree --n 40 --tree_rounds 3 --output_dir data/synthetic/tree
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not found. Run: pip install anthropic")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_DEFAULT = "claude-sonnet-4-6"
MAX_TOKENS    = 4096
TEMPERATURE   = 1.0   # high diversity for generation

FEATURE_NAMES = [
    "feature_1_lesion_size",
    "feature_2_lesion_location",
    "feature_3_calcifications_asymmetry",
    "feature_4_additional_enhancement_mri",
    "feature_5_extent",
    "feature_6_accurate_clip_placement",
    "feature_7_workup_recommendation",
    "feature_8_lymph_node",
    "feature_9_chronology_preserved",
    "feature_10_biopsy_method",
    "feature_11_invasive_component_size_pathology",
    "feature_12_histologic_diagnosis",
    "feature_13_receptor_status",
]

FEATURE_DESCRIPTIONS = {
    "feature_1_lesion_size":
        "Largest lesion dimension (imaging or pathology), units preserved.",
    "feature_2_lesion_location":
        "Laterality, quadrant, clock-face, depth, distance-from-nipple.",
    "feature_3_calcifications_asymmetry":
        "Presence/absence of calcifications or asymmetry; morphology and distribution if stated.",
    "feature_4_additional_enhancement_mri":
        "MRI enhancement beyond primary lesion (mass or non-mass).",
    "feature_5_extent":
        "Explicitly stated disease extent (multifocal, multicentric, bilateral, or localized).",
    "feature_6_accurate_clip_placement":
        "Whether a biopsy clip/marker was placed, shape and location if stated.",
    "feature_7_workup_recommendation":
        "Imaging follow-up, biopsy type, surgical evaluation, or surveillance (verbatim).",
    "feature_8_lymph_node":
        "Side-specific lymph node findings or explicit no-lymphadenopathy statement.",
    "feature_9_chronology_preserved":
        "Boolean: radiology studies ordered oldest → most recent.",
    "feature_10_biopsy_method":
        "Exact biopsy technique (e.g., US-guided 14g CNB, stereotactic VAB).",
    "feature_11_invasive_component_size_pathology":
        "Size of invasive component in cm if explicitly stated in pathology.",
    "feature_12_histologic_diagnosis":
        "Exact histologic subtype(s) as written, including size/site.",
    "feature_13_receptor_status":
        "ER, PR/PgR, HER2 IHC, HER2 ISH results with category, %, intensity, controls.",
}

DIVERSE_PROFILES = [
    {
        "cancer_type":  "IDC grade 2",
        "laterality":   "left",
        "modalities":   "mammogram + ultrasound + pathology + receptor",
        "special":      "clip placed, calcifications present",
    },
    {
        "cancer_type":  "DCIS high-grade",
        "laterality":   "right",
        "modalities":   "mammogram + MRI + stereotactic biopsy + pathology",
        "special":      "extensive calcifications, no invasion",
    },
    {
        "cancer_type":  "IDC grade 3 with DCIS component",
        "laterality":   "left",
        "modalities":   "ultrasound + MRI + US-guided biopsy + pathology + receptor",
        "special":      "multifocal disease, axillary lymph node involvement",
    },
    {
        "cancer_type":  "ILC grade 1",
        "laterality":   "right",
        "modalities":   "mammogram + MRI + US-guided biopsy + pathology",
        "special":      "no calcifications, non-mass MRI enhancement",
    },
    {
        "cancer_type":  "IDC triple-negative grade 3",
        "laterality":   "left",
        "modalities":   "ultrasound + MRI + core biopsy + pathology + receptor",
        "special":      "large tumor > 3 cm, no clip placed",
    },
    {
        "cancer_type":  "IDC HER2-positive grade 2",
        "laterality":   "right",
        "modalities":   "mammogram + ultrasound + MRI + biopsy + pathology + receptor",
        "special":      "clip placed (BB marker), low axillary node",
    },
    {
        "cancer_type":  "DCIS intermediate-grade",
        "laterality":   "left",
        "modalities":   "mammogram + stereotactic biopsy + pathology",
        "special":      "amorphous calcifications, localized extent",
    },
    {
        "cancer_type":  "IDC grade 1 luminal A",
        "laterality":   "right",
        "modalities":   "mammogram + ultrasound + US-guided biopsy + pathology + receptor",
        "special":      "small lesion < 1 cm, no lymph node involvement",
    },
]

# ── Prompt builders ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""
    You are a synthetic clinical document generator for a breast cancer AI research project.

    Your task is to write realistic, de-identified breast imaging and pathology source
    documents that look like OCR-extracted text from actual hospital reports.

    DOCUMENT STYLE:
    - Write in the third person, past tense (as actual radiology/pathology reports are written).
    - Include realistic section headers in ALL CAPS (e.g., MAMMOGRAM, ULTRASOUND, MRI,
      PATHOLOGY REPORT, RECEPTOR STATUS).
    - Include realistic dates (use fictitious dates in the format MM/DD/YYYY).
    - Include realistic radiologist/pathologist sign-off lines.
    - Documents may contain OCR artifacts (extra spaces, occasional line-breaks mid-word)
      to reflect the OCR-extracted nature of the source data.
    - Do NOT include real patient names or MRNs. Use placeholders like [PATIENT] or
      initials such as "J.K." if initials are needed.

    CLINICAL REALISM:
    - Use correct medical terminology for breast imaging and pathology.
    - Sizes must be plausible (lesions typically 0.3–5.0 cm).
    - BI-RADS categories must match the described findings.
    - Receptor status values must be internally consistent (e.g., ER+ means percentage > 0).
    - Chronology must be internally consistent (dates must be in logical order).

    ANTI-HALLUCINATION NOTE FOR GROUND-TRUTH JSON:
    - Every value in the features JSON must be directly supported by verbatim text in the
      document you just wrote. Do not invent features not present in the document text.
    - If a feature is not present in the document, set its value to "Not reported".
""").strip()


def _feature_json_template() -> str:
    lines = ["{\n  \"lesions\": [\n    {\n      \"lesion_id\": \"L1\""]
    for f in FEATURE_NAMES[:-1]:   # all except chronology
        lines.append(f'      "{f}": {{"value": "", "evidence": ""}}')
    lines.append("    }\n  ]")
    lines.append('  "feature_9_chronology_preserved": true\n}')
    return ",\n".join(lines)


def _format_seed(seed_text: str, max_chars: int = 3000) -> str:
    """Truncate seed to fit comfortably in context."""
    if len(seed_text) > max_chars:
        return seed_text[:max_chars] + "\n... [truncated for brevity]"
    return seed_text


def _build_naive_prompt(seed_text: str, index: int, total: int) -> str:
    return textwrap.dedent(f"""
        Below is one example of a real (de-identified) breast imaging + pathology source
        document in OCR-extracted text format.

        SEED DOCUMENT:
        ──────────────────────────────────────────
        {_format_seed(seed_text)}
        ──────────────────────────────────────────

        Generate synthetic document #{index} of {total}.

        Write a COMPLETE, realistic clinical source document that:
        1. Follows the same overall section structure as the seed.
        2. Contains DIFFERENT clinical findings (different sizes, locations, diagnoses).
        3. Includes ALL sections that allow extraction of the 13 clinical features listed below.

        After the document, output a JSON block (between ```json and ```) with the
        ground-truth values for all 13 features exactly as they appear in the document.

        FEATURES TO INCLUDE:
        {chr(10).join(f'  {k}: {v}' for k, v in FEATURE_DESCRIPTIONS.items())}

        OUTPUT FORMAT:
        [DOCUMENT TEXT — plain text, no markdown]

        ```json
        {{
          "lesions": [
            {{
              "lesion_id": "L1",
              {', '.join(f'"{f}": {{"value": "...", "evidence": "..."}}' for f in FEATURE_NAMES if f != "feature_9_chronology_preserved")}
            }}
          ],
          "feature_9_chronology_preserved": true
        }}
        ```
    """).strip()


def _build_diverse_prompt(profile: dict, index: int, total: int) -> str:
    return textwrap.dedent(f"""
        Generate a synthetic de-identified breast imaging + pathology source document.

        CLINICAL PROFILE FOR THIS DOCUMENT (#{index} of {total}):
        - Cancer type  : {profile['cancer_type']}
        - Laterality   : {profile['laterality']}
        - Modalities   : {profile['modalities']}
        - Special notes: {profile['special']}

        Write a COMPLETE, realistic clinical source document containing:
        - All modality sections that correspond to the profile above.
        - Realistic dates (chronological order), BI-RADS category, measurements,
          biopsy details, pathology findings, and receptor results.

        After the document, output a JSON block (between ```json and ```) with the
        ground-truth values for all 13 features exactly as they appear in the document.

        FEATURES:
        {chr(10).join(f'  {k}: {v}' for k, v in FEATURE_DESCRIPTIONS.items())}

        OUTPUT FORMAT:
        [DOCUMENT TEXT — plain text, no markdown]

        ```json
        {{
          "lesions": [
            {{
              "lesion_id": "L1",
              {', '.join(f'"{f}": {{"value": "...", "evidence": "..."}}' for f in FEATURE_NAMES if f != "feature_9_chronology_preserved")}
            }}
          ],
          "feature_9_chronology_preserved": true
        }}
        ```
    """).strip()


def _build_similar_prompt(seed_text: str, index: int, total: int) -> str:
    return textwrap.dedent(f"""
        Below is a seed breast imaging + pathology source document.

        SEED DOCUMENT:
        ──────────────────────────────────────────
        {_format_seed(seed_text)}
        ──────────────────────────────────────────

        Generate a CLOSE VARIATION (#{index} of {total}) that:
        - Keeps the same modality structure and section layout.
        - Changes measurement values (sizes, clock positions) by realistic amounts.
        - Changes receptor status or histologic subtype in a clinically plausible way.
        - Keeps the same laterality UNLESS you specifically want to test left/right swap.
        - Preserves the chronological ordering of studies.

        After the document, output a JSON block (between ```json and ```) with the
        ground-truth values for all 13 features exactly as they appear in the document.

        FEATURES:
        {chr(10).join(f'  {k}: {v}' for k, v in FEATURE_DESCRIPTIONS.items())}

        OUTPUT FORMAT:
        [DOCUMENT TEXT — plain text, no markdown]

        ```json
        {{
          "lesions": [
            {{
              "lesion_id": "L1",
              {', '.join(f'"{f}": {{"value": "...", "evidence": "..."}}' for f in FEATURE_NAMES if f != "feature_9_chronology_preserved")}
            }}
          ],
          "feature_9_chronology_preserved": true
        }}
        ```
    """).strip()


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_output(raw: str) -> tuple[str, Optional[dict]]:
    """Split Claude output into (document_text, features_dict)."""
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if json_match:
        doc_text = raw[:json_match.start()].strip()
        try:
            features = json.loads(json_match.group(1))
        except json.JSONDecodeError as exc:
            print(f"  [WARN] JSON parse error: {exc}")
            features = None
    else:
        doc_text  = raw.strip()
        features  = None
    return doc_text, features


# ── Claude call ───────────────────────────────────────────────────────────────

def call_claude(prompt: str, model: str, client: anthropic.Anthropic) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Seed loading ──────────────────────────────────────────────────────────────

def load_seeds(source_dir: Optional[str], seed_file: Optional[str]) -> list[str]:
    seeds = []
    if seed_file:
        p = Path(seed_file)
        if p.exists():
            seeds.append(p.read_text(encoding="utf-8", errors="replace"))
        else:
            print(f"[WARN] seed_file not found: {seed_file}")
    if source_dir:
        txt_files = sorted(Path(source_dir).glob("*.txt"))
        for f in txt_files:
            seeds.append(f.read_text(encoding="utf-8", errors="replace"))
        print(f"Loaded {len(txt_files)} seed files from {source_dir}")
    return seeds


# ── Resume helper ────────────────────────────────────────────────────────────

def _done_indices(out_dir: Path, setting: str) -> set[int]:
    """Return set of doc indices already saved in out_dir for this setting."""
    done = set()
    prefix = setting.split("_")[0]   # e.g. 'naive', 'diverse', 'similar'
    for p in out_dir.glob(f"{prefix}_*_document.txt"):
        m = re.match(rf"{re.escape(prefix)}_(\d{{4}})_", p.name)
        if m:
            done.add(int(m.group(1)))
    return done


# ── Per-setting generators ────────────────────────────────────────────────────

def run_naive(seeds: list[str], n: int, out_dir: Path,
              model: str, client: anthropic.Anthropic) -> list[Path]:
    if not seeds:
        raise ValueError("--setting naive requires --source_dir or --seed_file")
    done  = _done_indices(out_dir, "naive")
    saved = []
    seed_cycle = seeds * (n // len(seeds) + 1)
    for i in range(1, n + 1):
        if i in done:
            print(f"  [naive] skipping {i}/{n} (already saved)")
            continue
        seed = seed_cycle[i - 1]
        prompt = _build_naive_prompt(seed, i, n)
        print(f"  [naive] generating {i}/{n} ...", end=" ", flush=True)
        raw   = call_claude(prompt, model, client)
        saved.extend(_save_result(raw, out_dir, setting="naive", index=i))
        print("saved")
    return saved


def run_diverse(n: int, out_dir: Path,
                model: str, client: anthropic.Anthropic) -> list[Path]:
    done     = _done_indices(out_dir, "diverse")
    saved    = []
    profiles = DIVERSE_PROFILES * (n // len(DIVERSE_PROFILES) + 1)
    random.shuffle(profiles)
    for i in range(1, n + 1):
        if i in done:
            print(f"  [diverse] skipping {i}/{n} (already saved)")
            continue
        profile = profiles[i - 1]
        prompt  = _build_diverse_prompt(profile, i, n)
        print(f"  [diverse] generating {i}/{n} ({profile['cancer_type']}) ...",
              end=" ", flush=True)
        raw  = call_claude(prompt, model, client)
        saved.extend(_save_result(raw, out_dir, setting="diverse", index=i,
                                  meta=profile))
        print("saved")
    return saved


def run_similar(seeds: list[str], n: int, out_dir: Path,
                model: str, client: anthropic.Anthropic) -> list[Path]:
    if not seeds:
        raise ValueError("--setting similar requires --source_dir or --seed_file")
    done  = _done_indices(out_dir, "similar")
    saved = []
    seed_cycle = seeds * (n // len(seeds) + 1)
    for i in range(1, n + 1):
        if i in done:
            print(f"  [similar] skipping {i}/{n} (already saved)")
            continue
        seed   = seed_cycle[i - 1]
        prompt = _build_similar_prompt(seed, i, n)
        print(f"  [similar] generating {i}/{n} ...", end=" ", flush=True)
        raw    = call_claude(prompt, model, client)
        saved.extend(_save_result(raw, out_dir, setting="similar", index=i))
        print("saved")
    return saved


def run_tree(seeds: list[str], n: int, rounds: int, out_dir: Path,
             model: str, client: anthropic.Anthropic) -> list[Path]:
    """
    Tree setting: generated docs from round k become seeds for round k+1.
    Total generated = n (distributed evenly across rounds).
    """
    if not seeds:
        raise ValueError("--setting tree requires --source_dir or --seed_file")
    per_round   = max(1, n // rounds)
    current_seeds = list(seeds)
    all_saved     = []
    total_done    = 0

    for r in range(1, rounds + 1):
        this_n = per_round if r < rounds else n - total_done
        print(f"  [tree] round {r}/{rounds}: generating {this_n} docs ...")
        round_texts = []
        for i in range(1, this_n + 1):
            seed   = random.choice(current_seeds)
            prompt = _build_naive_prompt(seed, i, this_n)
            raw    = call_claude(prompt, model, client)
            paths  = _save_result(raw, out_dir, setting=f"tree_r{r}", index=i)
            all_saved.extend(paths)
            doc_path = next((p for p in paths if p.suffix == ".txt"), None)
            if doc_path:
                round_texts.append(doc_path.read_text(encoding="utf-8"))
            total_done += 1
            print(f"    round {r} doc {i}/{this_n} saved")
        current_seeds = round_texts   # synthetic docs become next-round seeds

    return all_saved


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_result(raw: str, out_dir: Path, setting: str,
                 index: int, meta: Optional[dict] = None) -> list[Path]:
    doc_text, features = _parse_output(raw)
    doc_id   = f"{setting}_{index:04d}_{uuid.uuid4().hex[:6]}"
    doc_path = out_dir / f"{doc_id}_document.txt"
    doc_path.write_text(doc_text, encoding="utf-8")

    saved = [doc_path]

    feat_payload = {
        "doc_id":   doc_id,
        "setting":  setting,
        "index":    index,
        "profile":  meta or {},
        "features": features,
        "raw_output_chars": len(raw),
        "json_parse_ok": features is not None,
    }
    feat_path = out_dir / f"{doc_id}_features.json"
    feat_path.write_text(json.dumps(feat_payload, indent=2), encoding="utf-8")
    saved.append(feat_path)
    return saved


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic breast imaging + pathology documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source_dir",  help="Directory of seed .txt OCR files.")
    p.add_argument("--seed_file",   help="Single seed .txt file (for --setting similar).")
    p.add_argument("--setting",     choices=["naive", "diverse", "similar", "tree"],
                   default="naive")
    p.add_argument("--n",           type=int, default=10,
                   help="Number of synthetic documents to generate.")
    p.add_argument("--tree_rounds", type=int, default=3,
                   help="Number of tree rounds (only used with --setting tree).")
    p.add_argument("--output_dir",  default="data/synthetic",
                   help="Directory to write output files.")
    p.add_argument("--model",       default=MODEL_DEFAULT)
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for reproducible profile selection.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to .env or environment.")
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    out_dir = Path(args.output_dir) / args.setting
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory : {out_dir}")
    print(f"Setting          : {args.setting}")
    print(f"N docs           : {args.n}")
    print(f"Model            : {args.model}")
    print()

    seeds = load_seeds(args.source_dir, args.seed_file)

    if args.setting == "naive":
        run_naive(seeds, args.n, out_dir, args.model, client)
    elif args.setting == "diverse":
        run_diverse(args.n, out_dir, args.model, client)
    elif args.setting == "similar":
        run_similar(seeds, args.n, out_dir, args.model, client)
    elif args.setting == "tree":
        run_tree(seeds, args.n, args.tree_rounds, out_dir, args.model, client)

    print()
    print(f"Done. Files written to: {out_dir}")
    json_ok  = sum(1 for f in out_dir.glob("*_features.json")
                   if json.loads(f.read_text()).get("json_parse_ok"))
    total_f  = sum(1 for _ in out_dir.glob("*_features.json"))
    print(f"JSON parse success: {json_ok}/{total_f}")


if __name__ == "__main__":
    main()
