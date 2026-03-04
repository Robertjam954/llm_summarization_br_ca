"""
test_llm_app.py - V2 Prompt Fabrication Test
=============================================
Pipeline (all Claude Sonnet 4):
  1. Claude Sonnet 4 Vision  -> OCR / extract text from scanned PDFs
  2. Claude Sonnet 4 + v2 prompt -> clinical feature extraction
  3. Claude Sonnet 4 (DeepEval)  -> validation (Faithfulness, Hallucination)

The ONLY change from the original run is the prompt (v2).

Usage:
    python test_llm_app.py
"""

import os
import sys
import base64
import time
import logging
from pathlib import Path
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric, HallucinationMetric, GEval

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PRIVATE_DIR = Path(
    os.getenv("DATA_PRIVATE_DIR", r"C:\Users\jamesr4\loc\data_private")
)
DEID_DIR = DATA_PRIVATE_DIR / "deidentified"
RAW_DIR = DATA_PRIVATE_DIR / "raw"
PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "library" / "updated_developer_prompt_v2.txt"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "v2_fabrication_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model — Claude Sonnet 4.5 for everything
# ---------------------------------------------------------------------------
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
EVAL_MODEL = f"anthropic/{CLAUDE_MODEL}"  # DeepEval judge format

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Surgeon name -> raw folder mapping
# ---------------------------------------------------------------------------
SURGEON_DIR_MAP = {
    "Barrio, Andrea": "Barrio",
    "Capko, Deborah": "Capko",
    "Kirstein, Laurie": "Kirstein",
    "Tadros, Audree": "Tadros",
    "El-Tamer, Mahmoud": "EL Tamer",
    "Heerdt, Alexandra": "Heerdt",
    "Lee, Min": "Lee",
    "Montag, Giacomo": "Montag",
    "Moo, Tracy-Ann": "Moo",
}


# ===================================================================
# STEP 1: Claude Sonnet 4 Vision OCR
# ===================================================================
def pdf_page_to_base64(pdf_path: Path, page_num: int, dpi: int = 200) -> str:
    """Render a single PDF page to a base64-encoded PNG."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(png_bytes).decode("utf-8")


def ocr_pdf_with_claude(pdf_path: Path) -> str:
    """
    OCR a PDF using Claude Sonnet 4 Vision: send each page as an
    image, get back extracted text.
    """
    client = Anthropic()
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    doc.close()

    all_text = []
    for pg in range(n_pages):
        b64 = pdf_page_to_base64(pdf_path, pg)
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a scanned clinical document "
                                "(radiology or pathology report). "
                                "Extract ALL visible text from this "
                                "image. Preserve the original layout, "
                                "headings, labels, and values as "
                                "faithfully as possible. Output only "
                                "the extracted text, nothing else."
                            ),
                        },
                    ],
                }],
            )
            page_text = resp.content[0].text if resp.content else ""
        except Exception as exc:
            page_text = f"[OCR_ERROR page {pg}: {exc}]"
            log.warning(
                "  OCR error page %d of %s: %s",
                pg, pdf_path.name, exc,
            )
        all_text.append(page_text)
        time.sleep(0.5)

    return "\n\n".join(all_text)


def classify_report(filename: str) -> str:
    fn = filename.lower()
    if "hpi_ai" in fn:
        return "hpi_ai"
    if "hpi_human" in fn:
        return "hpi_human"
    if "path" in fn or "biopsy" in fn or "bopsy" in fn:
        return "pathology"
    if any(kw in fn for kw in ("imaging", "mammo", "mri", "us", "mmg")):
        return "radiology"
    if "h&p" in fn or "hp" in fn:
        return "hpi_human"
    return "other"


def ocr_case_folder(case_dir: Path) -> str:
    """OCR all source PDFs in a case folder using Claude Sonnet 4 Vision."""
    texts = []
    for f in sorted(case_dir.iterdir()):
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        if f.suffix.lower() != ".pdf":
            continue
        rtype = classify_report(f.name)
        if rtype in ("hpi_ai", "hpi_human"):
            continue
        log.info("    OCR: %s (%s)", f.name, rtype)
        text = ocr_pdf_with_claude(f)
        if text.strip():
            texts.append(f"--- {f.name} [{rtype}] ---\n{text}")
    return "\n\n".join(texts)


# ===================================================================
# STEP 2: Claude Sonnet 4 + v2 prompt -> extraction
# ===================================================================
def run_v2_extraction(source_text: str, v2_prompt: str) -> str:
    """Send OCR'd text to Claude Sonnet 4 with v2 developer prompt."""
    client = Anthropic()
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=v2_prompt,
        messages=[{
            "role": "user",
            "content": (
                "Extract all clinical features from the following "
                "source documents. Output ONLY the JSON structure "
                "specified in the system prompt.\n\n"
                f"SOURCE DOCUMENTS:\n{source_text}"
            ),
        }],
    )
    return resp.content[0].text if resp.content else ""


# ===================================================================
# Load fabrication cases
# ===================================================================
def load_fabrication_cases() -> list[dict]:
    val = pd.read_excel(
        DEID_DIR / "validation_datasheet_deidentified.xlsx"
    )
    ai_cols = [c for c in val.columns if c.endswith("_status_ai")]
    fab_mask = val[ai_cols].apply(
        lambda row: any(str(v) == "3" for v in row), axis=1
    )
    fab_df = val[fab_mask]

    cases = []
    for idx, row in fab_df.iterrows():
        surgeon = row["surgeon"]
        initials = row["patient_initials"]
        fab_elems = [
            c.replace("_status_ai", "")
            for c in ai_cols if str(row[c]) == "3"
        ]
        surgeon_dir = SURGEON_DIR_MAP.get(
            surgeon, surgeon.split(",")[0].strip()
        )
        surgeon_path = RAW_DIR / surgeon_dir

        matched = None
        if surgeon_path.exists():
            for d in surgeon_path.iterdir():
                if d.is_dir() and d.name.split("_")[1:2] == [initials]:
                    matched = d
                    break

        cases.append({
            "row_idx": idx,
            "surgeon": surgeon,
            "initials": initials,
            "fab_elements": fab_elems,
            "folder": matched,
            "comments": str(row.get("comments", "")),
        })
    return cases


# ===================================================================
# STEP 3: DeepEval metrics (Claude Sonnet 4 as judge)
# ===================================================================
def get_metrics():
    faithfulness = FaithfulnessMetric(
        threshold=0.7, model=EVAL_MODEL, include_reason=True
    )
    hallucination = HallucinationMetric(
        threshold=0.5, model=EVAL_MODEL, include_reason=True
    )
    fab_check = GEval(
        name="Fabrication Check",
        criteria=(
            "Determine whether the 'actual output' contains any "
            "fabricated clinical information NOT present in the "
            "provided context. A fabrication is any clinical detail "
            "(size, location, receptor status, diagnosis, etc.) in "
            "the output that cannot be found in the source documents. "
            "Score 1.0 if NO fabrications, 0.0 if fabrications exist."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        model=EVAL_MODEL,
        threshold=0.5,
    )
    return [faithfulness, hallucination, fab_check]


# ===================================================================
# Main
# ===================================================================
def main():
    log.info("=" * 60)
    log.info("V2 Fabrication Test")
    log.info("  OCR:     %s (Anthropic Vision)", CLAUDE_MODEL)
    log.info("  Summary: %s (Anthropic + v2 prompt)", CLAUDE_MODEL)
    log.info("  Eval:    %s (DeepEval judge)", EVAL_MODEL)
    log.info("=" * 60)

    v2_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    log.info("Loaded v2 prompt (%d chars)", len(v2_prompt))

    fab_cases = load_fabrication_cases()
    log.info("Fabrication cases: %d", len(fab_cases))

    test_cases = []
    case_details = []

    for case in fab_cases:
        folder = case["folder"]
        if not folder:
            log.warning(
                "SKIP %s/%s - no source folder",
                case["surgeon"], case["initials"],
            )
            continue

        log.info(
            "--- %s / %s  fab=%s ---",
            case["surgeon"], case["initials"],
            case["fab_elements"],
        )

        # STEP 1: Claude Sonnet 4 Vision OCR
        log.info("  [1/3] Claude Sonnet 4 OCR...")
        source_text = ocr_case_folder(folder)
        if not source_text:
            log.warning("  No text extracted, skipping")
            continue
        log.info("  Extracted %d chars", len(source_text))

        # STEP 2: Claude Sonnet 4 summary
        log.info("  [2/3] Claude Sonnet 4 summary...")
        try:
            summary = run_v2_extraction(source_text, v2_prompt)
        except Exception as exc:
            log.error("  Summary failed: %s", exc)
            continue
        log.info("  Summary: %d chars", len(summary))
        time.sleep(1)

        # Build DeepEval test case
        ctx = source_text[:8000]
        tc = LLMTestCase(
            input=(
                f"Extract clinical features for {case['initials']} "
                f"({case['surgeon']}). "
                f"Original fabrications: {case['fab_elements']}"
            ),
            actual_output=summary,
            retrieval_context=[ctx],
            context=[ctx],
        )
        test_cases.append(tc)
        case_details.append({
            "surgeon": case["surgeon"],
            "initials": case["initials"],
            "fab_elements": case["fab_elements"],
            "source_chars": len(source_text),
            "summary_chars": len(summary),
            "summary_preview": summary[:500],
            "full_summary": summary,
            "source_text": source_text,
            "comments": case["comments"],
        })

    if not test_cases:
        log.error("No test cases built. Exiting.")
        sys.exit(1)

    # STEP 3: DeepEval evaluation
    log.info("=" * 60)
    log.info("[3/3] DeepEval evaluation (%d cases)...", len(test_cases))
    metrics = get_metrics()

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        run_async=False,
        print_results=True,
    )

    # --- Save outputs ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(case_details).to_csv(
        run_dir / "case_details.csv", index=False
    )

    metric_rows = []
    for i, (tc, detail) in enumerate(zip(test_cases, case_details)):
        for m in metrics:
            metric_rows.append({
                "case": i,
                "surgeon": detail["surgeon"],
                "initials": detail["initials"],
                "fab_elements": str(detail["fab_elements"]),
                "metric": m.__class__.__name__,
                "score": m.score,
                "reason": getattr(m, "reason", ""),
                "passed": (
                    m.score >= m.threshold
                    if m.score is not None else None
                ),
            })
    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(run_dir / "metric_results.csv", index=False)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RESULTS: V2 Prompt Fabrication Test")
    print("=" * 60)
    print(f"OCR model:     {CLAUDE_MODEL}")
    print(f"Summary model: {CLAUDE_MODEL} + v2 prompt")
    print(f"Eval model:    {EVAL_MODEL}")
    print(f"Cases:         {len(test_cases)}")
    print(f"Output:        {run_dir}\n")

    for m in metrics:
        name = getattr(m, "name", m.__class__.__name__)
        sub = mdf[mdf["metric"] == m.__class__.__name__]
        if not sub.empty:
            print(f"  {name}:")
            print(f"    Avg score: {sub['score'].mean():.3f}")
            print(f"    Pass rate: {sub['passed'].mean():.1%}\n")

    print("-" * 60)
    for i, detail in enumerate(case_details):
        print(
            f"\nCase {i}: {detail['surgeon']} / {detail['initials']}"
        )
        print(f"  Original fabrications: {detail['fab_elements']}")
        cm = mdf[mdf["case"] == i]
        for _, r in cm.iterrows():
            tag = "PASS" if r["passed"] else "FAIL"
            reason = str(r["reason"])[:120]
            print(f"  {r['metric']}: {r['score']:.3f} [{tag}]")
            print(f"    {reason}")

    print("\n" + "=" * 60)
    log.info("Done. Results in %s", run_dir)
    return results


if __name__ == "__main__":
    main()
