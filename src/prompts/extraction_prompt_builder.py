"""
extraction_prompt_builder.py
Builds the structured extraction prompt for a given feature
using the anti-fabrication developer prompt as the system identity.
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from src.workflows.extraction_state import Chunk

SYSTEM_IDENTITY = """You are a breast imaging + pathology information extraction system.

Priorities (in order):
1. Accuracy
2. Low fabrication
3. Verbatim fidelity
4. Chronology preservation

HARD CONSTRAINTS (ANTI-FABRICATION):
- Use only information explicitly stated in the retrieved chunks. No inference.
- If missing: output "Not reported".
- If conflicting or unclear: output "Indeterminate" and include verbatim snippets.
- Preserve units and wording exactly as written.
- Do not convert or normalize measurements."""

OUTPUT_SCHEMA = """{
  "value": "",
  "evidence": "",
  "page_refs": [],
  "confidence": 0.0,
  "reasoning_for_confidence": ""
}"""

FEW_SHOT_EXAMPLES = {
    # --- STANDARD: explicit positive finding ---
    "feature_1_lesion_size": (
        'EXAMPLE (standard):\n'
        'INPUT: "A 1.2 cm irregular mass is identified in the left breast at 10 o\'clock."\n'
        'OUTPUT: {"value": "1.2 cm", "evidence": "A 1.2 cm irregular mass is identified '
        'in the left breast at 10 o\'clock.", "page_refs": [3], "confidence": 1.0, '
        '"reasoning_for_confidence": "Exact measurement stated explicitly."}\n\n'
        'EXAMPLE (conflict — use Indeterminate):\n'
        'INPUT: "Initial report: \'3 cm mass\'. Later addendum: \'small 2 cm lesion\'."\n'
        'OUTPUT: {"value": "Indeterminate", "evidence": "Conflicting: \'3 cm mass\' vs '
        '\'small 2 cm lesion\'.", "page_refs": [2, 4], "confidence": 0.4, '
        '"reasoning_for_confidence": "Two contradictory measurements; cannot resolve without '
        'additional context."}'
    ),
    # --- RECEPTOR STATUS: standard + missing-section case ---
    "feature_13_receptor_status": (
        'EXAMPLE (standard):\n'
        'INPUT: "ER: Positive (95%, strong). PR: Positive (80%). HER2 IHC: 1+. HER2 ISH: Not performed."\n'
        'OUTPUT: {"value": "ER positive 95% strong; PR positive 80%; HER2 IHC 1+; HER2 ISH not performed", '
        '"evidence": "ER: Positive (95%, strong). PR: Positive (80%). HER2 IHC: 1+. HER2 ISH: Not performed.", '
        '"page_refs": [5], "confidence": 1.0, '
        '"reasoning_for_confidence": "All receptor results stated explicitly."}\n\n'
        'EXAMPLE (missing — no receptor section in document):\n'
        'INPUT: "Biopsy: invasive ductal carcinoma. [No receptor testing section present]"\n'
        'OUTPUT: {"value": "Not reported", "evidence": "", "page_refs": [], '
        '"confidence": 1.0, "reasoning_for_confidence": "No receptor testing results '
        'found in any retrieved chunk. Absence is explicit."}'
    ),
    # --- INVASIVE SIZE: standard + missing case ---
    "feature_11_invasive_component_size_pathology": (
        'EXAMPLE (standard):\n'
        'INPUT: "Invasive ductal carcinoma, grade 2, measuring 1.8 cm in greatest dimension."\n'
        'OUTPUT: {"value": "1.8 cm", "evidence": "Invasive ductal carcinoma, grade 2, '
        'measuring 1.8 cm in greatest dimension.", "page_refs": [6], "confidence": 1.0, '
        '"reasoning_for_confidence": "Invasive component size explicitly stated."}\n\n'
        'EXAMPLE (DCIS only — no invasive component):\n'
        'INPUT: "DCIS, high grade. No invasive carcinoma identified."\n'
        'OUTPUT: {"value": "Not reported", "evidence": "No invasive carcinoma identified.", '
        '"page_refs": [5], "confidence": 1.0, '
        '"reasoning_for_confidence": "Pathology confirms DCIS only; invasive size is absent by explicit statement."}'
    ),
    # --- BIOPSY CLIP: standard + confirmed placement ---
    "feature_6_accurate_clip_placement": (
        'EXAMPLE (standard):\n'
        'INPUT: "US-guided biopsy performed. Clip placed: titanium marker. '
        'Location: at lesion site. Post-proc mammo: clip visible, no migration."\n'
        'OUTPUT: {"value": "Yes, titanium marker, at lesion site, no migration on post-procedure mammogram", '
        '"evidence": "Clip placed: titanium marker. Location: at lesion site. '
        'Post-proc mammo: clip visible, no migration.", "page_refs": [3], '
        '"confidence": 1.0, "reasoning_for_confidence": "Clip type, placement location, '
        'and post-procedure confirmation all explicitly stated."}'
    ),
}


def build_extraction_prompt(
    feature_name: str,
    retrieved_chunks: List[Chunk],
    display_name: str = "",
    use_few_shot: bool = True,
    use_cot: bool = True,
) -> List:
    from src.rag.retrievers import format_chunks_for_prompt

    chunks_text = format_chunks_for_prompt(retrieved_chunks)
    example = FEW_SHOT_EXAMPLES.get(feature_name, "") if use_few_shot else ""

    cot_instruction = (
        """
Before outputting JSON, reason through the following steps:
STEP 1 — LOCATE: Identify every chunk that mentions this feature.
STEP 2 — VERIFY EXPLICITNESS: Is the value directly stated? (Not implied or inferred.)
STEP 3 — CHECK FOR CONFLICTS: Do different chunks give contradictory values?
          If yes → value = "Indeterminate", include both snippets in evidence.
STEP 4 — PRESERVE EXACT WORDING: Copy measurement/units/terminology verbatim.
STEP 5 — CITE EVIDENCE: Record the exact supporting sentence and page reference.
If no chunk mentions this feature at all → value = "Not reported".
"""
        if use_cot
        else ""
    )

    human_content = f"""Extract: {display_name or feature_name}

INSTRUCTIONS:
- Extract {display_name or feature_name} using ONLY the retrieved chunks below.
- No inference. Verbatim fidelity. Preserve units.
- If absent: "Not reported". If conflicting: "Indeterminate" + both snippets.
{cot_instruction}
{f'EXAMPLE:{chr(10)}{example}{chr(10)}' if example else ''}
RETRIEVED CHUNKS:
{chunks_text}

OUTPUT SCHEMA (strict JSON only):
{OUTPUT_SCHEMA}"""

    return [
        SystemMessage(content=SYSTEM_IDENTITY),
        HumanMessage(content=human_content),
    ]


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/prompts/extraction_prompt_builder.py
# Purpose: Builds structured, anti-fabrication extraction prompts combining
#          system identity, output schema, CoT instructions, and few-shot
#          examples for clinical feature extraction from OCR text.
#
# Functions:
#   build_extraction_prompt(feature_name, chunks_text, prompt_id,
#                           few_shot_examples) -> List[BaseMessage]
#     Assembles a [SystemMessage, HumanMessage] list for LangChain LLM.
#     Loads prompt template from prompts/library/extraction_prompts.yaml
#     keyed by prompt_id (default: "rag_verify_v1").
#     Injects: feature_name, retrieved chunks text, few-shot examples.
#
# Outputs:
#   List[BaseMessage] ready for ChatAnthropic.invoke() or similar.
#
# Consumed by:
#   src/agents/extract_agent.py
#   src/agents/self_consistency_agent.py
# =============================================================================
