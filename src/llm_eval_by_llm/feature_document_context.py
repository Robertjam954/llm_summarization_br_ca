# -*- coding: utf-8 -*-
"""
feature_document_context.py
============================
Defines the canonical source-document context for each of the 14 clinical
feature elements extracted in this study.

Used by:
  - 04b_text_consolidation_per_case.ipynb  — section routing
  - source_document_feature_extraction_v*.py — prompt construction
  - deepeval_multi_model_pipeline.py        — per-feature context injection

Structure of each entry
-----------------------
  key               : str  — matches ELEMENTS / ELEMENT_TRIPLES keys throughout the codebase
  display           : str  — human-readable label (matches validation datasheet column header)
  domain            : str  — "radiology" | "pathology" | "hpi" | "mixed"
  primary_sections  : list[str]
      Ordered list of consolidated-text section headers to include in the LLM context window.
      Values match the section keys in 04b: "hpi", "radiology", "pathology", "genetics"
  primary_doc_keywords : list[str]
      Filename keywords (lowercase) that identify the most informative individual documents.
      Used to rank/filter documents when context window is limited.
  secondary_doc_keywords : list[str]
      Supplementary documents that may corroborate or clarify the feature.
  value_type        : str  — "binary" | "categorical" | "numeric" | "text"
  expected_values   : str  — plain-language description of valid values
  extraction_hint   : str
      Concise instruction for the LLM about where in the source text to find this feature
      and what to look for. Written in second-person imperative for direct prompt use.
  fabrication_risk  : str  — "low" | "medium" | "high"
      Qualitative estimate of how likely an LLM is to hallucinate this feature.
"""

from __future__ import annotations
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Main context dictionary
# ---------------------------------------------------------------------------

FEATURE_DOCUMENT_CONTEXT: Dict[str, Dict[str, Any]] = {

    # ── 1. Lesion Size ───────────────────────────────────────────────────────
    "lesion_size": {
        "display":               "Lesion Size",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_us", "imaging_mri", "imaging_internal"],
        "secondary_doc_keywords":["imaging_mammo2", "imaging_us2", "imaging_mammo_post",
                                  "path_biopsy", "path_surgical"],
        "value_type":            "numeric",
        "expected_values":       (
            "Numeric measurement in mm or cm (e.g., '12 mm', '1.8 cm'). "
            "Use the largest single dimension from the most recent pre-biopsy imaging. "
            "If multiple modalities are available, prefer MRI > US > mammography. "
            "Record as absent if no size measurement is documented."
        ),
        "extraction_hint":       (
            "Look in radiology reports (mammogram, ultrasound, MRI) for the reported lesion "
            "size or mass measurement. The most informative value is usually found in the "
            "'Findings' or 'Impression' section as a numeric measurement followed by 'mm' or 'cm'. "
            "If multiple sizes are reported across modalities, extract the most recently dated "
            "measurement from the highest-resolution modality (MRI > ultrasound > mammogram)."
        ),
        "fabrication_risk":      "medium",
    },

    # ── 2. Lesion Laterality ─────────────────────────────────────────────────
    "lesion_laterality": {
        "display":               "Lesion Laterality",
        "domain":                "mixed",
        "primary_sections":      ["radiology", "hpi"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_us", "imaging_internal",
                                  "imaging_mri", "hpi_human", "hpi_ai"],
        "secondary_doc_keywords":["path_biopsy", "path_surgical"],
        "value_type":            "categorical",
        "expected_values":       (
            "'left', 'right', or 'bilateral'. "
            "Confirmed across at least two independent source documents when possible."
        ),
        "extraction_hint":       (
            "Look for 'left breast', 'right breast', or 'bilateral' in any radiology report "
            "or HPI. Laterality is typically stated at the beginning of the Findings section "
            "or in the clinical indication. Corroborate across imaging and HPI documents."
        ),
        "fabrication_risk":      "low",
    },

    # ── 3. Lesion Location ───────────────────────────────────────────────────
    "lesion_location": {
        "display":               "Lesion Location",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_us", "imaging_internal",
                                  "imaging_mri"],
        "secondary_doc_keywords":["imaging_mammo2", "imaging_us2", "hpi_human", "hpi_ai"],
        "value_type":            "text",
        "expected_values":       (
            "Clock-face position (e.g., '2 o'clock'), quadrant (UOQ, UIQ, LOQ, LIQ), "
            "sub-areolar, or axillary tail. Include distance from nipple if documented "
            "(e.g., '2 o'clock, 5 cm from nipple')."
        ),
        "extraction_hint":       (
            "Look for clock-face notation (e.g., '10 o'clock position'), quadrant descriptions "
            "(upper outer, lower inner, etc.), or anatomical region descriptions in the "
            "Findings section of mammogram and ultrasound reports. "
            "MRI reports may describe location by zone or quadrant."
        ),
        "fabrication_risk":      "medium",
    },

    # ── 4. Calcifications / Asymmetry ────────────────────────────────────────
    "calcifications_asymmetry": {
        "display":               "Calcifications / Asymmetry",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_mammo2", "imaging_mammo3",
                                  "imaging_internal"],
        "secondary_doc_keywords":["imaging_mri", "imaging_us"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) or absent (0). "
            "Present if the mammogram documents calcifications (fine linear, segmental, grouped, "
            "pleomorphic, etc.) OR an asymmetry (focal, global, developing asymmetry). "
            "Absent if no calcifications or asymmetry are described."
        ),
        "extraction_hint":       (
            "Look specifically in mammogram reports (not ultrasound or MRI) for descriptions of "
            "calcifications — fine, pleomorphic, linear, segmental, grouped — or for terms like "
            "'asymmetry', 'focal asymmetry', 'developing asymmetry', 'architectural distortion'. "
            "These features are mammography-specific and should NOT be inferred from MRI or US."
        ),
        "fabrication_risk":      "medium",
    },

    # ── 5. Additional Enhancement (MRI) ──────────────────────────────────────
    "additional_enhancement_mri": {
        "display":               "Additional Enhancement (MRI)",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mri"],
        "secondary_doc_keywords":["imaging_mammo_post", "imaging_internal"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) if MRI documents additional enhancing lesions, satellite foci, "
            "non-mass enhancement (NME), or ipsilateral/contralateral additional findings "
            "beyond the index lesion. Absent (0) if MRI shows only the index lesion or no MRI "
            "was performed (mark as absent/not applicable)."
        ),
        "extraction_hint":       (
            "Look ONLY in MRI reports. Search for terms such as 'additional enhancement', "
            "'satellite lesion', 'non-mass enhancement (NME)', 'additional focus/foci', "
            "'ipsilateral finding', 'contralateral finding', 'background parenchymal enhancement'. "
            "If no MRI report is present in the source documents, code as absent. "
            "Do NOT infer MRI findings from mammogram or ultrasound reports."
        ),
        "fabrication_risk":      "high",
    },

    # ── 6. Extent ────────────────────────────────────────────────────────────
    "extent": {
        "display":               "Extent",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_mri", "imaging_us",
                                  "imaging_internal"],
        "secondary_doc_keywords":["imaging_mammo2", "path_surgical"],
        "value_type":            "numeric",
        "expected_values":       (
            "Numeric span of disease in mm or cm (e.g., '35 mm extent of calcifications', "
            "'4 cm span of NME on MRI'). "
            "Record as absent if no extent measurement is documented separately from lesion size."
        ),
        "extraction_hint":       (
            "Look for terms like 'span', 'extent', 'field', 'distribution' when describing the "
            "overall area of disease — distinct from the index lesion size. "
            "Commonly reported for calcification clusters on mammography (e.g., 'calcifications "
            "spanning 3 cm') or non-mass enhancement on MRI (e.g., 'segmental NME extending "
            "over 4 cm'). Extent may appear in the Impression rather than Findings."
        ),
        "fabrication_risk":      "high",
    },

    # ── 7. Accurate Clip Placement ───────────────────────────────────────────
    "accurate_clip_placement": {
        "display":               "Accurate Clip Placement",
        "domain":                "radiology",
        "primary_sections":      ["radiology"],
        "primary_doc_keywords":  ["imaging_mammo_post", "imaging_mammo_postprocedure",
                                  "imaging_mammo_postbx", "imaging_mammo2"],
        "secondary_doc_keywords":["imaging_internal", "imaging_mammo"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) if post-biopsy mammogram confirms the biopsy clip/marker is correctly "
            "placed within or immediately adjacent to the sampled lesion. "
            "Absent (0) if the clip is displaced, not visualized, or no post-biopsy mammogram "
            "is available."
        ),
        "extraction_hint":       (
            "Look specifically in POST-PROCEDURE or POST-BIOPSY mammogram reports for confirmation "
            "of clip/marker placement. Key phrases: 'clip is seen within', 'marker is in place', "
            "'biopsy marker', 'post-biopsy clip', 'titanium marker', 'clip placement is accurate'. "
            "A displaced clip is still coded as present (the feature was addressed), but note "
            "displacement. If only pre-biopsy imaging is available, code as absent."
        ),
        "fabrication_risk":      "high",
    },

    # ── 8. Workup Recommendation ─────────────────────────────────────────────
    "workup_recommendation": {
        "display":               "Workup Recommendation",
        "domain":                "mixed",
        "primary_sections":      ["radiology", "hpi"],
        "primary_doc_keywords":  ["imaging_mammo", "imaging_us", "imaging_internal",
                                  "imaging_mri", "hpi_human", "hpi_ai"],
        "secondary_doc_keywords":["imaging_mammo2", "imaging_mammo_post"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) if any radiology report or HPI explicitly states a recommended next step "
            "beyond the current study (e.g., biopsy recommended, MRI recommended, short-interval "
            "follow-up, BI-RADS 4/5 assessment with biopsy recommendation). "
            "Absent (0) if only routine follow-up (BI-RADS 1/2/3) or no recommendation is stated."
        ),
        "extraction_hint":       (
            "Look in the Impression section or Assessment/Recommendation section of any radiology "
            "report for BI-RADS categories 4A, 4B, 4C, or 5 with an explicit recommendation for "
            "tissue sampling or additional workup. Also check HPI for phrases like 'recommended "
            "biopsy', 'advised MRI', 'suggested additional imaging'. "
            "BI-RADS 3 (probably benign) with a 6-month follow-up recommendation should be coded "
            "as present only if the follow-up is explicitly stated as a specific action plan."
        ),
        "fabrication_risk":      "medium",
    },

    # ── 9. Lymph Node ────────────────────────────────────────────────────────
    "lymph_node": {
        "display":               "Lymph Node",
        "domain":                "mixed",
        "primary_sections":      ["radiology", "pathology", "hpi"],
        "primary_doc_keywords":  ["imaging_us", "imaging_mri", "imaging_internal",
                                  "path_surgical", "path_biopsy", "hpi_human", "hpi_ai"],
        "secondary_doc_keywords":["imaging_mammo", "imaging_bs", "imaging_ctap"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) if any lymph node assessment (imaging or pathology) is documented — "
            "whether normal, abnormal, or indeterminate. "
            "Absent (0) if no lymph node assessment is mentioned anywhere in the source documents."
        ),
        "extraction_hint":       (
            "Search across ALL document types. In radiology: look for axillary lymph node "
            "description in ultrasound or MRI reports ('axillary lymph nodes appear normal', "
            "'suspicious lymph node', 'lymph node with cortical thickening'). "
            "In pathology: look for sentinel lymph node biopsy results or axillary dissection "
            "findings. In HPI: look for any mention of 'lymph node', 'axillary', 'sentinel'. "
            "Breast imaging MRI reports often include axillary assessment in the Findings section."
        ),
        "fabrication_risk":      "medium",
    },

    # ── 10. Chronology Preserved ─────────────────────────────────────────────
    "chronology_preserved": {
        "display":               "Chronology Preserved",
        "domain":                "hpi",
        "primary_sections":      ["hpi"],
        "primary_doc_keywords":  ["hpi_ai", "hpi_human"],
        "secondary_doc_keywords":["imaging_internal", "imaging_mammo"],
        "value_type":            "binary",
        "expected_values":       (
            "Present (1) if the HPI (either AI-generated or human-written) presents the clinical "
            "course in the correct temporal order: symptom onset → initial imaging → biopsy → "
            "pathology results → additional workup (if applicable). "
            "Absent (0) if the chronological sequence is disrupted, events are out of order, or "
            "no clear temporal structure can be identified."
        ),
        "extraction_hint":       (
            "Look ONLY in HPI documents (hpi_ai.docx, hpi_human.docx). Assess whether the "
            "narrative follows a logical timeline. Key markers of preserved chronology: "
            "use of temporal connectors ('initially', 'subsequently', 'at that time', 'following "
            "biopsy'), dates in sequence, and logical cause-effect ordering (e.g., imaging "
            "identified lesion → biopsy performed → results returned). "
            "Do NOT extract from radiology or pathology reports."
        ),
        "fabrication_risk":      "low",
    },

    # ── 11. Biopsy Method ────────────────────────────────────────────────────
    "biopsy_method": {
        "display":               "Biopsy Method",
        "domain":                "mixed",
        "primary_sections":      ["pathology", "radiology"],
        "primary_doc_keywords":  ["path_biopsy", "path_biopsy_internal", "path_biopsy_external"],
        "secondary_doc_keywords":["imaging_us", "imaging_mammo_post", "imaging_mammo_postbx",
                                  "imaging_mammo_postprocedure", "hpi_human", "hpi_ai"],
        "value_type":            "categorical",
        "expected_values":       (
            "One of: 'ultrasound-guided core needle biopsy (US-CNB)', "
            "'stereotactic core needle biopsy (ST-CNB)', 'MRI-guided biopsy', "
            "'vacuum-assisted biopsy (VAB)', 'fine needle aspiration (FNA)', "
            "'excisional biopsy', 'surgical biopsy'. "
            "Present (1 in binary coding) if any biopsy method is documented."
        ),
        "extraction_hint":       (
            "Look primarily in pathology reports in the 'Clinical History', 'Procedure', or "
            "'Specimen' section for the biopsy technique. Key terms: 'ultrasound-guided', "
            "'stereotactic', 'MRI-guided', 'core needle', 'vacuum-assisted', 'excision', "
            "'14-gauge', '9-gauge'. Also check post-procedure mammogram reports which often "
            "include a clinical indication line stating the biopsy type. "
            "HPI may also mention the biopsy technique."
        ),
        "fabrication_risk":      "low",
    },

    # ── 12. Invasive Component Size (Pathology) ───────────────────────────────
    "invasive_component_size": {
        "display":               "Invasive Component Size (Pathology)",
        "domain":                "pathology",
        "primary_sections":      ["pathology"],
        "primary_doc_keywords":  ["path_biopsy", "path_surgical", "path_biopsy_internal",
                                  "path_biopsy_external", "path_surgical_internal",
                                  "path_surgical_external"],
        "secondary_doc_keywords":[],
        "value_type":            "numeric",
        "expected_values":       (
            "Numeric measurement in mm or cm as reported on final pathology "
            "(e.g., '18 mm invasive ductal carcinoma', '2.2 cm invasive lobular carcinoma'). "
            "This is the pathologic (not radiologic) size. "
            "Absent if only DCIS is present (no invasive component) or no measurement is given."
        ),
        "extraction_hint":       (
            "Look ONLY in pathology reports (biopsy or surgical specimen). "
            "Search for 'invasive carcinoma measures', 'invasive component', 'tumor size', "
            "'greatest dimension' followed by a measurement. "
            "Distinguish from DCIS span (which is not the invasive component). "
            "The Synoptic Summary or Final Diagnosis section of the surgical pathology report "
            "is the most reliable location. Do NOT use radiologic size from imaging reports."
        ),
        "fabrication_risk":      "high",
    },

    # ── 13. Histologic Diagnosis ─────────────────────────────────────────────
    "histologic_diagnosis": {
        "display":               "Histologic Diagnosis",
        "domain":                "pathology",
        "primary_sections":      ["pathology"],
        "primary_doc_keywords":  ["path_biopsy", "path_biopsy_internal", "path_biopsy_external",
                                  "path_surgical", "path_surgical_internal",
                                  "path_surgical_external", "path_bopsy"],
        "secondary_doc_keywords":["hpi_human", "hpi_ai"],
        "value_type":            "categorical",
        "expected_values":       (
            "One of: 'invasive ductal carcinoma (IDC) / invasive carcinoma of no special type (NST)', "
            "'invasive lobular carcinoma (ILC)', 'DCIS (ductal carcinoma in situ)', "
            "'invasive ductal carcinoma with DCIS', 'mixed IDC/ILC', "
            "'mucinous carcinoma', 'tubular carcinoma', 'other'. "
            "Present (1 in binary coding) if any malignant histologic diagnosis is documented."
        ),
        "extraction_hint":       (
            "Look in the 'Final Diagnosis' or 'Diagnosis' section of pathology reports. "
            "The histologic type appears immediately after specimen designation, e.g., "
            "'Right breast, core biopsy: INVASIVE DUCTAL CARCINOMA, grade 2'. "
            "Also check the Synoptic Summary section for WHO classification. "
            "HPI may paraphrase the diagnosis but should not be used as the primary source — "
            "always verify against the pathology report."
        ),
        "fabrication_risk":      "low",
    },

    # ── 14. Receptor Status ──────────────────────────────────────────────────
    "receptor_status": {
        "display":               "Receptor Status",
        "domain":                "pathology",
        "primary_sections":      ["pathology", "genetics"],
        "primary_doc_keywords":  ["path_biopsy", "path_biopsy_internal", "path_biopsy_external",
                                  "path_surgical", "biomarkers", "receptor"],
        "secondary_doc_keywords":["genetics", "hpi_human", "hpi_ai"],
        "value_type":            "categorical",
        "expected_values":       (
            "Sub-components: ER (positive/negative + % staining), PR (positive/negative + %), "
            "HER2 IHC (0, 1+, 2+, 3+), HER2 ISH (amplified/not amplified, copy number ratio). "
            "Present (1 in binary coding) if any receptor result is documented. "
            "HER2 equivocal (2+ IHC) cases should trigger ISH/FISH lookup."
        ),
        "extraction_hint":       (
            "Look for receptor panel results in pathology reports. The receptor block is usually "
            "a separate addendum or appears under 'Ancillary Studies', 'Immunohistochemistry', "
            "or 'Special Studies' in the biopsy or surgical pathology report. "
            "Key markers: 'ER', 'PR', 'HER2', 'FISH', 'ISH', 'Ki-67'. "
            "ER/PR are reported as positive/negative with percentage (e.g., 'ER: positive, 95%'). "
            "HER2 IHC is scored 0–3+; if 2+, look for an ISH/FISH result in a separate report. "
            "A dedicated biomarkers document (e.g., 'biomarkers_AO.pdf') may contain the full "
            "receptor panel separately from the main pathology report."
        ),
        "fabrication_risk":      "medium",
    },
}


# ---------------------------------------------------------------------------
# Derived lookup helpers
# ---------------------------------------------------------------------------

# Features grouped by primary domain
RADIOLOGY_FEATURES = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["domain"] in ("radiology",)
]

PATHOLOGY_FEATURES = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["domain"] == "pathology"
]

HPI_FEATURES = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["domain"] == "hpi"
]

MIXED_FEATURES = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["domain"] == "mixed"
]

# Feature keys by fabrication risk level
HIGH_FABRICATION_RISK = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["fabrication_risk"] == "high"
]

MEDIUM_FABRICATION_RISK = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["fabrication_risk"] == "medium"
]

LOW_FABRICATION_RISK = [
    k for k, v in FEATURE_DOCUMENT_CONTEXT.items()
    if v["fabrication_risk"] == "low"
]

# Flat map: feature_key → display name (convenience alias matching ELEMENTS)
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    k: v["display"] for k, v in FEATURE_DOCUMENT_CONTEXT.items()
}

# Flat map: feature_key → primary_sections
FEATURE_SECTIONS: Dict[str, list] = {
    k: v["primary_sections"] for k, v in FEATURE_DOCUMENT_CONTEXT.items()
}

# Flat map: feature_key → extraction_hint (for direct prompt injection)
EXTRACTION_HINTS: Dict[str, str] = {
    k: v["extraction_hint"] for k, v in FEATURE_DOCUMENT_CONTEXT.items()
}


def get_context_for_feature(feature_key: str) -> Dict[str, Any]:
    """Return the full context dict for a single feature key."""
    if feature_key not in FEATURE_DOCUMENT_CONTEXT:
        raise KeyError(
            f"Unknown feature key: '{feature_key}'. "
            f"Valid keys: {list(FEATURE_DOCUMENT_CONTEXT.keys())}"
        )
    return FEATURE_DOCUMENT_CONTEXT[feature_key]


def get_relevant_sections(feature_key: str) -> list:
    """Return the ordered list of consolidated-text sections relevant to a feature."""
    return FEATURE_DOCUMENT_CONTEXT[feature_key]["primary_sections"]


def get_extraction_hint(feature_key: str) -> str:
    """Return the LLM extraction hint for a feature, ready for prompt injection."""
    return FEATURE_DOCUMENT_CONTEXT[feature_key]["extraction_hint"]


def filter_text_to_relevant_sections(
    consolidated_text: str,
    feature_key: str,
    section_header_map: dict | None = None,
) -> str:
    """
    Given a consolidated case text (output of NB04b) and a feature key,
    return only the section(s) of the text that are relevant to that feature.

    Parameters
    ----------
    consolidated_text : str
        Full consolidated text from extracted_text_consolidated/{case_id}.txt
    feature_key : str
        One of the keys in FEATURE_DOCUMENT_CONTEXT
    section_header_map : dict, optional
        Maps section key ("hpi", "radiology", "pathology", "genetics") to the
        exact header string used in the consolidated file.
        Defaults to the headers defined in NB04b.

    Returns
    -------
    str
        Filtered text containing only the relevant sections.
    """
    if section_header_map is None:
        section_header_map = {
            "hpi":       "=== HPI / CLINICAL NOTES ===",
            "radiology": "=== RADIOLOGY REPORTS ===",
            "pathology": "=== PATHOLOGY REPORTS ===",
            "genetics":  "=== GENETICS / MOLECULAR ===",
        }

    relevant_sections = get_relevant_sections(feature_key)
    relevant_headers = [
        section_header_map[s]
        for s in relevant_sections
        if s in section_header_map
    ]

    if not relevant_headers:
        return consolidated_text

    # All known headers (for splitting)
    all_headers = list(section_header_map.values())

    # Build a regex split pattern on all section headers
    import re
    split_pattern = "|".join(re.escape(h) for h in all_headers)
    parts = re.split(f"({split_pattern})", consolidated_text)

    # Reconstruct only the relevant sections
    output_blocks = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part in relevant_headers:
            # Include this header + the content that follows
            block = part
            if i + 1 < len(parts) and parts[i + 1] not in all_headers:
                block += parts[i + 1]
                i += 1
            output_blocks.append(block.strip())
        i += 1

    return "\n\n".join(output_blocks)


# ---------------------------------------------------------------------------
# Quick summary for inspection / debugging
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Total features defined : {len(FEATURE_DOCUMENT_CONTEXT)}")
    print(f"Radiology features     : {RADIOLOGY_FEATURES}")
    print(f"Pathology features     : {PATHOLOGY_FEATURES}")
    print(f"HPI features           : {HPI_FEATURES}")
    print(f"Mixed features         : {MIXED_FEATURES}")
    print()
    print(f"High fabrication risk  : {HIGH_FABRICATION_RISK}")
    print(f"Medium fabrication risk: {MEDIUM_FABRICATION_RISK}")
    print(f"Low fabrication risk   : {LOW_FABRICATION_RISK}")
    print()
    for key, ctx in FEATURE_DOCUMENT_CONTEXT.items():
        print(f"  [{ctx['fabrication_risk'].upper():6}] {key:40s} → {ctx['primary_sections']}")
