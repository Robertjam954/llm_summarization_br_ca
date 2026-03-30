"""
feature_queries.py
Central feature registry — single source of truth for all 13 clinical extraction targets.
Retrieval depth, criticality, and verification thresholds are synchronized here
so that prompts, retrieval, and evaluation stay consistent.
"""

from typing import Any, Dict, Set

FEATURES: Dict[str, Dict[str, Any]] = {
    "feature_1_lesion_size": {
        "display_name": "Lesion Size",
        "query": (
            "What is the lesion size, mass size, tumor size, or dimensions? "
            "Look for measurements in cm, mm, or any size descriptor."
        ),
        "k": 3,
        "k_second_pass": 7,
        "critical": True,
        "verification_threshold": 0.8,
        "modalities": ["mammogram", "ultrasound", "mri", "pathology"],
    },
    "feature_2_lesion_location": {
        "display_name": "Lesion Location",
        "query": (
            "Where is the lesion located? Laterality (left/right), quadrant, "
            "clock-face position, depth, distance from nipple."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.75,
        "modalities": ["mammogram", "ultrasound", "mri"],
    },
    "feature_3_calcifications_asymmetry": {
        "display_name": "Calcifications / Asymmetry",
        "query": (
            "Are there calcifications or asymmetry? Morphology (pleomorphic, amorphous, "
            "punctate), distribution (grouped, linear, segmental, regional, diffuse)."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.75,
        "modalities": ["mammogram"],
    },
    "feature_4_additional_enhancement_mri": {
        "display_name": "Additional Enhancement (MRI)",
        "query": (
            "What additional MRI enhancement is present beyond the primary lesion? "
            "Mass or non-mass enhancement, distribution, characteristics."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.7,
        "modalities": ["mri"],
    },
    "feature_5_extent": {
        "display_name": "Disease Extent",
        "query": (
            "What is the explicitly stated disease extent? "
            "Multifocal, multicentric, bilateral, or localized?"
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.75,
        "modalities": ["mammogram", "ultrasound", "mri", "pathology"],
    },
    "feature_6_accurate_clip_placement": {
        "display_name": "Clip Placement",
        "query": (
            "Was a biopsy clip or marker placed? What shape is the clip or marker? "
            "Where is it located relative to the lesion?"
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.7,
        "modalities": ["ultrasound", "mammogram"],
    },
    "feature_7_workup_recommendation": {
        "display_name": "Workup Recommendation",
        "query": (
            "What follow-up or workup is recommended? "
            "Biopsy, imaging follow-up, surgical evaluation, surveillance interval."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.7,
        "modalities": ["mammogram", "ultrasound", "mri"],
    },
    "feature_8_lymph_node": {
        "display_name": "Lymph Node Findings",
        "query": (
            "Are there lymph node findings, axillary adenopathy, or lymphadenopathy? "
            "Side-specific results or explicit 'no lymphadenopathy'."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.75,
        "modalities": ["ultrasound", "mri", "pathology"],
    },
    "feature_9_chronology_preserved": {
        "display_name": "Chronology Preserved",
        "query": (
            "Are radiologic studies ordered chronologically from oldest to most recent? "
            "Mammogram, ultrasound, MRI, PET-CT, bone scan dates."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.7,
        "modalities": ["mammogram", "ultrasound", "mri"],
    },
    "feature_10_biopsy_method": {
        "display_name": "Biopsy Method",
        "query": (
            "What biopsy technique was performed? Core needle biopsy, stereotactic, "
            "US-guided, MRI-guided, FNA, excisional biopsy, surgical."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.75,
        "modalities": ["pathology", "ultrasound"],
    },
    "feature_11_invasive_component_size_pathology": {
        "display_name": "Invasive Component Size (Pathology)",
        "query": (
            "What is the invasive component size in the pathology report? "
            "Invasive carcinoma size, invasive tumor dimensions in cm or mm."
        ),
        "k": 5,
        "k_second_pass": 7,
        "critical": True,
        "verification_threshold": 0.85,
        "modalities": ["pathology"],
    },
    "feature_12_histologic_diagnosis": {
        "display_name": "Histologic Diagnosis",
        "query": (
            "What is the histologic subtype or diagnosis? "
            "Invasive ductal, lobular, DCIS, mucinous, metaplastic, other histology."
        ),
        "k": 3,
        "k_second_pass": 5,
        "critical": False,
        "verification_threshold": 0.8,
        "modalities": ["pathology"],
    },
    "feature_13_receptor_status": {
        "display_name": "Receptor Status",
        "query": (
            "What are the ER, PR, HER2 IHC, HER2 ISH, and FISH results? "
            "Include category, percentage, intensity, controls, scoring, and comments."
        ),
        "k": 5,
        "k_second_pass": 7,
        "critical": True,
        "verification_threshold": 0.9,
        "modalities": ["pathology"],
    },
}

CRITICAL_FEATURES: Set[str] = {
    k for k, v in FEATURES.items() if v["critical"]
}

FEATURE_LIST = list(FEATURES.keys())

HIGH_RISK_SELF_CONSISTENCY: Set[str] = {
    "feature_11_invasive_component_size_pathology",
    "feature_13_receptor_status",
}

RETRIEVAL_K_MAP: Dict[str, int] = {
    k: v["k"] for k, v in FEATURES.items()
}

RETRIEVAL_K_SECOND_PASS_MAP: Dict[str, int] = {
    k: v["k_second_pass"] for k, v in FEATURES.items()
}

VERIFICATION_THRESHOLD_MAP: Dict[str, float] = {
    k: v["verification_threshold"] for k, v in FEATURES.items()
}


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/rag/feature_queries.py
# Purpose: Central registry for all 13 clinical extraction targets. Each
#          entry defines the retrieval query, k values, criticality flag,
#          verification threshold, and source modalities.
#
# Exports (data — no callable functions):
#   FEATURES                  - Dict[str, dict] with all 13 feature configs.
#     Keys per entry: display_name, query, k, k_second_pass, critical,
#                     verification_threshold, modalities.
#   CRITICAL_FEATURES         - Set[str] of features requiring strict checks.
#   HIGH_RISK_SELF_CONSISTENCY - Set[str] of features requiring 3-pass SC.
#   RETRIEVAL_K_MAP           - Dict[str, int] feature -> default k.
#   RETRIEVAL_K_SECOND_PASS_MAP - Dict[str, int] feature -> second-pass k.
#   VERIFICATION_THRESHOLD_MAP - Dict[str, float] feature -> min confidence.
#
# Consumed by:
#   src/rag/retrievers.py          (k values, queries)
#   src/workflows/extraction_graph.py (feature queue, SC list)
#   src/agents/verify_agent.py    (threshold map)
#   src/agents/self_consistency_agent.py (SC feature set)
#   src/workflows/extraction_state.py (initial feature_queue)
# =============================================================================
