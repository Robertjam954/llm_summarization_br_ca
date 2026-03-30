"""
hcat_schema.py
HCAT (Harm, Calibration, Accuracy, Traceability) safety schema
for the extraction pipeline evaluation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HCATScore(BaseModel):
    case_id: str
    run_id: str
    fabrication_rate: float = Field(ge=0.0, le=1.0)
    omission_rate: float = Field(ge=0.0, le=1.0)
    accuracy: float = Field(ge=0.0, le=1.0)
    verification_pass_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    evidence_traceability_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    self_consistency_agreement_rate: Optional[float] = None
    retrieval_precision_at_k: Optional[float] = None
    query_rewrite_success_rate: Optional[float] = None
    correction_accuracy: Optional[float] = None
    n_features: int = 0
    n_fabrications: int = 0
    n_omissions: int = 0
    n_correct: int = 0
    n_uncertain: int = 0

    @property
    def safety_score(self) -> float:
        return 1.0 - self.fabrication_rate

    def to_dict(self) -> Dict:
        return self.model_dump()


class HCATBatchReport(BaseModel):
    run_id: str
    prompt_id: str
    model_id: str
    n_cases: int
    scores: List[HCATScore]

    @property
    def mean_fabrication_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.fabrication_rate for s in self.scores) / len(self.scores)

    @property
    def mean_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.accuracy for s in self.scores) / len(self.scores)

    @property
    def mean_omission_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.omission_rate for s in self.scores) / len(self.scores)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : eval/schemas/hcat_schema.py
# Purpose: Pydantic models for the HCAT (Harm, Calibration, Accuracy,
#          Traceability) safety scoring framework applied to pipeline results.
#
# Classes:
#   HCATScore (BaseModel)
#     Per-case safety score. Fields: case_id, feature_name, verdict,
#     fabrication_flag, omission_flag, confidence, verification_confidence,
#     traceability_score, calibration_error, overall_risk.
#     Property: safety_score -> float (composite 0-1 safety score).
#
#   HCATBatchReport (BaseModel)
#     Aggregated batch report. Fields: run_id, n_cases, scores List[HCATScore].
#     Properties:
#       mean_fabrication_rate -> float
#       mean_omission_rate -> float
#       mean_safety_score -> float
#       high_risk_cases -> List[str] (cases with safety_score < 0.6)
#
# Consumed by:
#   eval/metrics/hcat_metrics.py
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
#   fabrication_analysis/02_document_text_metrics.ipynb
# =============================================================================
