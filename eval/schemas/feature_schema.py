"""
feature_schema.py
Pydantic schemas for feature extraction output validation.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class FeatureExtractionOutput(BaseModel):
    value: str = Field(..., description="Extracted feature value")
    evidence: str = Field(..., description="Verbatim supporting text")
    page_refs: List[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_for_confidence: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: object) -> str:
        return str(v) if v is not None else "Not reported"


class VerificationOutput(BaseModel):
    supported: bool
    exact_support_quote: Optional[str] = None
    page_ref: Optional[int] = None
    support_strength: Optional[Literal["direct", "indirect", "none"]] = None
    verification_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: Optional[str] = None


class FeatureResult(BaseModel):
    feature_name: str
    value: str
    evidence: str
    page_refs: List[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning_for_confidence: str = ""
    supported: Optional[bool] = None
    verification_quote: Optional[str] = None
    verification_confidence: Optional[float] = None
    verdict: Optional[Literal["CORRECT", "FABRICATION", "OMISSION", "UNCERTAIN"]] = None
    corrected_value: Optional[str] = None
    retrieval_attempts: int = 0
    verification_method: Optional[str] = None
