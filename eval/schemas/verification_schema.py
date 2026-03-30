"""
verification_schema.py
Pydantic schema for verification outputs and confidence rubric.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

CONFIDENCE_RUBRIC = {
    1.0: "Explicit, exact support found",
    0.8: "Explicit support with minor ambiguity",
    0.6: "Likely supported but partially fragmented",
    0.4: "Conflict or indirect support",
    0.0: "Not verifiable",
}


class VerificationResult(BaseModel):
    supported: bool
    exact_support_quote: Optional[str] = None
    page_ref: Optional[int] = None
    support_strength: Optional[Literal["direct", "indirect", "none"]] = None
    verification_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reason: Optional[str] = None
    verification_method: str = "rag_verification"

    def to_rubric_label(self) -> str:
        vc = self.verification_confidence
        if vc >= 1.0:
            return CONFIDENCE_RUBRIC[1.0]
        elif vc >= 0.8:
            return CONFIDENCE_RUBRIC[0.8]
        elif vc >= 0.6:
            return CONFIDENCE_RUBRIC[0.6]
        elif vc >= 0.4:
            return CONFIDENCE_RUBRIC[0.4]
        return CONFIDENCE_RUBRIC[0.0]
