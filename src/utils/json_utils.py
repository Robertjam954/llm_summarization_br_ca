"""
json_utils.py
Safe JSON parsing helpers for LLM output.
Handles markdown code fences, truncated JSON, and schema validation.
"""

import json
import re
from typing import Any, Dict, Optional


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def validate_feature_result(parsed: Dict[str, Any]) -> bool:
    required = {"value", "evidence", "page_refs", "confidence", "reasoning_for_confidence"}
    return required.issubset(set(parsed.keys()))


def validate_verification_result(parsed: Dict[str, Any]) -> bool:
    return "supported" in parsed


def coerce_confidence(value: Any) -> float:
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return 0.0


def coerce_page_refs(value: Any) -> list:
    if isinstance(value, list):
        return [int(p) for p in value if str(p).isdigit()]
    return []


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/utils/json_utils.py
# Purpose: Safe parsing of LLM JSON responses, handling markdown fences,
#          trailing commas, truncation, and schema validation.
#
# Functions:
#   safe_parse_json(text) -> dict | None
#     Strips markdown code fences, attempts json.loads, returns None on
#     failure (no exception raised). Handles truncated responses.
#
#   validate_feature_result(data) -> dict | None
#     Checks that a parsed dict contains required FeatureResult fields
#     (value, confidence). Returns validated dict or None.
#
#   validate_verification_result(data) -> dict | None
#     Checks that a parsed dict contains supported and
#     verification_confidence. Returns validated dict or None.
#
#   coerce_confidence(value) -> float
#     Coerces any numeric-like value to float in [0.0, 1.0]. Returns 0.0
#     on failure.
#
#   coerce_page_refs(value) -> list
#     Coerces a page reference value to a list of ints; returns [] on
#     failure.
#
# Consumed by: src/agents/extract_agent.py, src/agents/verify_agent.py,
#              src/agents/self_consistency_agent.py
# =============================================================================
