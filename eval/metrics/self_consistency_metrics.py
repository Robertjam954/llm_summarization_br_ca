"""
self_consistency_metrics.py
Measures self-consistency agreement rates for high-risk features.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def sc_agreement_rate(
    df: pd.DataFrame,
    method_col: str = "verification_method",
    verdict_col: str = "verdict",
) -> Dict[str, Any]:
    sc_mask = df[method_col].str.startswith("self_consistency", na=False)
    sc_df = df[sc_mask]
    if sc_df.empty:
        return {"sc_total": 0, "sc_pass_rate": None, "sc_fail_rate": None}

    passed = (sc_df[method_col] == "self_consistency_passed").sum()
    failed = (sc_df[method_col] == "self_consistency_failed").sum()
    total = len(sc_df)

    return {
        "sc_total": total,
        "sc_pass_count": int(passed),
        "sc_fail_count": int(failed),
        "sc_pass_rate": float(passed / total) if total > 0 else None,
        "sc_fail_rate": float(failed / total) if total > 0 else None,
    }


def sc_by_feature(
    df: pd.DataFrame,
    feature_col: str = "feature_name",
    method_col: str = "verification_method",
) -> pd.DataFrame:
    sc_mask = df[method_col].str.startswith("self_consistency", na=False)
    sc_df = df[sc_mask]
    if sc_df.empty:
        return pd.DataFrame()

    rows = []
    for feat, group in sc_df.groupby(feature_col):
        total = len(group)
        passed = (group[method_col] == "self_consistency_passed").sum()
        failed = (group[method_col] == "self_consistency_failed").sum()
        rows.append({
            "feature_name": feat,
            "sc_runs": total,
            "sc_passed": int(passed),
            "sc_failed": int(failed),
            "sc_pass_rate": float(passed / total) if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : eval/metrics/self_consistency_metrics.py
# Purpose: Self-consistency pass/fail rates for the two high-risk features
#          (receptor status, invasive component size) that undergo 3-pass
#          majority-vote extraction.
#
# Functions:
#   sc_agreement_rate(df, method_col) -> dict
#     Computes overall SC pass rate (fraction of rows where
#     verification_method == "self_consistency_pass").
#     Returns: total_sc_runs, sc_passed, sc_failed, sc_pass_rate.
#
#   sc_by_feature(df, feature_col, method_col) -> pd.DataFrame
#     Per-feature SC stats: sc_runs, sc_passed, sc_failed, sc_pass_rate.
#     Filtered to only rows where SC was invoked (method_col contains
#     "self_consistency").
#
# Consumed by:
#   eval/metrics/hcat_metrics.py
#   fabrication_analysis/02_document_text_metrics.ipynb
# =============================================================================
