"""
hcat_metrics.py
Computes the full HCAT safety report for a batch run.
Aggregates fabrication, omission, accuracy, traceability, and SC metrics.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from eval.metrics.fabrication_metrics import compute_fabrication_rate
from eval.metrics.retrieval_metrics import compute_retrieval_stats
from eval.metrics.self_consistency_metrics import sc_agreement_rate
from eval.schemas.hcat_schema import HCATBatchReport, HCATScore


def compute_hcat_score(
    case_df: pd.DataFrame,
    case_id: str,
    run_id: str,
) -> HCATScore:
    total = len(case_df)
    fab_stats = compute_fabrication_rate(case_df)
    retrieval_stats = compute_retrieval_stats(case_df)
    sc_stats = sc_agreement_rate(case_df)

    verdict_col = "verdict"
    correct = (case_df[verdict_col] == "CORRECT").sum() if verdict_col in case_df else 0
    omissions = (case_df[verdict_col] == "OMISSION").sum() if verdict_col in case_df else 0
    uncertain = (case_df[verdict_col] == "UNCERTAIN").sum() if verdict_col in case_df else 0
    fabrications = fab_stats.get("fabrication_count", 0)

    v_conf = case_df.get("verification_confidence", pd.Series(dtype=float))
    v_pass_rate = float(
        (v_conf.fillna(0) >= 0.8).mean()
    ) if len(v_conf) > 0 else 0.0

    evidence_rate = float(
        case_df["verification_quote"].notna().mean()
    ) if "verification_quote" in case_df else 0.0

    return HCATScore(
        case_id=case_id,
        run_id=run_id,
        fabrication_rate=fab_stats.get("fabrication_rate", 0.0) or 0.0,
        omission_rate=float(omissions / total) if total > 0 else 0.0,
        accuracy=float(correct / total) if total > 0 else 0.0,
        verification_pass_rate=v_pass_rate,
        evidence_traceability_rate=evidence_rate,
        self_consistency_agreement_rate=sc_stats.get("sc_pass_rate"),
        retrieval_precision_at_k=retrieval_stats.get("mean_verification_confidence"),
        n_features=total,
        n_fabrications=int(fabrications),
        n_omissions=int(omissions),
        n_correct=int(correct),
        n_uncertain=int(uncertain),
    )


def compute_batch_hcat(
    df: pd.DataFrame,
    run_id: str,
    prompt_id: str,
    model_id: str,
    case_id_col: str = "case_id",
) -> HCATBatchReport:
    scores: List[HCATScore] = []
    for case_id, case_df in df.groupby(case_id_col):
        score = compute_hcat_score(case_df, str(case_id), run_id)
        scores.append(score)

    return HCATBatchReport(
        run_id=run_id,
        prompt_id=prompt_id,
        model_id=model_id,
        n_cases=len(scores),
        scores=scores,
    )


def hcat_report_to_df(report: HCATBatchReport) -> pd.DataFrame:
    rows = [s.to_dict() for s in report.scores]
    return pd.DataFrame(rows)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : eval/metrics/hcat_metrics.py
# Purpose: Computes the full HCAT (Harm, Calibration, Accuracy, Traceability)
#          safety report for individual cases and entire batch runs.
#
# Functions:
#   compute_hcat_score(case_result, feature_name) -> HCATScore
#     Derives a per-(case, feature) HCATScore from a case result dict.
#     Computes: fabrication_flag, omission_flag, traceability_score
#     (based on verification_quote presence), calibration_error
#     (|confidence - verification_confidence|), overall_risk.
#
#   compute_batch_hcat(batch_results, run_id) -> HCATBatchReport
#     Runs compute_hcat_score over all cases and features in a batch.
#     Returns an HCATBatchReport with aggregated statistics.
#
#   hcat_report_to_df(report) -> pd.DataFrame
#     Flattens an HCATBatchReport into a tidy DataFrame via .to_dict()
#     on each HCATScore. One row per (case, feature).
#
# Outputs:
#   HCATBatchReport with mean_fabrication_rate, mean_safety_score,
#   high_risk_cases list. DataFrame for plotting.
#
# Consumed by:
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
#   fabrication_analysis/02_document_text_metrics.ipynb
# =============================================================================
