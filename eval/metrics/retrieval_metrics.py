"""
retrieval_metrics.py
Retrieval quality metrics: precision@k, retrieval success rate,
modality-specific retrieval analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def precision_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_page_refs: List[int],
    k: int,
) -> float:
    if not retrieved_chunks or not relevant_page_refs:
        return 0.0
    top_k = retrieved_chunks[:k]
    retrieved_pages = {c.get("page_num", -1) for c in top_k}
    relevant_set = set(relevant_page_refs)
    hits = len(retrieved_pages & relevant_set)
    return hits / k


def mean_precision_at_k(
    results: List[Dict[str, Any]],
    k: int = 3,
) -> float:
    scores = []
    for r in results:
        page_refs = r.get("page_refs", [])
        chunks = r.get("retrieved_chunks", [])
        if page_refs:
            scores.append(precision_at_k(chunks, page_refs, k))
    return sum(scores) / len(scores) if scores else 0.0


def retrieval_summary_by_feature(
    df: pd.DataFrame,
    feature_col: str = "feature_name",
    attempts_col: str = "retrieval_attempts",
    verdict_col: str = "verdict",
) -> pd.DataFrame:
    rows = []
    for feat, group in df.groupby(feature_col):
        total = len(group)
        mean_attempts = group[attempts_col].mean() if attempts_col in group else 1.0
        rewrite_rate = (
            (group[attempts_col] > 1).mean() if attempts_col in group else 0.0
        )
        verified = (group[verdict_col] == "CORRECT").sum()
        rows.append({
            "feature_name": feat,
            "n": total,
            "mean_retrieval_attempts": round(float(mean_attempts), 2),
            "rewrite_rate": round(float(rewrite_rate), 3),
            "verification_pass_rate": round(float(verified / total), 3),
        })
    return pd.DataFrame(rows).sort_values(
        "verification_pass_rate"
    ).reset_index(drop=True)


def compute_retrieval_stats(
    df: pd.DataFrame,
    attempts_col: str = "retrieval_attempts",
    v_confidence_col: str = "verification_confidence",
) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if attempts_col in df.columns:
        stats["mean_retrieval_attempts"] = float(df[attempts_col].mean())
        stats["rewrite_rate"] = float((df[attempts_col] > 1).mean())
    if v_confidence_col in df.columns:
        stats["mean_verification_confidence"] = float(
            df[v_confidence_col].dropna().mean()
        )
        stats["verification_pass_rate"] = float(
            (df[v_confidence_col].fillna(0) >= 0.8).mean()
        )
    return stats


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : eval/metrics/retrieval_metrics.py
# Purpose: RAG retrieval quality metrics — precision@k, rewrite rate,
#          verification pass rate, and modality-level breakdown.
#
# Functions:
#   precision_at_k(retrieved_chunks, relevant_keywords) -> float
#     Fraction of retrieved chunks containing at least one keyword.
#     Proxy for retrieval precision without ground-truth relevance labels.
#
#   mean_precision_at_k(results_df, chunk_col,
#                       keywords_col) -> float
#     Average precision@k over all rows in a results DataFrame.
#
#   retrieval_summary_by_feature(df, feature_col, attempts_col,
#                                k_col) -> pd.DataFrame
#     Per-feature mean retrieval_attempts, mean k, rewrite_rate.
#     rewrite_rate = fraction of cases that needed >1 retrieval pass.
#
#   compute_retrieval_stats(df) -> dict
#     Overall retrieval stats: mean_k, rewrite_rate, mean_retrieval_attempts,
#     mean_verification_confidence, verification_pass_rate (conf >= 0.8).
#
# Consumed by:
#   fabrication_analysis/02_document_text_metrics.ipynb
# =============================================================================
