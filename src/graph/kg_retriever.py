"""
kg_retriever.py
Retrieves evidence from the knowledge graph by feature, verdict, or case.
Complements vector-store retrieval with structured graph traversal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_features_by_case(
    G: nx.DiGraph, case_id: str
) -> List[Dict[str, Any]]:
    features = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == "ClinicalFeature" and attrs.get("case_id") == case_id:
            features.append({"node_id": node_id, **attrs})
    return features


def get_fabrications(G: nx.DiGraph) -> List[Dict[str, Any]]:
    fabs = []
    for node_id, attrs in G.nodes(data=True):
        if (
            attrs.get("type") == "ClinicalFeature"
            and attrs.get("verdict") == "FABRICATION"
        ):
            fabs.append({"node_id": node_id, **attrs})
    return fabs


def get_evidence_for_feature(
    G: nx.DiGraph,
    feature_id: str,
) -> List[Dict[str, Any]]:
    evidence = []
    if feature_id not in G:
        return evidence
    for _, neighbor, edge_attrs in G.out_edges(feature_id, data=True):
        neighbor_attrs = G.nodes[neighbor]
        if neighbor_attrs.get("type") == "EvidenceChunk":
            evidence.append({
                "chunk_id": neighbor,
                "relation": edge_attrs.get("relation", ""),
                **neighbor_attrs,
            })
    return evidence


def get_verdict_for_claim(
    G: nx.DiGraph,
    claim_id: str,
) -> Optional[Dict[str, Any]]:
    for _, neighbor, edge_attrs in G.out_edges(claim_id, data=True):
        if edge_attrs.get("relation") == "VALIDATED_AS":
            return {"verdict_id": neighbor, **G.nodes[neighbor]}
    return None


def summarize_graph_stats(G: nx.DiGraph) -> Dict[str, int]:
    type_counts: Dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        **type_counts,
    }


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/graph/kg_retriever.py
# Purpose: Querying functions for the NetworkX evidence graph, enabling
#          case-, feature-, and verdict-level lookups.
#
# Functions:
#   get_features_by_case(G, case_id) -> List[dict]
#     Returns all ExtractionClaim node attribute dicts for a given case_id.
#
#   get_fabrications(G, case_id) -> List[dict]
#     Returns all ValidationVerdict nodes with verdict="FABRICATION" for
#     the case (or all cases if case_id is None).
#
#   get_evidence_for_feature(G, case_id, feature_name) -> List[dict]
#     Returns EvidenceChunk nodes linked to a specific case + feature claim.
#
#   summarize_graph_stats(G) -> dict
#     Returns n_nodes, n_edges, and per-type node counts.
#
# Consumed by:
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
#   fabrication_analysis/02_document_text_metrics.ipynb
# =============================================================================
