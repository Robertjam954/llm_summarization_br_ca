"""
build_graph.py
Builds a NetworkX knowledge graph from extraction results.
Exports to GraphML for persistence and downstream analysis.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx

from src.graph.graph_schema import KnowledgeGraph
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_networkx_graph(kg: KnowledgeGraph) -> nx.DiGraph:
    G = nx.DiGraph()

    for p in kg.patients:
        G.add_node(p.patient_id, type="Patient", **vars(p))

    for d in kg.documents:
        G.add_node(d.document_id, type="SourceDocument", **vars(d))
        G.add_edge(d.case_id, d.document_id, relation="HAS_DOCUMENT")

    for f in kg.features:
        G.add_node(f.feature_id, type="ClinicalFeature", **vars(f))
        G.add_edge(f.case_id, f.feature_id, relation="HAS_FEATURE")

    for chunk in kg.evidence_chunks:
        G.add_node(chunk.chunk_id, type="EvidenceChunk", **vars(chunk))
        G.add_edge(chunk.document_id, chunk.chunk_id, relation="HAS_CHUNK")

    for claim in kg.claims:
        G.add_node(claim.claim_id, type="ExtractionClaim", **vars(claim))
        G.add_edge(claim.feature_id, claim.claim_id, relation="HAS_CLAIM")

    for verdict in kg.verdicts:
        G.add_node(verdict.verdict_id, type="ValidationVerdict", **vars(verdict))
        G.add_edge(
            verdict.claim_id, verdict.verdict_id, relation="VALIDATED_AS"
        )

    return G


def results_to_kg(
    case_results: List[Dict[str, Any]],
) -> KnowledgeGraph:
    from src.graph.graph_schema import (
        ClinicalFeatureNode,
        ExtractionClaimNode,
        PatientNode,
        ValidationVerdictNode,
    )

    kg = KnowledgeGraph()
    for result in case_results:
        case_id = result["case_id"]
        run_id = result.get("run_id", "")
        model_id = result.get("model_id", "")
        prompt_id = result.get("prompt_id", "")

        kg.patients.append(
            PatientNode(patient_id=case_id, case_id=case_id)
        )

        for feat_name, feat_data in result.get("features", {}).items():
            feat_id = f"{case_id}_{feat_name}"
            feat_node = ClinicalFeatureNode(
                feature_id=feat_id,
                case_id=case_id,
                feature_name=feat_name,
                value=feat_data.get("value", ""),
                confidence=feat_data.get("confidence", 0.0),
                verdict=feat_data.get("verdict"),
            )
            kg.features.append(feat_node)

            claim_id = str(uuid.uuid4())[:8]
            kg.claims.append(
                ExtractionClaimNode(
                    claim_id=claim_id,
                    feature_id=feat_id,
                    value=feat_data.get("value", ""),
                    model_id=model_id,
                    prompt_id=prompt_id,
                    run_id=run_id,
                )
            )

            verdict = feat_data.get("verdict")
            if verdict:
                kg.verdicts.append(
                    ValidationVerdictNode(
                        verdict_id=str(uuid.uuid4())[:8],
                        claim_id=claim_id,
                        verdict=verdict,
                        verification_confidence=feat_data.get(
                            "verification_confidence"
                        ),
                        verification_quote=feat_data.get("verification_quote"),
                    )
                )
    return kg


def save_graphml(G: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(path))
    logger.info(f"[graph] saved GraphML to {path}")


def load_graphml(path: Path) -> nx.DiGraph:
    return nx.read_graphml(str(path))


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/graph/build_graph.py
# Purpose: Builds a NetworkX DiGraph evidence graph from batch pipeline
#          results and saves/loads it as GraphML for downstream analysis.
#
# Functions:
#   results_to_kg(batch_results, run_id) -> KnowledgeGraph
#     Converts a list of case result dicts into a typed KnowledgeGraph
#     by creating Patient, Feature, Claim, and Verdict nodes.
#
#   build_networkx_graph(kg) -> nx.DiGraph
#     Creates a NetworkX DiGraph from a KnowledgeGraph. Adds nodes with
#     attribute dicts and directed edges (PATIENT->CLAIM, CLAIM->VERDICT,
#     CLAIM->EVIDENCE).
#
#   save_graphml(G, path) -> None
#     Writes graph to disk as GraphML at the given Path.
#
#   load_graphml(path) -> nx.DiGraph
#     Reads a GraphML file back into a NetworkX DiGraph.
#
# Outputs:
#   *.graphml file in data/knowledge_graph/{run_id}_kg.graphml
#
# Consumed by:
#   src/graph/kg_retriever.py
#   src/graph/neo4j_io.py
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb
# =============================================================================
