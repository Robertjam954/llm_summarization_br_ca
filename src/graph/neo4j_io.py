"""
neo4j_io.py
Optional Neo4j export for the clinical knowledge graph.
Falls back gracefully if neo4j driver is not installed.
Primary path uses NetworkX + GraphML (no external DB required).
"""

from __future__ import annotations

from typing import Any, Dict

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def export_to_neo4j(
    kg_data: Dict[str, Any],
    uri: str,
    user: str,
    password: str,
) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning(
            "[neo4j_io] neo4j driver not installed. "
            "Install with: uv add neo4j"
        )
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for patient in kg_data.get("patients", []):
            session.run(
                "MERGE (p:Patient {patient_id: $patient_id, case_id: $case_id})",
                **patient,
            )
        for feat in kg_data.get("features", []):
            session.run(
                "MERGE (f:ClinicalFeature {feature_id: $feature_id}) "
                "SET f += $props",
                feature_id=feat["feature_id"],
                props=feat,
            )
        for verdict in kg_data.get("verdicts", []):
            session.run(
                "MERGE (v:ValidationVerdict {verdict_id: $verdict_id}) "
                "SET v += $props",
                verdict_id=verdict["verdict_id"],
                props=verdict,
            )
    driver.close()
    logger.info("[neo4j_io] export complete")


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/graph/neo4j_io.py
# Purpose: Optional export of the evidence knowledge graph to a Neo4j
#          graph database. Gracefully skips if the neo4j driver is absent.
#
# Functions:
#   export_to_neo4j(kg, uri, user, password) -> None
#     Connects to Neo4j via the bolt driver. MERGEs Patient, Feature,
#     Claim, EvidenceChunk, and ValidationVerdict nodes, then creates
#     directed relationships between them.
#     Skips silently if neo4j is not installed (ImportError handled).
#     Logs completion via logging_utils.
#
# Outputs:
#   Neo4j graph database nodes and relationships (no file output).
#   Requires: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables.
#
# Consumed by:
#   fabrication_analysis/01_langgraph_extraction_pipeline.ipynb (optional)
# =============================================================================
