"""
graph_schema.py
Knowledge graph node and edge definitions for the clinical evidence graph.
Sits beside the vector store to provide structured evidence linkage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PatientNode:
    patient_id: str
    case_id: str


@dataclass
class SourceDocumentNode:
    document_id: str
    case_id: str
    modality: str
    date: Optional[str] = None


@dataclass
class ObservationNode:
    observation_id: str
    case_id: str
    modality: str


@dataclass
class ClinicalFeatureNode:
    feature_id: str
    case_id: str
    feature_name: str
    value: str
    confidence: float
    verdict: Optional[str] = None


@dataclass
class EvidenceChunkNode:
    chunk_id: str
    case_id: str
    document_id: str
    page_num: int
    text: str
    retrieval_score: Optional[float] = None


@dataclass
class ExtractionClaimNode:
    claim_id: str
    feature_id: str
    value: str
    model_id: str
    prompt_id: str
    run_id: str


@dataclass
class ValidationVerdictNode:
    verdict_id: str
    claim_id: str
    verdict: str
    verification_confidence: Optional[float] = None
    verification_quote: Optional[str] = None


@dataclass
class KnowledgeGraph:
    patients: List[PatientNode] = field(default_factory=list)
    documents: List[SourceDocumentNode] = field(default_factory=list)
    observations: List[ObservationNode] = field(default_factory=list)
    features: List[ClinicalFeatureNode] = field(default_factory=list)
    evidence_chunks: List[EvidenceChunkNode] = field(default_factory=list)
    claims: List[ExtractionClaimNode] = field(default_factory=list)
    verdicts: List[ValidationVerdictNode] = field(default_factory=list)


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/graph/graph_schema.py
# Purpose: Dataclass definitions for all node and edge types in the clinical
#          evidence knowledge graph built from pipeline results.
#
# Dataclasses (nodes):
#   PatientNode           - case_id, mrn, surgeon, patient_initials.
#   SourceDocumentNode    - doc_id, case_id, filename, modality, ocr_chars.
#   EvidenceChunkNode     - chunk_id, case_id, text, modality, page_num.
#   ClinicalFeatureNode   - feature_id, feature_name, display_name.
#   ExtractionClaimNode   - claim_id, case_id, feature_name, value,
#                           confidence, evidence, page_refs.
#   ValidationVerdictNode - verdict_id, case_id, feature_name, verdict,
#                           verification_confidence, verification_quote.
#   ObservationNode       - observation_id, case_id, feature_name, verdict.
#
# Dataclasses (containers):
#   KnowledgeGraph        - Holds lists of all node types for a batch run.
#
# Consumed by:
#   src/graph/build_graph.py   (populates nodes from pipeline results)
#   src/graph/kg_retriever.py  (queries node attributes)
# =============================================================================
