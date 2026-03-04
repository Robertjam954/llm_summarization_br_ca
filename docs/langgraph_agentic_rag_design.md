# LangGraph Agentic RAG Pipeline Design
**Memorial Sloan Kettering | Goel Lab**  
**Date:** March 2026

---

## Overview

This document outlines the design for two new project layers:
1. **HCAT Safety Metrics Framework** — Assess residual PHI risk after deidentification
2. **LangGraph Agentic RAG Pipeline** — Mitigate LLM fabrication via knowledge graph validation

---

## 1. HCAT Safety Metrics Framework (NB11)

### 1.1 Purpose
Quantify the safety of our deidentification pipeline using the **HCAT (HIPAA Compliance Assessment Tool)** framework, with a focus on measuring **residual PHI risk** after all deidentification steps.

### 1.2 HCAT Metrics

| Metric | Definition | Calculation |
|--------|------------|-------------|
| **PHI Recall** | % of PHI instances successfully redacted | `redacted_phi / total_phi_instances` |
| **False Negative Rate** | % of PHI missed by redaction | `1 - PHI_recall` |
| **Over-redaction Rate** | % of non-PHI text incorrectly redacted | `false_positives / total_non_phi_tokens` |
| **Residual PHI Risk Score** | Weighted risk score based on PHI type severity | `Σ(missed_phi_count × severity_weight)` |
| **Re-identification Risk** | Probability of patient re-identification from residual PHI | Computed via k-anonymity, l-diversity metrics |

### 1.3 PHI Severity Weights

```python
PHI_SEVERITY = {
    "SSN": 10.0,           # Highest risk
    "MRN": 8.0,
    "NAME": 7.0,
    "DOB": 6.0,
    "ADDRESS": 5.0,
    "PHONE": 4.0,
    "EMAIL": 3.0,
    "DATE": 2.0,
    "ZIP": 1.5,
    "CONTEXT_NUMERIC": 1.0  # Lowest risk
}
```

### 1.4 Validation Approach

**Ground Truth:** Manual PHI annotation on a random sample of 50 source PDFs (stratified by surgeon, complexity)

**Metrics Computation:**
1. Compare NB01 redaction log against manual PHI annotations
2. Compute TP (correctly redacted), FP (over-redacted), FN (missed PHI), TN (correctly preserved)
3. Calculate HCAT metrics + 95% bootstrap CIs
4. Generate residual PHI risk heatmap by document type (radiology vs pathology)

**Outputs:**
- `reports/hcat_safety_metrics.csv` — Per-document HCAT scores
- `reports/hcat_summary_statistics.csv` — Aggregate metrics with CIs
- `reports/hcat_residual_phi_risk_heatmap.png` — Risk visualization
- `reports/hcat_phi_type_breakdown.csv` — Breakdown by PHI type

---

## 2. LangGraph Agentic RAG Pipeline (NB12)

### 2.1 Purpose
Build a **knowledge graph** from source documents and validation ground truth, then use **agentic RAG** to:
1. Validate high-risk LLM extractions against known truth
2. Query the knowledge graph to detect fabrications
3. Provide evidence-based corrections for fabricated features

### 2.2 Architecture

Based on the LangChain LangGraph tutorial, our pipeline will have the following nodes:

```
START
  ↓
generate_query_or_validate
  ↓
[tools_condition]
  ├─→ retrieve_from_kg (if tool_call)
  └─→ END (if direct response)
  ↓
grade_retrieved_evidence
  ├─→ generate_validation_result (if relevant)
  └─→ rewrite_query (if not relevant) → loop back
  ↓
END
```

### 2.3 Knowledge Graph Structure

**Nodes:**
- **Patient** — `case_id`, `tumor_type` (invasive/DCIS), `complexity_status`
- **ClinicalFeature** — `feature_name`, `domain` (radiology/pathology), `source_value` (0/1)
- **Observation** — `observation_id`, `modality`, `date`, `surgeon_id`
- **Evidence** — `text_chunk`, `page_num`, `confidence_score`, `extraction_method`

**Edges:**
- `Patient --HAS_OBSERVATION--> Observation`
- `Observation --CONTAINS_FEATURE--> ClinicalFeature`
- `ClinicalFeature --SUPPORTED_BY--> Evidence`
- `Evidence --EXTRACTED_FROM--> SourceDocument`

**Graph Database:** Neo4j (local) or in-memory NetworkX for prototyping

### 2.4 Graph Construction (§12.1–12.3)

**Input Sources:**
1. Deidentified validation Excel (`validation_datasheet_deidentified.xlsx`) — ground truth feature labels
2. Deidentified source PDFs + OCR text from NB04 (`extracted_text/`) — evidence chunks
3. NB03 fabrication analysis (`fabrication_rate_element_level.csv`) — high-risk features

**Construction Steps:**
1. **Parse validation Excel** → Create `Patient`, `ClinicalFeature`, `Observation` nodes
2. **Chunk source PDFs** → Create `Evidence` nodes (500-token chunks with 50-token overlap)
3. **Embed evidence chunks** → Use `text-embedding-3-large` (3072-dim) for semantic search
4. **Link evidence to features** → For each `source_value=1` feature, retrieve top-k most relevant evidence chunks via cosine similarity
5. **Store in graph** → Persist to Neo4j or serialize NetworkX graph

### 2.5 Agentic RAG Nodes

#### Node 1: `generate_query_or_validate`
```python
def generate_query_or_validate(state: MessagesState):
    """
    Given a feature extraction claim (e.g., "Lesion Size = 2.3 cm"),
    decide whether to:
    1. Query the knowledge graph for supporting evidence
    2. Respond directly if feature is low-risk
    """
    response = response_model.bind_tools([kg_retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}
```

#### Node 2: `kg_retriever_tool`
```python
@tool
def kg_retriever_tool(query: str) -> str:
    """
    Retrieve evidence from knowledge graph.
    Query format: "case_id=CASE_ABC123, feature=Lesion Size"
    Returns: Top-3 evidence chunks with confidence scores
    """
    # Parse query → extract case_id, feature_name
    # Query Neo4j: MATCH (p:Patient {case_id})-[:HAS_OBSERVATION]->(o)-[:CONTAINS_FEATURE]->(f {name})-[:SUPPORTED_BY]->(e:Evidence)
    # Return e.text_chunk, e.confidence_score, e.page_num
```

#### Node 3: `grade_retrieved_evidence`
```python
def grade_retrieved_evidence(state: MessagesState) -> Literal["generate_validation_result", "rewrite_query"]:
    """
    Grade whether retrieved evidence supports the LLM's extraction claim.
    Uses structured output: GradeEvidence(binary_score="yes"/"no", explanation=str)
    """
    question = state["messages"][0].content  # Original claim
    evidence = state["messages"][-1].content  # Retrieved chunks
    
    prompt = f"""
    You are validating an LLM's clinical feature extraction.
    
    Claim: {question}
    Evidence from source documents: {evidence}
    
    Does the evidence support the claim? Answer 'yes' or 'no' and explain.
    """
    
    response = grader_model.with_structured_output(GradeEvidence).invoke([{"role": "user", "content": prompt}])
    
    if response.binary_score == "yes":
        return "generate_validation_result"
    else:
        return "rewrite_query"
```

#### Node 4: `generate_validation_result`
```python
def generate_validation_result(state: MessagesState):
    """
    Generate final validation verdict:
    - CORRECT: Evidence supports claim
    - FABRICATION: Evidence contradicts claim
    - OMISSION: Evidence shows feature present but LLM missed it
    - UNCERTAIN: Evidence insufficient
    """
    claim = state["messages"][0].content
    evidence = state["messages"][-1].content
    
    prompt = f"""
    Based on the evidence, classify the LLM's extraction:
    
    Claim: {claim}
    Evidence: {evidence}
    
    Classification: CORRECT | FABRICATION | OMISSION | UNCERTAIN
    Explanation: [brief justification]
    Corrected Value: [if fabrication/omission, provide correct value]
    """
    
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
```

#### Node 5: `rewrite_query`
```python
def rewrite_query(state: MessagesState):
    """
    If initial evidence retrieval failed, rewrite the query with:
    - Synonyms (e.g., "tumor size" → "lesion size", "mass dimensions")
    - Broader context (e.g., add modality: "mammogram lesion size")
    """
    original_query = state["messages"][0].content
    
    prompt = f"""
    The initial query did not retrieve relevant evidence.
    Rewrite this query to improve retrieval:
    
    Original: {original_query}
    
    Rewritten query (use synonyms, add context):
    """
    
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
```

### 2.6 High-Risk Fabrication Detection Workflow

**Trigger:** Run validation for all features where `fabrication_rate > 0.15` (from NB03)

**Process:**
1. Load `fabrication_rate_element_level.csv` → identify high-risk features
2. For each high-risk feature × case:
   - Query: `"case_id={case_id}, feature={feature_name}, ai_extraction={ai_value}"`
   - Run agentic RAG pipeline
   - Collect validation results
3. Generate report:
   - `reports/langgraph_fabrication_validation.csv` — Per-case validation results
   - `reports/langgraph_fabrication_correction_rate.csv` — % of fabrications caught by RAG
   - `reports/langgraph_evidence_quality_scores.csv` — Evidence retrieval quality metrics

### 2.7 LangGraph State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState

class ValidationState(MessagesState):
    """Extended state for fabrication validation."""
    case_id: str
    feature_name: str
    ai_extraction: str
    source_truth: str  # Ground truth from validation Excel
    evidence_chunks: list[dict]  # Retrieved evidence
    validation_verdict: str  # CORRECT | FABRICATION | OMISSION | UNCERTAIN
    confidence_score: float
    corrected_value: str | None
```

### 2.8 Evaluation Metrics

Compare LangGraph validation results against human validation (NB03 ground truth):

| Metric | Definition |
|--------|------------|
| **Fabrication Detection Rate** | % of human-identified fabrications caught by RAG |
| **False Positive Rate** | % of correct extractions flagged as fabrications |
| **Correction Accuracy** | % of fabrications where RAG provided correct value |
| **Evidence Retrieval Precision@3** | % of top-3 chunks relevant to feature |
| **Query Rewrite Success Rate** | % of failed retrievals fixed by query rewriting |

---

## 3. Implementation Plan

### Phase 1: HCAT Safety Metrics (NB11)
1. Create manual PHI annotation tool (simple Jupyter widget)
2. Annotate 50-PDF sample (stratified)
3. Implement HCAT metric calculations
4. Generate safety report + visualizations

### Phase 2: Knowledge Graph Construction (NB12 §1-3)
1. Set up Neo4j local instance or NetworkX graph
2. Parse validation Excel → create Patient/Feature/Observation nodes
3. Chunk + embed source PDFs → create Evidence nodes
4. Link evidence to features via semantic search

### Phase 3: Agentic RAG Pipeline (NB12 §4-7)
1. Implement 5 LangGraph nodes (query, retrieve, grade, validate, rewrite)
2. Assemble graph with conditional edges
3. Test on sample high-risk cases

### Phase 4: Evaluation (NB12 §8)
1. Run validation on all high-risk fabrications
2. Compute detection/correction metrics
3. Generate final report

---

## 4. Expected Outputs

### NB11 Outputs
- `reports/hcat_safety_metrics.csv`
- `reports/hcat_summary_statistics.csv`
- `reports/hcat_residual_phi_risk_heatmap.png`
- `reports/hcat_phi_type_breakdown.csv`
- `data_private/deidentified/manual_phi_annotations.csv` (50 PDFs, not committed)

### NB12 Outputs
- `data/knowledge_graph/clinical_kg.graphml` (NetworkX) or Neo4j database
- `data/knowledge_graph/evidence_embeddings.npy` (3072-dim vectors)
- `reports/langgraph_fabrication_validation.csv`
- `reports/langgraph_fabrication_correction_rate.csv`
- `reports/langgraph_evidence_quality_scores.csv`
- `reports/langgraph_validation_confusion_matrix.png`

---

## 5. Integration with Existing Pipeline

**Updated Notebook Sequence:**
1. NB01 — Deidentification ✅
2. NB02 — Missing Data Analysis ✅
3. NB03 — EDA & Diagnostic Metrics ✅
4. NB04 — Source Doc Text Extraction (Colab)
5. NB05 — Feature Extraction (Colab)
6. NB06 — Metadata & Data Dictionary ✅
7. NB07 — Validation Methods Comparison (Colab)
8. NB08 — OCR Image Quality (optional)
9. NB09 — MCodeGPT DAG (Colab)
10. NB10 — OpenAI Predictive Model (Colab)
11. **NB11 — HCAT Safety Metrics** ← NEW (local)
12. **NB12 — LangGraph Agentic RAG** ← NEW (local, requires NB04 text extraction)

**Dependencies:**
- NB11 depends on: NB01 (deidentification log)
- NB12 depends on: NB01 (deidentified Excel), NB03 (fabrication rates), NB04 (extracted text)

---

## 6. References

- LangChain LangGraph Agentic RAG Tutorial: https://docs.langchain.com/oss/python/langgraph/agentic-rag
- HIPAA Safe Harbor De-identification: https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- HCAT Framework: (internal MSK compliance tool)
- Neo4j Python Driver: https://neo4j.com/docs/python-manual/current/
