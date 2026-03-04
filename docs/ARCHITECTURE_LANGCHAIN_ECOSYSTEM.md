# LangChain Ecosystem Architecture
**LangChain + LangGraph + LangSmith + DeepEval**  
**Memorial Sloan Kettering | Goel Lab**  
**Date:** March 2026

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Component Stack](#component-stack)
4. [LangChain: Core Framework](#langchain-core-framework)
5. [LangGraph: Agentic Workflows](#langgraph-agentic-workflows)
6. [LangSmith: Observability & Monitoring](#langsmith-observability--monitoring)
7. [DeepEval: Evaluation & Testing](#deepeval-evaluation--testing)
8. [Event Tracing Schema](#event-tracing-schema)
9. [Production Deployment Architecture](#production-deployment-architecture)
10. [Integration Patterns](#integration-patterns)
11. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

This document describes the complete architecture for our LLM validation and production monitoring system using the LangChain ecosystem. The system combines:

- **LangChain** — Core framework for retrieval, agents, and tools
- **LangGraph** — State-based agentic workflows with multi-agent orchestration
- **LangSmith** — Production observability, tracing, and monitoring
- **DeepEval** — Comprehensive LLM evaluation metrics and testing

### Use Cases

1. **LLM Validation** — Validate clinical feature extractions against source documents via agentic RAG
2. **Fabrication Detection** — Query knowledge graph to detect and correct LLM fabrications
3. **Production Monitoring** — Real-time monitoring of LLM performance with alerts
4. **Queryable Retrieval Agent** — App-accessible agent for clinical data queries

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Web App     │  │  API Client  │  │  Jupyter NB  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼────────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                    LANGGRAPH AGENT LAYER                             │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │         Agentic RAG State Graph (NB12)               │           │
│  │                                                       │           │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────┐ │           │
│  │  │ Query/      │───▶│ KG Retriever │───▶│ Grade   │ │           │
│  │  │ Validate    │    │ Tool         │    │ Evidence│ │           │
│  │  └─────────────┘    └──────────────┘    └────┬────┘ │           │
│  │         │                                     │      │           │
│  │         │            ┌──────────────┐         │      │           │
│  │         └───────────▶│ Rewrite      │◀────────┘      │           │
│  │                      │ Query        │                │           │
│  │                      └──────────────┘                │           │
│  │                             │                        │           │
│  │                      ┌──────▼───────┐                │           │
│  │                      │ Generate     │                │           │
│  │                      │ Validation   │                │           │
│  │                      └──────────────┘                │           │
│  └───────────────────────────────────────────────────────┘           │
│                             │                                        │
│                    (MessagesState + ValidationState)                 │
└────────────────────────────┼────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                    LANGCHAIN CORE LAYER                              │
│                             │                                        │
│  ┌──────────────┐  ┌───────▼────────┐  ┌──────────────┐            │
│  │  Document    │  │  Vector Store  │  │  LLM Models  │            │
│  │  Loaders     │  │  (Chroma/FAISS)│  │  (OpenAI/    │            │
│  │              │  │                │  │   Anthropic) │            │
│  └──────┬───────┘  └───────┬────────┘  └──────┬───────┘            │
│         │                  │                   │                    │
│  ┌──────▼───────┐  ┌───────▼────────┐  ┌──────▼───────┐            │
│  │  Text        │  │  Embeddings    │  │  Prompt      │            │
│  │  Splitters   │  │  (OpenAI       │  │  Templates   │            │
│  │              │  │   text-emb-3)  │  │              │            │
│  └──────────────┘  └────────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH LAYER                             │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │         Clinical Knowledge Graph (Neo4j)             │           │
│  │                                                       │           │
│  │  Patient ──HAS_OBSERVATION──▶ Observation            │           │
│  │     │                              │                 │           │
│  │     │                       CONTAINS_FEATURE         │           │
│  │     │                              │                 │           │
│  │     │                              ▼                 │           │
│  │     │                      ClinicalFeature           │           │
│  │     │                              │                 │           │
│  │     │                       SUPPORTED_BY             │           │
│  │     │                              │                 │           │
│  │     │                              ▼                 │           │
│  │     └────────────────────▶    Evidence               │           │
│  │                                   │                  │           │
│  │                            EXTRACTED_FROM            │           │
│  │                                   │                  │           │
│  │                                   ▼                  │           │
│  │                           SourceDocument             │           │
│  └───────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│              LANGSMITH OBSERVABILITY LAYER                           │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │                  Trace Collection                     │           │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │           │
│  │  │ Prompt  │  │ LLM     │  │ Tool    │  │Retriever│ │           │
│  │  │ Span    │  │ Span    │  │ Span    │  │ Span    │ │           │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │           │
│  └───────────────────────────────────────────────────────┘           │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │              Monitoring & Dashboards                  │           │
│  │  • Latency metrics    • Error rates                  │           │
│  │  • Token usage        • Cost tracking                │           │
│  │  • Custom alerts      • Performance trends           │           │
│  └───────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────────┐
│                    DEEPEVAL EVALUATION LAYER                         │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │              Offline Evaluation (NB12)                │           │
│  │  • Faithfulness       • Answer Relevancy              │           │
│  │  • Contextual Recall  • Contextual Precision          │           │
│  │  • Hallucination      • Toxicity                      │           │
│  │  • Bias               • Task Completion               │           │
│  └───────────────────────────────────────────────────────┘           │
│                             │                                        │
│  ┌──────────────────────────▼───────────────────────────┐           │
│  │           Online Evaluation (Production)              │           │
│  │  • Real-time metric calculation                       │           │
│  │  • Metric collections via CallbackHandler            │           │
│  │  • Automated alerts on metric degradation            │           │
│  └───────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Stack

### Core Dependencies

```python
# LangChain ecosystem
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-community>=0.3.0
langsmith>=0.2.0

# Vector stores & embeddings
chromadb>=0.5.0
faiss-cpu>=1.8.0

# Knowledge graph
neo4j>=5.0.0
networkx>=3.0

# Evaluation
deepeval>=1.0.0

# Observability
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
```

---

## LangChain: Core Framework

### Retrieval Components

LangChain handles the complete text preprocessing pipeline from raw documents to searchable embeddings. OpenAI only provides the embedding model - all document loading, splitting, and chunking is done by LangChain.

**Step 1: Document Loaders**
```python
from langchain_community.document_loaders import PyMuPDFLoader

# Load deidentified PDFs - LangChain extracts text and metadata
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()  # Returns list of Document objects

# Each Document has:
# - page_content: extracted text
# - metadata: {source, page, case_id, etc.}
```

**Step 2: Text Splitters (Chunking)**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain splits documents into chunks with overlap
# Recursively tries separators: ["\n\n", "\n", " ", ""]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,           # Max characters per chunk
    chunk_overlap=200,         # Overlap to preserve context
    add_start_index=True,      # Track position in original doc
    length_function=len,       # Measure by character count
)

# Split documents into smaller chunks
all_splits = text_splitter.split_documents(documents)
# Returns list of Document objects with chunked content + metadata
```

**Step 3: Embeddings (OpenAI)**
```python
from langchain_openai import OpenAIEmbeddings

# OpenAI ONLY provides embedding model
# LangChain handles batching and API calls
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=3072  # High-dimensional for better accuracy
)
```

**Step 4: Vector Store (Storage + Retrieval)**
```python
from langchain_community.vectorstores import Chroma

# LangChain creates embeddings for all chunks and stores them
vectorstore = Chroma.from_documents(
    documents=all_splits,      # Chunked documents from Step 2
    embedding=embeddings,      # Embedding model from Step 3
    collection_name="clinical_evidence",
    persist_directory="./data/vectorstore"
)

# Create retriever with metadata filtering
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,  # Return top 3 chunks
        "filter": {"case_id": "CASE_ABC123"}  # Filter by metadata
    }
)
```

**Complete Pipeline Example**
```python
# Full indexing pipeline (LangChain handles everything except embeddings)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load documents
loader = PyMuPDFLoader("path/to/deidentified.pdf")
docs = loader.load()

# 2. Split into chunks (LangChain preprocessing)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 3. Create embeddings (OpenAI) and store (LangChain)
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large")
)

# 4. Retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("What is the lesion size?")
```

### Tools

**Retriever Tool**
```python
from langchain.tools.retriever import create_retriever_tool

kg_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="retrieve_clinical_evidence",
    description="Retrieves evidence from source documents for a given case_id and feature. "
                "Input format: 'case_id=CASE_ABC123, feature=Lesion Size'"
)
```

---

## LangGraph: Agentic Workflows

### State Management

**Base State Schema**
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
    retrieval_attempts: int  # Track query rewrites
```

### Graph Construction

**Agentic RAG Graph**
```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Initialize graph
workflow = StateGraph(ValidationState)

# Add nodes
workflow.add_node("generate_query_or_validate", generate_query_or_validate)
workflow.add_node("retrieve", ToolNode([kg_retriever_tool]))
workflow.add_node("grade_evidence", grade_retrieved_evidence)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_validation", generate_validation_result)

# Add edges
workflow.add_edge(START, "generate_query_or_validate")

# Conditional edge: decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_validate",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Conditional edge: grade retrieved evidence
workflow.add_conditional_edges(
    "retrieve",
    grade_evidence,
    {
        "generate_validation": "generate_validation",
        "rewrite_query": "rewrite_query"
    }
)

workflow.add_edge("generate_validation", END)
workflow.add_edge("rewrite_query", "generate_query_or_validate")

# Compile
graph = workflow.compile()
```

### Multi-Agent Patterns

**Supervisor Pattern** (for complex workflows)
```python
from langgraph.graph import StateGraph

# Define specialized agents
retrieval_agent = create_retrieval_agent()
validation_agent = create_validation_agent()
correction_agent = create_correction_agent()

# Supervisor decides which agent to call
def supervisor_node(state):
    if state["retrieval_attempts"] < 3:
        return "retrieval_agent"
    elif state["evidence_chunks"]:
        return "validation_agent"
    else:
        return "correction_agent"

# Build supervisor graph
supervisor_graph = StateGraph(ValidationState)
supervisor_graph.add_node("supervisor", supervisor_node)
supervisor_graph.add_node("retrieval_agent", retrieval_agent)
supervisor_graph.add_node("validation_agent", validation_agent)
supervisor_graph.add_node("correction_agent", correction_agent)
# ... add conditional edges based on supervisor decision
```

---

## LangSmith: Observability & Monitoring

### Tracing Setup

**Enable Tracing**
```python
import os

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "llm-validation-production"
```

**Automatic Tracing** (LangChain/LangGraph)
```python
# Tracing is automatic for LangChain/LangGraph
# Every invoke/ainvoke call is traced
result = graph.invoke({"messages": [{"role": "user", "content": query}]})
```

**Manual Tracing** (custom code)
```python
from langsmith import traceable

@traceable(name="custom_validation_step")
def validate_extraction(case_id: str, feature: str, ai_value: str):
    # Custom validation logic
    # Automatically traced with inputs/outputs
    return validation_result
```

### Event Tracing Schema (LangSmith Format)

**Trace Structure**
```
Trace (Run)
├── Span 1: Prompt Templating
│   ├── Input: {case_id, feature_name, ai_extraction}
│   ├── Output: Formatted prompt string
│   ├── Metadata: {template_version, tokens}
│   └── Duration: 2ms
│
├── Span 2: LLM Call (generate_query_or_validate)
│   ├── Input: Prompt + conversation history
│   ├── Output: AIMessage with tool_calls
│   ├── Metadata: {model: "gpt-4o", temperature: 0, tokens: 450}
│   └── Duration: 1200ms
│
├── Span 3: Tool Usage (kg_retriever_tool)
│   ├── Input: {query: "case_id=CASE_ABC, feature=Lesion Size"}
│   ├── Output: [Evidence chunks with metadata]
│   ├── Metadata: {retriever_type: "similarity", k: 3}
│   └── Duration: 350ms
│
├── Span 4: LLM Call (grade_evidence)
│   ├── Input: Question + retrieved evidence
│   ├── Output: GradeEvidence(binary_score="yes", explanation="...")
│   ├── Metadata: {model: "gpt-4o", structured_output: true}
│   └── Duration: 800ms
│
└── Span 5: LLM Call (generate_validation)
    ├── Input: Claim + graded evidence
    ├── Output: ValidationResult(verdict="CORRECT", confidence=0.95)
    ├── Metadata: {model: "gpt-4o", final_verdict: true}
    └── Duration: 950ms

Total Trace Duration: 3302ms
Total Tokens: 1250 (prompt: 800, completion: 450)
Total Cost: $0.0125
```

**Span Types**
1. **Prompt Span** — Template formatting, variable substitution
2. **LLM Span** — Model invocation (OpenAI, Anthropic, etc.)
3. **Tool Span** — Tool execution (retriever, calculator, API calls)
4. **Retriever Span** — Vector store queries, document retrieval
5. **Chain Span** — Sequential operations (LCEL chains)
6. **Agent Span** — Agent decision-making loops

### Monitoring & Dashboards

**Key Metrics**
```python
# Automatically tracked by LangSmith
- Latency (p50, p95, p99)
- Token usage (prompt + completion)
- Cost per request
- Error rate
- Success rate
- Feedback scores
```

**Custom Metrics**
```python
from langsmith import Client

client = Client()

# Log custom metrics
client.create_feedback(
    run_id=run.id,
    key="validation_accuracy",
    score=0.95,
    comment="Correctly identified fabrication"
)
```

**Alerts**
```python
# Configure via LangSmith UI or API
# Alert conditions:
- Latency > 5s (p95)
- Error rate > 5%
- Cost per day > $100
- Custom metric threshold (e.g., validation_accuracy < 0.8)
```

---

## DeepEval: Evaluation & Testing

### Offline Evaluation (Development)

**Metrics**
```python
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric
)

# Define metrics for RAG evaluation
faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o")
answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o")
contextual_recall = ContextualRecallMetric(threshold=0.7, model="gpt-4o")
hallucination = HallucinationMetric(threshold=0.3, model="gpt-4o")
```

**Test Cases**
```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

# Create test cases
test_cases = [
    LLMTestCase(
        input="What is the lesion size for case CASE_ABC123?",
        actual_output="2.3 cm",
        expected_output="2.3 cm",
        retrieval_context=["Evidence chunk 1", "Evidence chunk 2"],
        context=["Ground truth from validation Excel"]
    ),
    # ... more test cases
]

dataset = EvaluationDataset(test_cases=test_cases)
```

**Run Evaluation**
```python
from deepeval import evaluate

# Evaluate with multiple metrics
results = evaluate(
    test_cases=dataset,
    metrics=[faithfulness, answer_relevancy, contextual_recall, hallucination]
)

# View results
print(results.test_results)
```

### Online Evaluation (Production)

**LangChain Integration**
```python
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import TaskCompletionMetric

# Add callback to production agent
result = graph.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={
        "callbacks": [
            CallbackHandler(
                metrics=[TaskCompletionMetric()],
                # Or use metric collection from Confident AI
                # metric_collection="production-validation-metrics"
            )
        ]
    }
)
```

**Continuous Monitoring**
```python
# DeepEval automatically logs to Confident AI platform
# View real-time metrics, traces, and alerts at:
# https://app.confident-ai.com/

# Metrics tracked:
- Task completion rate
- Faithfulness score
- Answer relevancy
- Hallucination rate
- Latency
- Cost
```

### Custom Metrics

**Fabrication Detection Metric**
```python
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class FabricationDetectionMetric(BaseMetric):
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def measure(self, test_case: LLMTestCase):
        # Compare AI extraction against source evidence
        # Return score 0-1 (1 = no fabrication)
        
        # Custom logic using knowledge graph
        evidence = retrieve_from_kg(test_case.input)
        score = compare_extraction_to_evidence(
            test_case.actual_output,
            evidence
        )
        
        self.score = score
        self.success = score >= self.threshold
        self.reason = f"Fabrication score: {score:.2f}"
        
        return self.score
```

---

## Event Tracing Schema

### Trace Hierarchy

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional

@dataclass
class Span:
    """Individual span within a trace."""
    span_id: str
    parent_span_id: Optional[str]
    name: str
    span_type: str  # "prompt" | "llm" | "tool" | "retriever" | "chain" | "agent"
    start_time: datetime
    end_time: datetime
    duration_ms: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class Trace:
    """Complete trace for a single request."""
    trace_id: str
    project_name: str
    start_time: datetime
    end_time: datetime
    total_duration_ms: float
    spans: List[Span]
    metadata: Dict[str, Any]
    tags: List[str]
    feedback: Optional[Dict[str, Any]] = None
    
    # Aggregated metrics
    total_tokens: int
    total_cost: float
    success: bool
    error: Optional[str] = None
```

### Span Metadata Schema

**Prompt Span**
```python
{
    "template_name": "validation_query_template",
    "template_version": "v2.1",
    "variables": {"case_id": "CASE_ABC", "feature": "Lesion Size"},
    "output_tokens": 120
}
```

**LLM Span**
```python
{
    "model": "gpt-4o",
    "provider": "openai",
    "temperature": 0.0,
    "max_tokens": 500,
    "prompt_tokens": 450,
    "completion_tokens": 180,
    "total_tokens": 630,
    "cost": 0.0063,
    "structured_output": true,
    "output_schema": "GradeEvidence"
}
```

**Tool Span**
```python
{
    "tool_name": "kg_retriever_tool",
    "tool_type": "retriever",
    "search_type": "similarity",
    "k": 3,
    "filter": {"case_id": "CASE_ABC"},
    "num_results": 3,
    "avg_score": 0.87
}
```

**Retriever Span**
```python
{
    "retriever_type": "vectorstore",
    "vectorstore": "chroma",
    "collection": "clinical_evidence",
    "embedding_model": "text-embedding-3-large",
    "query": "lesion size mammogram",
    "k": 3,
    "results": [
        {"doc_id": "chunk_123", "score": 0.92, "metadata": {...}},
        {"doc_id": "chunk_456", "score": 0.85, "metadata": {...}},
        {"doc_id": "chunk_789", "score": 0.78, "metadata": {...}}
    ]
}
```

### Implementation

**Automatic Tracing (LangChain/LangGraph)**
```python
# Tracing is automatic when LANGSMITH_TRACING=true
# No code changes needed
```

**Manual Span Creation**
```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable(name="custom_validation_logic", run_type="chain")
def validate_with_kg(case_id: str, feature: str):
    run_tree = get_current_run_tree()
    
    # Add custom metadata
    run_tree.extra = {
        "case_id": case_id,
        "feature": feature,
        "validation_type": "knowledge_graph"
    }
    
    # Your validation logic
    result = perform_validation(case_id, feature)
    
    return result
```

---

## Production Deployment Architecture

### Deployment Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼──────────┐ ┌─▼───────────────┐
│  API Server 1   │ │ API Server 2 │ │  API Server 3   │
│  (FastAPI)      │ │  (FastAPI)   │ │  (FastAPI)      │
│                 │ │              │ │                 │
│  LangGraph      │ │  LangGraph   │ │  LangGraph      │
│  Agent          │ │  Agent       │ │  Agent          │
└────────┬────────┘ └───┬──────────┘ └─┬───────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼────────┐ ┌───▼──────────┐ ┌─▼───────────────┐
│  Vector Store   │ │  Neo4j KG    │ │  LangSmith      │
│  (Chroma)       │ │              │ │  (Tracing)      │
└─────────────────┘ └──────────────┘ └─────────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                  ┌──────▼──────┐
                  │  DeepEval   │
                  │  (Confident │
                  │   AI)       │
                  └─────────────┘
```

### FastAPI Application

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langsmith import Client
from deepeval.integrations.langchain import CallbackHandler

app = FastAPI(title="Clinical Validation API")

# Initialize LangSmith client
langsmith_client = Client()

# Load compiled graph
validation_graph = load_validation_graph()

class ValidationRequest(BaseModel):
    case_id: str
    feature_name: str
    ai_extraction: str
    source_truth: str

class ValidationResponse(BaseModel):
    verdict: str
    confidence: float
    corrected_value: str | None
    evidence: list[dict]
    trace_url: str

@app.post("/validate", response_model=ValidationResponse)
async def validate_extraction(request: ValidationRequest):
    """Validate LLM extraction against knowledge graph."""
    
    try:
        # Run validation with tracing + evaluation
        result = validation_graph.invoke(
            {
                "case_id": request.case_id,
                "feature_name": request.feature_name,
                "ai_extraction": request.ai_extraction,
                "source_truth": request.source_truth
            },
            config={
                "callbacks": [
                    CallbackHandler(
                        metric_collection="production-validation-metrics"
                    )
                ],
                "metadata": {
                    "endpoint": "/validate",
                    "user_id": "system"
                }
            }
        )
        
        # Get trace URL from LangSmith
        run_id = result.get("run_id")
        trace_url = f"https://smith.langchain.com/o/{org_id}/projects/p/{project_id}/r/{run_id}"
        
        return ValidationResponse(
            verdict=result["validation_verdict"],
            confidence=result["confidence_score"],
            corrected_value=result.get("corrected_value"),
            evidence=result["evidence_chunks"],
            trace_url=trace_url
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "tracing": "enabled"}
```

### Monitoring & Alerts

**LangSmith Alerts**
```python
# Configure via LangSmith UI or API
alerts = [
    {
        "name": "High Latency",
        "condition": "p95_latency > 5000",  # 5 seconds
        "action": "email",
        "recipients": ["team@example.com"]
    },
    {
        "name": "Error Rate Spike",
        "condition": "error_rate > 0.05",  # 5%
        "action": "slack",
        "channel": "#llm-alerts"
    },
    {
        "name": "Cost Threshold",
        "condition": "daily_cost > 100",  # $100/day
        "action": "email + slack"
    },
    {
        "name": "Validation Accuracy Drop",
        "condition": "avg(validation_accuracy) < 0.8",
        "action": "pagerduty"
    }
]
```

**DeepEval Continuous Monitoring**
```python
# Automatic monitoring via CallbackHandler
# View metrics at: https://app.confident-ai.com/

# Tracked metrics:
- Faithfulness (avg, p95)
- Answer relevancy (avg, p95)
- Hallucination rate
- Task completion rate
- Latency distribution
- Cost per request
```

---

## Integration Patterns

### Pattern 1: Validation Pipeline

```python
# NB12: Offline validation of high-risk fabrications
from deepeval import evaluate
from deepeval.test_case import LLMTestCase

# Load high-risk cases from NB03
high_risk_df = pd.read_csv("reports/fabrication_rate_element_level.csv")
high_risk_df = high_risk_df[high_risk_df["fabrication_rate_ai"] > 0.15]

# Create test cases
test_cases = []
for _, row in high_risk_df.iterrows():
    test_case = LLMTestCase(
        input=f"case_id={row['case_id']}, feature={row['feature_name']}",
        actual_output=row["ai_extraction"],
        expected_output=row["source_truth"],
        retrieval_context=retrieve_evidence(row["case_id"], row["feature_name"])
    )
    test_cases.append(test_case)

# Run validation with tracing
results = evaluate(
    test_cases=test_cases,
    metrics=[
        FabricationDetectionMetric(),
        FaithfulnessMetric(),
        ContextualRecallMetric()
    ]
)

# Save results
results.to_csv("reports/langgraph_fabrication_validation.csv")
```

### Pattern 2: Queryable Retrieval Agent (App Integration)

```python
# FastAPI endpoint for app queries
@app.post("/query")
async def query_clinical_data(query: str, case_id: str):
    """Query clinical data via agentic RAG."""
    
    result = query_graph.invoke(
        {
            "messages": [{"role": "user", "content": query}],
            "case_id": case_id
        },
        config={
            "callbacks": [
                CallbackHandler(
                    metric_collection="app-query-metrics"
                )
            ]
        }
    )
    
    return {
        "answer": result["messages"][-1].content,
        "sources": result.get("evidence_chunks", []),
        "confidence": result.get("confidence_score", 0.0)
    }
```

### Pattern 3: Production Monitoring

```python
# Continuous monitoring with LangSmith + DeepEval
import asyncio
from langsmith import Client

async def monitor_production():
    """Monitor production metrics and trigger alerts."""
    
    client = Client()
    
    while True:
        # Fetch last hour of traces
        runs = client.list_runs(
            project_name="llm-validation-production",
            start_time=datetime.now() - timedelta(hours=1)
        )
        
        # Compute metrics
        latencies = [r.total_time for r in runs]
        errors = [r for r in runs if r.error]
        costs = [r.total_cost for r in runs]
        
        # Check thresholds
        if np.percentile(latencies, 95) > 5000:
            send_alert("High latency detected")
        
        if len(errors) / len(runs) > 0.05:
            send_alert("Error rate spike")
        
        if sum(costs) > 100:
            send_alert("Daily cost threshold exceeded")
        
        await asyncio.sleep(300)  # Check every 5 minutes
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Tasks:**
1. ✅ Install LangChain ecosystem packages
2. ✅ Set up LangSmith account + API keys
3. ✅ Set up DeepEval/Confident AI account
4. ✅ Configure environment variables
5. ⬜ Create knowledge graph schema (Neo4j or NetworkX)
6. ⬜ Implement document loaders + text splitters
7. ⬜ Create vector store with embeddings

**Deliverables:**
- Working vector store with clinical evidence
- LangSmith tracing enabled
- DeepEval test suite template

### Phase 2: Agentic RAG Pipeline (Week 3-4)

**Tasks:**
1. ⬜ Build knowledge graph from validation Excel + source PDFs
2. ⬜ Implement 5 LangGraph nodes (query, retrieve, grade, validate, rewrite)
3. ⬜ Assemble state graph with conditional edges
4. ⬜ Test on sample high-risk cases
5. ⬜ Integrate LangSmith tracing
6. ⬜ Add DeepEval metrics

**Deliverables:**
- NB12: LangGraph Agentic RAG notebook
- Validation results for high-risk fabrications
- Trace visualization in LangSmith

### Phase 3: Evaluation & Testing (Week 5)

**Tasks:**
1. ⬜ Create comprehensive test dataset (50-100 cases)
2. ⬜ Run offline evaluation with DeepEval
3. ⬜ Compute fabrication detection rate
4. ⬜ Generate evaluation report
5. ⬜ Tune retrieval parameters based on results

**Deliverables:**
- Evaluation report with metrics
- Tuned hyperparameters
- Documented failure modes

### Phase 4: Production Deployment (Week 6-7)

**Tasks:**
1. ⬜ Build FastAPI application
2. ⬜ Containerize with Docker
3. ⬜ Deploy to cloud (AWS/GCP/Azure)
4. ⬜ Configure LangSmith alerts
5. ⬜ Set up DeepEval continuous monitoring
6. ⬜ Create monitoring dashboard

**Deliverables:**
- Production API endpoint
- Monitoring dashboard
- Alert configuration
- Deployment documentation

### Phase 5: HCAT Safety Metrics (Week 8)

**Tasks:**
1. ⬜ Implement NB11 (HCAT safety metrics)
2. ⬜ Manual PHI annotation (50-PDF sample)
3. ⬜ Compute residual PHI risk scores
4. ⬜ Generate safety report

**Deliverables:**
- NB11: HCAT Safety Metrics notebook
- Safety report with risk scores
- Recommendations for deidentification improvements

---

## References

### Documentation
- [LangChain Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith Observability](https://docs.langchain.com/langsmith/observability)
- [DeepEval Documentation](https://deepeval.com/)
- [DeepEval LangChain Integration](https://deepeval.com/integrations/frameworks/langchain)

### Tutorials
- [Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [Trace a RAG app with LangSmith](https://docs.langchain.com/langsmith/observability-llm-tutorial)
- [Evaluate a graph with DeepEval](https://docs.langchain.com/langsmith/evaluate-graph)

### GitHub
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [DeepEval](https://github.com/confident-ai/deepeval)
