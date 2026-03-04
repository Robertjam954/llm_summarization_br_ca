# Final Metrics Selection for DeepEval Integration
**LLM Summarization Breast Cancer | Goel Lab**  
**Date:** March 2026

---

## Overview

This document defines the **final comprehensive metrics** for implementing DeepEval evaluation of our RAG-based clinical feature extraction system. The selection combines:

1. **Existing Analysis Metrics** (from `src/llm_eval_by_human/metrics_utils.py`)
2. **HCAT RAG Metrics** (from HCAT framework paper)
3. **HCAT Safety Metrics** (Toxicity + Privacy Protection only)

---

## 1. Existing Analysis Metrics (Classification & Diagnostic)

These metrics are **already implemented** in our current analysis pipeline and will be **mapped to DeepEval's evaluation framework**.

### **Source:** `@src/llm_eval_by_human/metrics_utils.py:92-138`

```python
def compute_metrics_from_counts(TP: int, FP: int, FN: int, TN: int) -> Dict[str, float]:
    """Compute diagnostic + classification metrics from confusion counts."""
    
    return {
        "sensitivity": sensitivity,      # = Recall = TPR
        "specificity": specificity,      # = TNR
        "ppv": ppv,                      # = Precision
        "npv": npv,                      # Negative Predictive Value
        "accuracy": accuracy,            # (TP + TN) / Total
        "precision": precision,          # = PPV
        "recall": recall,                # = Sensitivity
        "f1": f1,                        # 2 * (Precision * Recall) / (Precision + Recall)
        "fabrication_rate": fabrication_rate,  # FP / (TP + FP)
        "prevalence": prevalence,
        "balanced_accuracy": balanced_accuracy,
    }
```

### **Implementation Strategy:**

These metrics will be computed **post-evaluation** using DeepEval's test results:
- Extract TP/FP/FN/TN from DeepEval metric scores
- Apply existing `compute_metrics_from_counts()` function
- Generate classification reports alongside DeepEval outputs

**Metrics List:**
1. ✅ **Accuracy** - Overall correctness
2. ✅ **Precision** (PPV) - Positive Predictive Value
3. ✅ **Recall** (Sensitivity/TPR) - True Positive Rate
4. ✅ **F1 Score** - Harmonic mean of Precision and Recall
5. ✅ **Specificity** (TNR) - True Negative Rate
6. ✅ **NPV** - Negative Predictive Value
7. ✅ **Balanced Accuracy** - Average of Sensitivity and Specificity
8. ✅ **Fabrication Rate** - FP / (TP + FP) - **Critical for our analysis**

---

## 2. HCAT RAG Metrics (All Included)

Per HCAT framework requirements, **all RAG quality metrics** are included.

### **Context Relevancy** ✅
- **HCAT Definition:** How well retrieved documents address the input query
- **Method:** Sentence-level semantic similarity
- **DeepEval Metric:** `ContextualRelevancyMetric`
- **Use Case:** Validate retrieval quality for each clinical feature query
- **Threshold:** ≥ 0.7

### **Groundedness (Faithfulness)** ✅
- **HCAT Definition:** Generated answer is supported by retrieved context (no hallucinations)
- **Method:** Sentence similarity + NLI entailment probabilities
- **DeepEval Metrics:** 
  - `FaithfulnessMetric` (primary)
  - `HallucinationMetric` (secondary)
- **Use Case:** **Primary fabrication detection** - ensure LLM extractions are grounded in source PDFs
- **Threshold:** ≥ 0.8 (strict for clinical safety)

### **Completeness (Contextual Recall)** ✅
- **HCAT Definition:** Answer covers all essential points from context
- **Method:** Sentence similarity + Wasserstein Distance (Earth Mover's Distance)
- **DeepEval Metric:** `ContextualRecallMetric`
- **Use Case:** Ensure all relevant clinical information is extracted
- **Threshold:** ≥ 0.75

### **Answer Relevancy** ✅
- **HCAT Definition:** Response directly addresses user's specific intent
- **Method:** Semantic similarity between query and answer
- **DeepEval Metric:** `AnswerRelevancyMetric`
- **Use Case:** Validate extractions are on-topic for each feature
- **Threshold:** ≥ 0.7

### **Contextual Precision** ✅
- **HCAT Definition:** Proportion of retrieved context that is relevant
- **Method:** Relevance scoring of each retrieved chunk
- **DeepEval Metric:** `ContextualPrecisionMetric`
- **Use Case:** Minimize irrelevant context in retrieval
- **Threshold:** ≥ 0.7

---

## 3. HCAT Safety Metrics (Toxicity + Privacy Only)

Per user requirements: **Exclude BiasMetric** (no demographic data available).

### **Toxicity** ✅
- **HCAT Definition:** Likelihood of generating harmful or offensive content
- **Importance:** Critical for maintaining professional standards in clinical settings
- **DeepEval Metric:** `ToxicityMetric`
- **Method:** Local NLP model (no LLM required, fast evaluation)
- **Use Case:** **HCAT Safety** - Ensure all LLM outputs are safe for clinical use
- **Threshold:** ≤ 0.1 (very low tolerance)
- **Status:** **SELECTED FOR IMPLEMENTATION**

### **Privacy Protection (PII Leakage)** ✅
- **HCAT Definition:** Ensure model does not disclose sensitive PHI
- **Method:** Named Entity Recognition (NER) + adversarial testing
- **DeepEval Metric:** `PIILeakageMetric`
- **Use Case:** **HCAT Safety** - Validate deidentification, detect residual PHI in LLM outputs
- **HIPAA Relevance:** Direct alignment with Safe Harbor requirements (18 identifiers)
- **Entities Detected:** Names, MRN, SSN, dates, addresses, phone, email, etc.
- **Threshold:** = 0.0 (zero tolerance for PHI leakage)
- **Status:** **SELECTED FOR IMPLEMENTATION**

### **Bias Evaluation** ❌
- **Status:** **EXCLUDED** - No demographic data (race, gender) available currently
- **Future:** May add later when demographic data is collected

---

## 4. Task Completion Metric

### **Task Completion** ✅
- **Definition:** High-level evaluation of whether extraction task was completed successfully
- **DeepEval Metric:** `TaskCompletionMetric`
- **Method:** LLM-based evaluation of task success
- **Use Case:** Overall extraction quality assessment
- **Threshold:** ≥ 0.8

---

## 5. Advanced Evaluation: DAG Metric

### **DAGMetric (Decision Tree)** ✅
- **Definition:** Multi-step conditional evaluation workflow using Deep Acyclic Graph
- **DeepEval Metric:** `DAGMetric`
- **Use Case:** **Complex fabrication detection pipeline** with decision tree logic
- **Features:**
  - Custom nodes for multi-step evaluation
  - Conditional branching based on intermediate results
  - Exportable graph structure and execution trace
  - Verbose logging with step-by-step reasoning

**DAG Structure for Fabrication Detection:**
```
Root Node: Check Retrieval Context Exists
    ├─ YES → Grade Evidence Quality (NonBinary: 0-10)
    │         └─ Check Faithfulness (NonBinary: 0-1)
    │              └─ Detect Hallucination (Binary: Yes/No)
    │                   └─ Validate Completeness (NonBinary: 0-1)
    │                        └─ Task Completion (Binary: Yes/No)
    │                             └─ VERDICT: CORRECT | FABRICATION | OMISSION
    └─ NO → FAIL: Missing retrieval context (Score: 0.0)
```

---

## Final Metrics Summary

### **Total: 14 Metrics**

#### **Group 1: Existing Classification Metrics (8 metrics)**
1. Accuracy
2. Precision (PPV)
3. Recall (Sensitivity)
4. F1 Score
5. Specificity (TNR)
6. NPV
7. Balanced Accuracy
8. Fabrication Rate

#### **Group 2: HCAT RAG Metrics (5 metrics)**
9. ContextualRelevancyMetric
10. FaithfulnessMetric
11. HallucinationMetric
12. ContextualRecallMetric
13. AnswerRelevancyMetric
14. ContextualPrecisionMetric (added for completeness)

#### **Group 3: HCAT Safety Metrics (2 metrics)**
15. ToxicityMetric
16. PIILeakageMetric

#### **Group 4: Task Metrics (1 metric)**
17. TaskCompletionMetric

#### **Group 5: Advanced Evaluation (1 metric)**
18. DAGMetric (Decision Tree)

**Grand Total: 18 Metrics**

---

## Implementation Roadmap

### **Phase 1: Setup & Ground Truth Creation**
1. Install DeepEval: `pip install deepeval`
2. Configure API keys (OpenAI, Confident AI)
3. Create `Golden` test cases from validation dataset
4. Map high-risk features (fabrication_rate > 0.15) from NB03
5. Set up vector store for retrieval context

### **Phase 2: Core Metrics Implementation**
1. Implement 5 HCAT RAG metrics
2. Implement 2 HCAT safety metrics
3. Implement TaskCompletionMetric
4. Test on sample cases
5. Validate thresholds

### **Phase 3: DAG Metric Implementation**
1. Design decision tree structure (6 nodes)
2. Implement custom nodes:
   - `BinaryJudgementNode` (retrieval check, hallucination, task completion)
   - `NonBinaryJudgementNode` (evidence quality, faithfulness, completeness)
3. Test DAG execution
4. Export graph visualization (JSON + PNG)

### **Phase 4: Classification Metrics Integration**
1. Extract TP/FP/FN/TN from DeepEval results
2. Apply existing `compute_metrics_from_counts()` function
3. Generate classification reports
4. Compare with existing NB03 analysis

### **Phase 5: Evaluation & Reporting**
1. Run evaluation on high-risk features
2. Generate comprehensive reports:
   - `reports/deepeval_fabrication_validation.csv`
   - `reports/deepeval_metrics_summary.csv`
   - `reports/deepeval_classification_metrics.csv`
   - `reports/deepeval_dag_execution_trace.json`
   - `reports/deepeval_dag_visualization.png`
   - `reports/hcat_safety_metrics.csv`
3. Perform weakness analysis (marginal + bivariate)
4. Document findings

---

## Integration with Existing Pipeline

```
NB01 (Deidentification)
    ↓
NB02 (Missing Data)
    ↓
NB03 (EDA + Fabrication Rate) ← Identify high-risk features
    ↓                            ← Existing classification metrics
NB04 (Text Extraction - Colab)
    ↓
NB12 (LangGraph RAG + DeepEval) ← NEW
    ├── Build Knowledge Graph (LangGraph)
    ├── Create Goldens from Validation Data
    ├── Run DeepEval Metrics (18 total)
    │   ├── HCAT RAG Metrics (5)
    │   ├── HCAT Safety Metrics (2)
    │   ├── Task Completion (1)
    │   └── DAG Metric (1)
    ├── Compute Classification Metrics (8)
    ├── Export Results
    └── Generate Reports
    ↓
NB11 (HCAT Safety Metrics) ← NEW
    ├── Residual PHI Risk Assessment
    ├── Toxicity Analysis
    └── Safety Report Generation
```

---

## Example Code Structure

### **1. Golden Creation**
```python
from deepeval.dataset import Golden, EvaluationDataset
import pandas as pd

# Load validation data + fabrication rates
validation_df = pd.read_excel("data_private/deidentified/validation_datasheet_deidentified.xlsx")
fabrication_df = pd.read_csv("reports/fabrication_rate_element_level.csv")

# Filter high-risk features
high_risk = fabrication_df[fabrication_df["fabrication_rate_ai"] > 0.15]

goldens = []
for _, row in high_risk.iterrows():
    golden = Golden(
        input=f"What is the {row['feature_name']} for case {row['case_id']}?",
        actual_output=row["ai_extraction"],
        expected_output=row["source_truth"],
        retrieval_context=retrieve_from_vector_store(row["case_id"], row["feature_name"]),
        additional_metadata={
            "case_id": row["case_id"],
            "feature_name": row["feature_name"],
            "fabrication_rate_ai": row["fabrication_rate_ai"]
        }
    )
    goldens.append(golden)

dataset = EvaluationDataset(goldens=goldens)
```

### **2. Metrics Initialization**
```python
from deepeval.metrics import (
    FaithfulnessMetric,
    HallucinationMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    AnswerRelevancyMetric,
    ToxicityMetric,
    PIILeakageMetric,
    TaskCompletionMetric,
    DAGMetric
)

# HCAT RAG Metrics
faithfulness = FaithfulnessMetric(threshold=0.8)
hallucination = HallucinationMetric(threshold=0.5)
contextual_recall = ContextualRecallMetric(threshold=0.75)
contextual_relevancy = ContextualRelevancyMetric(threshold=0.7)
contextual_precision = ContextualPrecisionMetric(threshold=0.7)
answer_relevancy = AnswerRelevancyMetric(threshold=0.7)

# HCAT Safety Metrics
toxicity = ToxicityMetric(threshold=0.1)
pii_leakage = PIILeakageMetric(threshold=0.0)

# Task Completion
task_completion = TaskCompletionMetric(threshold=0.8)

# DAG Metric (defined separately)
dag_metric = create_fabrication_dag_metric()

metrics = [
    faithfulness, hallucination, contextual_recall,
    contextual_relevancy, contextual_precision, answer_relevancy,
    toxicity, pii_leakage, task_completion, dag_metric
]
```

### **3. Evaluation Execution**
```python
import deepeval

# Run evaluation
deepeval.evaluate(
    test_cases=dataset.test_cases,
    metrics=metrics,
    run_async=True
)

# Compute classification metrics
from src.llm_eval_by_human.metrics_utils import compute_metrics_from_counts

results_df = extract_deepeval_results()
classification_metrics = compute_classification_metrics(results_df)
```

---

## References

- **HCAT Paper:** [arxiv.org/pdf/2411.16391](https://arxiv.org/pdf/2411.16391)
- **DeepEval Docs:** [deepeval.com/docs](https://deepeval.com/docs)
- **DeepEval GitHub:** [github.com/confident-ai/deepeval](https://github.com/confident-ai/deepeval)
- **LangGraph Examples:** `C:\Users\jamesr4\local git repos\langgraph\examples\rag`
- **DeepEval Examples:** `C:\Users\jamesr4\local git repos\deepeval\examples`
- **Existing Metrics:** `@src/llm_eval_by_human/metrics_utils.py`
