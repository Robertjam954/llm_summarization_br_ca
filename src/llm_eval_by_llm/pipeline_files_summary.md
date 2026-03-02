# LLM Evaluation Pipeline Files Summary

## Phase 1: Source Document Feature Extraction

### 1. `source_document_feature_extraction.py` (Original)
**Purpose**: Extract text features from source PDFs/DOCX and link to validation annotations.
**Key Features**:
- Uses pypdf and pdfplumber for text extraction
- Computes basic text statistics (word count, lexical diversity, etc.)
- Classifies report types (radiology, pathology, HPI)
- Links to human validation data
- Outputs JSON per case + combined analysis

**Outputs**:
- `data/processed/source_doc_features/case_*.json` (individual cases)
- `data/processed/source_doc_features/all_cases_features.json` (combined)
- `data/processed/source_doc_features/feature_correctness_analysis.json` (analysis)

**Issue Found**: 18/57 matched cases had 0 words extracted due to scanned image PDFs.

---

### 2. `source_document_feature_extraction_v2.py` (OCR Version)
**Purpose**: Enhanced version with OCR fallback for scanned PDFs.
**New Features**:
- Integrates EasyOCR for scanned image PDFs
- Tracks extraction methods per file
- Preserves metadata about OCR usage

**Status**: Created but not fully tested due to memory issues with OCR processing.

---

### 3. `source_document_feature_extraction_v3.py` (Text Preservation)
**Purpose**: Preserves actual text content for embedding analysis.
**New Features**:
- Stores `text_content` for each file
- Stores `combined_source_text` for each case
- Optimized for similarity analysis with Voyage AI

**Status**: Created but requires OCR for scanned PDFs.

---

### 4. `source_document_feature_extraction_v3_simple.py` (No OCR)
**Purpose**: Text preservation without OCR for faster processing.
**Features**:
- Same as v3 but skips OCR
- Flags files with failed extraction
- Suitable for cases with extractable text only

**Outputs**: Same structure as v1 but with preserved text content.

---

## Phase 2: Document Similarity Analysis

### 5. `document_similarity_analysis.py`
**Purpose**: Analyze similar cases with different annotation outcomes and detect error patterns.
**Key Features**:
- Uses Voyage AI embeddings for semantic similarity
- Finds similar case pairs with different AI outcomes
- Clusters cases by error patterns using DBSCAN
- Creates t-SNE visualizations
- Generates comprehensive analysis report

**Dependencies**:
- `voyageai` for embeddings
- `sklearn` for clustering and similarity
- `matplotlib`/`seaborn` for visualizations

**Outputs**:
- `data/processed/similarity_analysis/similar_case_pairs.json`
- `data/processed/similarity_analysis/error_clusters.json`
- `data/processed/similarity_analysis/similarity_map_*.png`
- `data/processed/similarity_analysis/similarity_analysis_report.json`

---

## Phase 3: DeepEval Multi-Model Pipeline

### 6. `deepeval_multi_model_pipeline.py`
**Purpose**: Run multiple LLMs with various prompts on source documents.
**Key Features**:
- Loads prompt library from CSV
- Integrates BFOP and 2POP prompts from mCodeGPT
- Calls multiple LLM APIs (OpenAI, Anthropic, Mistral)
- Parses JSON outputs and computes metrics
- Optional DeepEval hallucination/faithfulness metrics
- Saves validation sheets and run summaries

**Outputs**:
- Excel validation sheets per configuration
- JSON details per case
- Metrics summaries
- Run metadata

---

## Phase 4: Prompt Iteration Tracking

### 7. `prompt_iteration_tracker.py`
**Purpose**: Track metrics across prompt versions and auto-generate refined prompts.
**Key Features**:
- Loads all DeepEval run results
- Builds tracking table of metrics per element/prompt
- Identifies problematic elements
- LLM-based prompt auto-refinement with reasoning
- Visualizes metric trends

**Outputs**:
- Tracking tables (CSV/JSON)
- Refined prompts with reasoning
- Trend plots

---

## Phase 5: Time-Series Forecasting

### 8. `timeseries_prompt_forecasting.py`
**Purpose**: Forecast accuracy/fabrication rates over prompt iterations.
**Key Features**:
- Linear regression trend analysis
- Exponential smoothing
- ARIMA modeling (if sufficient data)
- Change-point detection
- Feasibility assessment

**Outputs**:
- Forecast plots
- Model performance metrics
- Feasibility report

---

## Supporting Files

### 9. `requirements_pipeline.txt`
**Purpose**: List all required dependencies for the pipeline.
**Contents**: pandas, numpy, pypdf, python-docx, LLM SDKs, deepeval, statistics libraries, visualization tools.

---

### 10. `text_extraction_issues_log.md`
**Purpose**: Document the scanned PDF issue discovered during Phase 1.
**Contents**:
- Description of the problem (scanned image PDFs)
- Technical investigation results
- Impact on analysis
- Proposed solutions (OCR integration, exclusion, hybrid approach)

---

### 11. `test_similarity_analysis.py`
**Purpose**: Test similarity analysis without Voyage AI using mock embeddings.
**Features**:
- Validates data loading
- Checks text preservation
- Demonstrates similarity concept with mock data

---

## Data Flow

```
Source Documents (PDF/DOCX)
    ↓
Phase 1: Feature Extraction (v3_simple)
    ↓ (preserved text + features)
Phase 2: Similarity Analysis
    ↓ (identifies patterns)
Phase 3: DeepEval Pipeline
    ↓ (generates summaries)
Phase 4: Prompt Tracking
    ↓ (optimizes prompts)
Phase 5: Time-Series Forecasting
```

## Current Status

1. **Phase 1**: Working, but needs to handle scanned PDFs (flag/exclude)
2. **Phase 2**: Ready, needs Voyage AI API key
3. **Phase 3**: Ready, needs LLM API keys
4. **Phase 4**: Ready, needs Phase 3 outputs
5. **Phase 5**: Ready, needs Phase 4 outputs

## Next Steps

1. Run `source_document_feature_extraction_v3_simple.py` to get text-preserved data
2. Set up Voyage AI API key for similarity analysis
3. Set up LLM API keys for DeepEval pipeline
4. Consider OCR integration for the 18 cases with scanned PDFs if needed
