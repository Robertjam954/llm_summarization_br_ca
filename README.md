# LLM Summarization BR/CA Project

## Project Overview

This repository contains a robust, production-grade pipeline for LLM-based document summarization focused on Brazilian Portuguese (BR) and Canadian French (CA) language processing. The project implements end-to-end evaluation, experimentation tracking, and deployment capabilities.

## Repository Structure

### `data/`
Storage and management of all datasets and processed outputs.

- **`raw/`** - Source PDFs and original documents (typically internal/private)
  - Contains unprocessed source materials
  - Generally not shared externally  

- **`processed/`** - Deidentified and cleaned outputs
  - Merged validation sheets
  - Element-level metrics CSVs
  - Tabular outputs ready for analysis  

- **`splits/`** - Train/test/dev dataset definitions
  - Train/test splits for model evaluation
  - Dev/test definitions for hyperparameter tuning
  - Metadata about data partition strategy

### `prompts/`
Prompt engineering and management system.

- **`library/`** - Versioned prompt templates with metadata
  - Prompt templates organized by approach/intent
  - Metadata: approach, intent, constraints
  - Version control for prompt iterations  

- **`generated/`** - Agent-created and derived prompts
  - Prompts generated during experimentation
  - Lineage tracking (parent prompt references)
  - Automatic generation metadata

### `models/`
Model configuration and management.

- **`configs/`** - Model specifications and parameters
  - Model name, version, and variant information
  - Temperature and sampling parameters
  - System prompts and initialization configs
  - API keys and authentication (in `.env.example`)

### `eval/`
Evaluation framework and metrics.

- **`schemas/`** - Evaluation label definitions
  - Label schema and taxonomy
  - Element definitions
  - Adjudication rules for conflicting annotations  

- **`metrics/`** - Metric implementations
  - TP/FN/FP/TN calculation code
  - Hallucination/fabrication detection
  - Custom metric definitions
  - Performance reporting functions

### `reports/`
Auto-generated evaluation and analysis outputs.

- Exported tables and figures
- Hash and timestamp metadata for reproducibility
- Performance dashboards and summaries

### `experiments/`
Experiment tracking and run management.

- **`runs/`** - Individual experiment run records
  - `run_id`: Unique identifier for each run
  - `git_commit_hash`: Version control link
  - `prompt_id`: Reference to prompts used
  - `model_id`: Reference to model configuration
  - `dataset_snapshot_id`: Data version reference
  - Results, logs, and metrics from each run

### `apps/`
Deployed applications and services.

- **`dashboard/`** - Live forecasting and monitoring
  - Real-time prediction interface
  - Alert system for anomalies
  - Performance monitoring dashboard
  - Model serving infrastructure

### `docs/`
Comprehensive documentation.

- **`protocol.md`** - Frozen evaluation protocol
  - Final, versioned evaluation methodology
  - Do not modify without proper version control
  - Reference for reproducibility  

- **`data_dictionary.md`** - Data schema documentation
  - Column descriptions and data types
  - Value ranges and constraints
  - Data quality assumptions  

- **`risk_and_safety.md`** - Risk assessment and safety guidelines
  - Model limitations and biases
  - Ethical considerations
  - Safety precautions and guardrails
  - Compliance and regulatory notes  

### `notebooks/`
Jupyter notebooks for exploration and analysis.

- Ad-hoc analysis
- Results visualization
- Development notebooks

### `src/`
Core application source code.

- Reusable modules and utilities
- Main pipeline logic
- Data processing functions

### `tools/`
Utility scripts and CLI tools.

- Data preprocessing scripts
- Evaluation runners
- Administrative utilities

### `references/`
Reference materials and external resources.

- Academic papers
- External documentation
- Background materials

## Key Features

- **Reproducible Experimentation**: Every run captures commit hash, prompt ID, model ID, and dataset snapshot
- **Rigorous Evaluation**: Comprehensive schema, multiple metrics, and automatic reports
- **Prompt Engineering**: Versioned prompt library with lineage tracking
- **Production Ready**: Dashboard and monitoring capabilities
- **Language Support**: Specialized for Brazilian Portuguese and Canadian French
- **Data Privacy**: Built-in deidentification pipeline

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `pyproject.toml`

### Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and configuration
```

### Quick Start

1. Place raw PDFs in `data/raw/`
2. Configure your model in `models/configs/`
3. Select or create prompts in `prompts/library/`
4. Run experiments and track results in `experiments/runs/`
5. View results and metrics in `reports/`

## Documentation

- **Protocol & Methodology**: See `docs/protocol.md`
- **Data Schema**: See `docs/data_dictionary.md`
- **Risk Assessment**: See `docs/risk_and_safety.md`

## License

See LICENSE file for details.

---

Built with focus on reproducibility, safety, and production-grade evaluation methodology.