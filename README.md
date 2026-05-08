# LLM (Large Language Model) Summarization BR/CA

Prompt-technique evaluation for feature-level **comparison of human and LLM-based clinical feature extraction** in breast cancer radiology and pathology documentation.

## Project Overview

This repository evaluates how reliably LLMs extract structured clinical features from unstructured/scanned source documents.

- **Primary goal:** reduce clinically unsafe extraction errors, especially hallucinated features.
- **Core comparison:** Human annotator vs LLM annotator against source-document ground truth.
- **Dataset scale:** 200 de-identified patient cases, 14 clinical elements, 45 core columns.
- **Primary safety metric:** fabrication rate.

## Research Questions

1. Which clinical elements are most fragile for LLM extraction?
2. Where does AI fabrication/omission differ significantly from human performance?
3. How do prompt strategies (zero-shot, Chain-of-Thought (CoT), Retrieval-Augmented Generation (RAG), few-shot, Program-Aided Language (PAL)-style prompting, Reason+Act (ReAct), etc.) change outcomes?
4. Which document and OCR features predict AI failure?

## What Is in This Repository

- End-to-end analysis notebooks for data quality, diagnostics, modeling, and validation.
- Prompt artifacts and prompt-library resources for extraction experiments.
- Evaluation scripts for human-vs-LLM comparisons and LLM-as-judge workflows.
- Reports and figures generated from notebook and pipeline outputs.
- Supporting manuscript and conference materials.

## Repository Layout

```text
<project_root>
├── docs/                 # project docs, architecture notes, summaries, guides
├── notebooks/            # main analysis notebooks
├── src/                  # pipeline/evaluation/modeling scripts
├── prompts/              # prompt library, prompt assets, generated prompts
├── eval/                 # evaluation schemas and metric resources
├── experiments/          # run tracking structure
├── reports/              # generated tables/plots/analysis outputs
├── models/               # model configs
├── references/           # papers and technical references
└── conferences/          # conference submission material
```

## Documentation Guide

Start here based on your need:

- **High-level summary:** `docs/executive_summary.md`
- **Dataset details:** `docs/dataset_metadata.md`
- **LangGraph framework and RAG design:** `docs/langgraph_agentic_rag_design.md`
- **LangChain ecosystem architecture:** `docs/ARCHITECTURE_LANGCHAIN_ECOSYSTEM.md`
- **Metric selection decisions:** `docs/FINAL_METRICS_SELECTION.md`, `docs/hcat_metrics_selection.md`
- **Colab execution workflow:** `docs/colab_pipeline_guide.md`
- **Project structure + privacy rules:** `docs/project_directory_structure_privacy_rules.md`

## Data and Labeling Conventions

- **Source label:** feature present/absent in source documents.
- **Annotator coding:**
  - `1` = Correct extraction
  - `2` = Omission
  - `3` = Fabrication
  - `N/A` = Not applicable
- **Domains covered:** radiology and pathology elements.

## Environment and Setup

### Requirements

- Python 3.11+
- Local dependencies from `pyproject.toml` or `requirements.txt`
- API keys via `.env` (do not commit secrets)

### Installation

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. Prepare/confirm de-identified inputs.
2. Run notebooks in sequence for preprocessing, diagnostics, extraction analysis, and validation.
3. Generate updated outputs in `reports/`.
4. Compare metrics across prompt variants and evaluation methods.

## Privacy and Safety

- Never commit PHI or identifiable patient data.
- Keep sensitive data in controlled private storage.
- Commit only de-identified/approved derived outputs.
- Store credentials only in `.env`.

## Project Status

Active research and experimentation repository with ongoing updates to prompts, evaluation methods, and reporting artifacts.
