# References Index

Annotated bibliography of reference materials used in the LLM Summarization BR/CA project.

---

## Academic Papers

### 1. Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation
- **File:** `framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation.pdf`
- **Relevance:** Core methodological reference. Provides a structured framework for evaluating hallucination (fabrication) rates in LLM-generated clinical summaries. Directly informs our fabrication rate metric design, safety outcome definitions, and the element-level evaluation approach used across Notebooks 03 and 07.
- **Key Contributions:**
  - Taxonomy of hallucination types in medical summarisation (omission, fabrication, distortion)
  - Safety-oriented evaluation protocol for clinical NLP systems
  - Metric definitions aligned with clinical risk assessment

### 2. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- **File:** `Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.pdf`
- **Relevance:** Foundational paper on using LLMs as automated evaluators ("LLM-as-a-Judge"). Informs the LLM evaluation layer in our project — how to use one LLM to judge the output quality of another, including bias mitigation strategies and agreement metrics with human judges.
- **Key Contributions:**
  - MT-Bench multi-turn benchmark for LLM evaluation
  - Position bias, verbosity bias, and self-enhancement bias in LLM judges
  - Agreement rates between LLM judges and human annotators (~80% on MT-Bench)

### 3. How Can Artificial Intelligence Decrease Cognitive (and Other) Biases
- **File:** `How_can_artificial_intelligence_decrease_cognitive.pdf` (duplicate: `How_can_artificial_intelligence_decrease_cognitive (1).pdf`)
- **Relevance:** Discusses how AI systems can both introduce and mitigate cognitive biases in clinical decision-making. Relevant to understanding why human annotators and AI annotators may systematically disagree on clinical feature extraction, and how prompt design can reduce extraction bias.
- **Key Contributions:**
  - Catalogue of cognitive biases relevant to medical AI (anchoring, confirmation, availability)
  - Framework for AI-assisted debiasing in clinical workflows
  - Implications for annotator training and inter-rater reliability

### 4. Review of Large Language Models for Patient and Caregiver Support in Cancer Care Delivery (Kouzy et al., 2025)
- **File:** `kouzy-et-al-2025-review-of-large-language-models-for-patient-and-caregiver-support-in-cancer-care-delivery.pdf`
- **Relevance:** Recent review of LLM applications in oncology care delivery. Provides clinical context for our breast cancer summarisation use case — where LLMs are being deployed, what safety concerns exist, and what validation standards are expected in cancer care settings.
- **Key Contributions:**
  - Survey of LLM use cases in oncology (patient communication, clinical documentation, decision support)
  - Safety and accuracy benchmarks in cancer-specific NLP
  - Regulatory and institutional considerations for LLM deployment in cancer centres

---

## Prompt Engineering References

### 5. Automated Prompt Engineering: The Definitive Hands-On Guide (Towards Data Science)
- **File:** `Automated Prompt Engineering_ The Definitive Hands-On Guide _ Towards Data Science.pdf`
- **Relevance:** Comprehensive guide to automated prompt engineering techniques. Informs our prompt library design (9 prompt versions), the prompt escalation process documented in the executive summary, and the systematic prompt iteration tracked in Notebook 04's prompt history section.
- **Key Contributions:**
  - Automated prompt search and optimisation algorithms
  - Prompt templating best practices
  - Evaluation-driven prompt refinement workflows

### 6. LLMs as Prompt Optimizers
- **File:** `llms as prompt optimizers.pdf`
- **Relevance:** Research on using LLMs themselves to generate and refine prompts. Relevant to the program-aided and ReAct prompt variants in our prompt library, where the LLM is given meta-instructions about how to approach the extraction task.
- **Key Contributions:**
  - Self-optimising prompt generation pipelines
  - Comparison of human-authored vs LLM-generated prompts
  - Iterative prompt refinement using LLM feedback loops

### 7. Prompt Engineering Techniques Taxonomy (Infographic)
- **File:** `classes_prompt_techniques.webp`
- **Relevance:** Visual taxonomy of prompt engineering techniques organised into three tiers: Single Prompt Techniques (zero-shot, few-shot, chain-of-thought, program-aided language, RAG), LLM with External Tools (ReAct, Reflexion), and Multiple Prompt Techniques (tree of thoughts, self-consistency, least-to-most, prompt chaining, directional stimulus, generated knowledge). This taxonomy directly maps to our prompt library's 9 variants.
- **Techniques Shown:**
  - **Single:** Zero-Shot, Few-Shot, Chain-of-Thought, Program-Aided Language, RAG
  - **External Tools:** ReAct, Reflexion
  - **Multiple:** Tree of Thoughts, Self-Consistency, Least-to-Most, Prompt Chaining, Directional Stimulus, Generated Knowledge

---

## Machine Learning & Modeling References

### 8. H2O.ai GLM Overview
- **File:** `H2Oai_GLM_Overview.pdf`
- **Relevance:** Technical documentation for H2O's Generalized Linear Model implementation. Referenced in Notebook 05's feature interaction analysis, which uses H2O's XGBoost and GBM estimators to identify document-level predictors of AI extraction errors.
- **Key Contributions:**
  - H2O AutoML and GLM API reference
  - Regularisation options (L1/L2/elastic net) for high-dimensional feature spaces
  - Variable importance extraction methods

### 9. Model Performance Over Time Plot
- **File:** `model performance over time plot.pdf`
- **Relevance:** Visual reference for time-series model performance tracking. Inspired the prompt history metric trend plots in Notebook 04 (Section 4.9), which track accuracy, fabrication rate, and omission rate across prompt versions.

---

## Formatting & Style References

### 10. APA Table Setup Reference
- **File:** `table-setup-image_tcm11-262906_w1024_n.jpg`
- **Relevance:** APA 7th edition table formatting guide showing proper table anatomy (table number, title, column headings, spanners, stubs, notes). Used as a style reference for tables generated in the data dictionary (Notebook 06) and reports.
- **Elements Shown:** Table number, title, stub headings, column spanners, decked heads, table body, general/specific/probability notes

---

## Setup & Infrastructure

### 11. Environment Setup Guide
- **File:** `environment_setup.md`
- **Relevance:** Step-by-step guide for setting up the development environment on macOS with VS Code: virtual environment creation (Python 3.12), dependency installation, API key configuration (OpenAI, Anthropic, Phoenix), VS Code interpreter binding, and notebook/script execution. Serves as the onboarding document for new collaborators.
- **Sections:** Prerequisites, VS Code setup, venv creation, dependency management, environment variables, verification, running scripts/notebooks

---

## Notes

- `How_can_artificial_intelligence_decrease_cognitive (1).pdf` is a duplicate of `How_can_artificial_intelligence_decrease_cognitive.pdf` — consider removing one.
- All API keys in `environment_setup.md` have been redacted to placeholder values for repository safety.
