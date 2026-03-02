# References Index

Annotated bibliography of reference materials used in the LLM Summarization BR/CA project.

---

## Academic Papers

### 1. Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Text Summarisation
- **File:** `_framework to assess clinical safety and hallucination rates of LLMs for medical text summarisation.pdf`
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

---

## LLM Evaluation & Judging

### 12. Judge's Verdict: A Comprehensive Review of LLM Evaluation
- **File:** `21214_Judge_s_Verdict_A_Compre.pdf`
- **Relevance:** Comprehensive overview of methods for evaluating LLM outputs, with a focus on judge-based evaluation frameworks. Complements the LLM-as-a-Judge reference (entry 2) and informs the correctness scoring design used in our DeepEval-based evaluation layer.
- **Key Contributions:**
  - Taxonomy of LLM evaluation paradigms (automated, human, hybrid)
  - Critique of LLM judge biases and mitigation strategies
  - Recommendations for reproducible evaluation pipelines

### 13. LLM-as-a-Judge Bot Comparisons
- **File:** `LLM_as_judge_bot_comparisons.pdf`
- **Relevance:** Comparative analysis of different LLM models used as automated judges. Used to select and justify the choice of evaluation model in our automated quality assessment pipeline, and to understand inter-judge agreement patterns.
- **Key Contributions:**
  - Side-by-side judge performance comparison across GPT-4, Claude, and open-source models
  - Agreement rates with human reviewers per task type
  - Practical guidance for choosing a judge model in production eval pipelines

### 14. A Framework to Assess Clinical Safety and Hallucination Rates of LLMs for Medical Summarisation
- **File:** `a framework to assess clinical safety and hallucination rates of LLMs for medical summarisation.pdf`
- **Relevance:** Alternative version / earlier edition of the clinical safety framework (see entry 1). Useful for tracing the evolution of the hallucination taxonomy and comparing metric definitions across versions. Directly relevant to fabrication rate calculations in the project dataset.
- **Key Contributions:**
  - Early definition of omission vs. fabrication in clinical LLM outputs
  - Safety-oriented scoring rubric for clinical NLP systems
  - Comparison of safety rates across LLM architectures

### 15. LLM Judge Performance — Answer Location & Transcript Analysis
- **Files:** `llm_judge/answer_location_transcript_llm_judge_performance.pdf`, `llm_judge/fine_tuned_bot_performance_human_agreement.pdf`
- **Relevance:** Internal/project-level empirical analyses of judge model performance. Documents how well the LLM judge locates correct answers within source text, and how fine-tuning the bot affects human-LLM agreement rates. Directly informs the evaluation design in Notebook 07.
- **Key Contributions:**
  - Empirical answer-location accuracy across element types
  - Human–AI agreement rates before and after fine-tuning
  - Failure mode analysis for LLM-as-judge on clinical text

---

## Deep Learning & NLP Foundations

### 16. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding
- **File:** `bert pretraining transformer.pdf`
- **Relevance:** Foundational BERT paper. Provides the theoretical basis for the BERT fine-tuning approach explored in Notebook 05 for feature extraction from source medical documents.
- **Key Contributions:**
  - Masked language model (MLM) and next sentence prediction (NSP) pretraining objectives
  - Bidirectional transformer architecture
  - Transfer learning from general-domain text to downstream NLP tasks

### 17. BERT Fine-Tuning for Downstream Tasks
- **File:** `bert fine tuning.pdf`
- **Relevance:** Practical guide to fine-tuning BERT models on classification and extraction tasks. Referenced in Notebook 05 (OCR + BERT feature extraction pipeline), where BERT is used to extract structured features from raw source document text.
- **Key Contributions:**
  - Fine-tuning methodology for classification, NER, and question answering
  - Hyperparameter recommendations for small medical datasets
  - Comparison of full fine-tuning vs. layer-freezing approaches

### 18. Transformer Model — Attention Mechanism (MIT Lecture)
- **File:** `transformer_model_attention_mit_lecture.pdf`
- **Relevance:** MIT lecture slides on the Transformer architecture and attention mechanism. Provides the theoretical background for understanding how GPT-4 and Claude process medical document text during structured extraction.
- **Key Contributions:**
  - Scaled dot-product attention and multi-head attention explained
  - Positional encoding and its role in sequence processing
  - Encoder-decoder vs. decoder-only architectures

### 37. Attention Is All You Need (Vaswani et al., 2017)
- **File:** `NIPS-2017-attention-is-all-you-need-Paper.pdf`
- **Relevance:** The original Transformer architecture paper (NeurIPS 2017). Foundational reference for understanding the self-attention mechanism underlying GPT-4, Claude, and BERT — all models used in this project's extraction and evaluation pipelines.
- **Key Contributions:**
  - Introduced the Transformer: multi-head self-attention replacing recurrence and convolution
  - Positional encoding, feed-forward sublayers, and layer normalisation
  - Established encoder-decoder architecture now used in all modern LLMs

### 38. Efficient Estimation of Word Representations in Vector Space (Word2Vec — Mikolov et al., 2013)
- **File:** `1301.3781v3.pdf`
- **Relevance:** The Word2Vec paper (arXiv 1301.3781). Foundational reference for dense word embeddings. Provides theoretical background for the text vectorisation approaches benchmarked in Notebook 07 and for understanding embedding-based feature representations.
- **Key Contributions:**
  - Skip-gram and CBOW architectures for learning word embeddings
  - Semantic and syntactic relationships captured in vector space
  - Basis for downstream embedding models (GloVe, FastText, BERT)

### 19. GLUE Benchmark
- **File:** `glue_benchmark.pdf`
- **Relevance:** The General Language Understanding Evaluation (GLUE) benchmark. Referenced when assessing model selection for the LLM evaluation layer — models are compared against GLUE tasks as a general NLP capability proxy.
- **Key Contributions:**
  - Multi-task benchmark covering NLI, sentiment, similarity, and QA
  - Baseline performance for BERT, GPT, and other transformer models
  - Diagnostic test suite for linguistic phenomena

### 20. Long Short-Term Memory (LSTM) RNNs for Sequence Classification
- **File:** `long short term memory rnn for sequence classification.pdf`
- **Relevance:** Reference on LSTM-based sequence classifiers. Provides historical context for recurrent approaches to clinical text classification that preceded transformer-based methods used in this project.
- **Key Contributions:**
  - LSTM cell architecture (input, forget, output gates)
  - Application to clinical sequence labeling and note classification
  - Comparison with simpler RNN architectures

### 21. Neural Networks and Deep Learning
- **File:** `Neural networks and deep learning.pdf`
- **Relevance:** Core deep learning textbook reference (Nielsen). Provides theoretical grounding for the neural network components used in feature importance models (XGBoost, MLP) in the classifier analysis pipeline.
- **Key Contributions:**
  - Backpropagation, gradient descent, and regularisation fundamentals
  - Convolutional and recurrent network architectures
  - Practical training techniques (dropout, batch normalisation)

### 22. Activation Functions — ML Glossary
- **File:** `Activation Functions — ML Glossary documentation.pdf`
- **Relevance:** Reference documentation on activation functions (ReLU, sigmoid, tanh, softmax). Used when designing and interpreting the MLP classifier in the feature importance analysis pipeline.
- **Key Contributions:**
  - Definitions and mathematical formulations of common activation functions
  - Guidance on activation function selection by layer type and task
  - Vanishing gradient problem and its relationship to activation choice

### 23. Gradient Descent for Machine Learning
- **File:** `Gradient Descent For Machine Learning - MachineLearningMastery.com.pdf`
- **Relevance:** Accessible reference on gradient descent variants (batch, stochastic, mini-batch). Referenced in the training process of XGBoost and neural network models used in the feature importance and fabrication prediction analyses.
- **Key Contributions:**
  - Intuitive explanation of cost function minimisation
  - Comparison of SGD, Adam, and RMSProp optimisers
  - Learning rate tuning and convergence diagnostics

---

## Machine Learning: Classification Models for Feature Importance

### 24. XGBoost Text Classification vs. LLMs
- **File:** `xgb text classification vs llm.pdf`
- **Relevance:** Empirical comparison of XGBoost-based classifiers against LLMs for structured text classification tasks. Directly informs the model selection rationale in the feature importance pipeline, where XGBoost is used to predict AI extraction errors from document-level features.
- **Key Contributions:**
  - Performance comparison across tabular and text classification benchmarks
  - Computational efficiency trade-offs between XGBoost and fine-tuned LLMs
  - Use cases where traditional ML outperforms LLMs on structured inputs

### 25. TensorFlow ResNet: Building, Training and Scaling Residual Networks
- **File:** `TensorFlow ResNet_ Building, Training and Scaling Residual Networks on TensorFlow - MissingLink.ai.pdf`
- **Relevance:** Technical reference for implementing ResNet architectures in TensorFlow. Background reference for deep learning infrastructure used in Notebook 05's BERT fine-tuning pipeline.
- **Key Contributions:**
  - Residual connection design and skip layer implementation
  - Training and scaling deep networks in TensorFlow
  - Transfer learning using pre-trained ResNet weights

### 26. Feature Importance Classification Models (Subdirectory)
- **Directory:** `feature_importance_prediction_accurate_omission_fabrication/`
- **Relevance:** Collection of visual guides and tutorials for the classification models used to predict AI extraction accuracy, omission rates, and fabrication rates from document-level features.
- **Files:**
  - `Bernoulli Naive Bayes, Explained...pdf/.txt` — Binary feature naive Bayes classifier
  - `Decision Tree Classifier, Explained...pdf/.txt` — Tree-based classification with visual guide
  - `Dummy Classifier Explained...pdf/.txt` — Baseline model reference for performance benchmarking
  - `Gaussian Naive Bayes, Explained...pdf/.txt` — Continuous feature naive Bayes
  - `K Nearest Neighbor Classifier _ TDS Archive.pdf/.txt` — KNN classifier with TDS examples
  - `Logistic Regression, Explained...txt` — Logistic regression for binary outcomes
  - `Multilayer Perceptron, Explained...pdf/.txt` — MLP neural network classifier
  - `PCA in KNN_ Gaussian Naive Bayes.pdf` — Dimensionality reduction applied to KNN and Naive Bayes
  - `Support Vector Classifier, Explained...pdf/.txt` — SVC with kernel methods
  - `read this to understand different predictive models_classification_timetoevent.pdf` — Overview of classification vs. time-to-event models
  - `supervised learning models by type image.webp` — Visual taxonomy of supervised learning models

---

## Clinical NLP & Radiology/Pathology Informatics

### 39. Natural Language Processing for Breast Imaging: A Systematic Review
- **File:** `Natural Language Processing for Breast Imaging SR.pdf`
- **Relevance:** Systematic review of NLP applied specifically to breast imaging reports. Directly situates the project within the clinical literature — provides a survey of what NLP tasks have been applied to mammography and breast MRI/US reports, accuracy benchmarks, and known challenges with unstructured radiology text.
- **Key Contributions:**
  - Survey of NLP tasks on breast imaging reports (classification, IE, summarisation)
  - Accuracy benchmarks and failure modes in radiology NLP
  - Methodological considerations specific to breast imaging report structure

### 40. Natural Language Processing in Radiology: A Systematic Review (Pons et al., 2016)
- **File:** `pons-et-al-2016-natural-language-processing-in-radiology-a-systematic-review.pdf`
- **Relevance:** Broad systematic review of NLP methods applied to radiology reports. Directly relevant to Notebook 04's text extraction pipeline and the broader context of processing scanned radiology reports in breast cancer cases.
- **Key Contributions:**
  - Taxonomy of NLP approaches used in radiology (rule-based, ML, deep learning)
  - Survey of radiology NLP tasks: IE, coding, summarisation, de-identification
  - Benchmarks and limitations specific to radiology text processing

### 41. Clinical Informatics Reference (JAMIA)
- **File:** `amiajnl1560.pdf`
- **Relevance:** Journal of the American Medical Informatics Association (JAMIA) paper (ID: amiajnl-1560). Likely addresses clinical NLP, EHR-based feature extraction, or informatics methods directly relevant to structured data extraction from clinical documents.
- **Note:** Filename suggests JAMIA publication; exact title and authors unconfirmed without reading the PDF. Review and update this entry with full citation.

---

## Oncology-Specific References

### 27. Large Language Models in Oncology
- **File:** `general_uses_llm_onc/Large language models in oncology.pdf`
- **Relevance:** Review of LLM applications across the oncology domain. Provides broader clinical context for the breast cancer surgical summarisation use case, and situates the project within the emerging field of oncology NLP.
- **Key Contributions:**
  - Survey of LLM use cases across cancer types (diagnosis, treatment planning, documentation)
  - Accuracy benchmarks in oncology-specific NLP tasks
  - Safety concerns and regulatory considerations in oncology AI deployment

---

## LLM Overview References

### 42. Compact Guide to Large Language Models
- **File:** `compact-guide-to-large-language-models.pdf`
- **Relevance:** Concise reference guide covering the architecture, training, and deployment of LLMs. Useful as a background reference for the introduction and methods section of the manuscript, and for onboarding collaborators to the LLM extraction pipeline context.
- **Key Contributions:**
  - Overview of LLM architectures (GPT, Claude, open-source models)
  - Pre-training, fine-tuning, and RLHF concepts
  - Practical deployment and safety considerations

### 43. NLP-Progress: English Summarization Benchmarks
- **File:** `NLP-progress_english_summarization.md at master � sebastianruder_NLP-progress.pdf`
- **Relevance:** Sebastian Ruder's NLP-Progress tracker for English summarization tasks. Provides a reference frame for state-of-the-art summarization performance across standard benchmarks (CNN/DailyMail, XSum, etc.), useful for contextualising the project's extraction accuracy metrics relative to the broader field.
- **Key Contributions:**
  - Benchmark leaderboard for abstractive and extractive summarization
  - ROUGE and human evaluation scores across model families
  - Links to datasets and evaluation protocols

---

## Prompt Engineering Resources

### 28. Prompt Optimization Reference
- **File:** `prompt engingeering optimization.pdf`
- **Relevance:** Reference on prompt optimization techniques including iterative refinement, evaluation-driven prompt search, and structured prompt templates. Informs the 9-version prompt library and the prompt iteration tracking documented in Notebook 04.
- **Key Contributions:**
  - Evaluation-driven prompt iteration methodology
  - Structured prompt template design for extraction tasks
  - Optimization strategies for clinical information extraction

### 29. Prompt Library Reference Materials
- **Directory:** `Prompts/`
- **Relevance:** Source prompt materials and reference documents used to develop the project's structured extraction prompt library.
- **Files:**
  - `Initial prompt for extraction.docx` — First-generation extraction prompt used at project inception
  - `mcode gpt zeroshot extraction.pdf` — mCODE-structured zero-shot extraction with GPT; used as a benchmark for structured oncology extraction
  - `mcode_structure.xlsx` — mCODE (minimal Common Oncology Data Elements) data dictionary; defines standardised oncology data fields relevant to the extraction template
  - `prompt_library (1).csv` — Compiled prompt library across versions 1–9 with task descriptions and performance notes

---

## Setup & Infrastructure

### 30. Installing Packages with pip and venv (Python Packaging User Guide)
- **File:** `Install packages in a virtual environment using pip and venv - Python Packaging User Guide.pdf`
- **Relevance:** Official Python packaging guide for virtual environment setup using `pip` and `venv`. Supplements the `environment_setup.md` guide for new collaborators setting up the project environment.
- **Key Contributions:**
  - Step-by-step virtual environment creation and activation on macOS/Windows
  - Dependency installation and `requirements.txt` management
  - Best practices for isolating project dependencies

### 31. Complete Guide to Artificial Neural Network Concepts and Models
- **Files:** `Complete Guide to Artificial Neural Network Concepts & Models.pdf`, `complete guide to ann.pdf`
- **Relevance:** Comprehensive ANN reference covering network architectures, training procedures, and regularisation. Background reference for the neural network components used in the feature importance classifiers.
- **Note:** Two copies exist — `complete guide to ann.pdf` may be an earlier version or alternate format of the same material.

---

## Miscellaneous

### 32. Target Trial Framework for Causal Inference in Observational Studies
- **File:** `target trial casual inference observational studies.pdf`
- **Relevance:** Methodological reference on the target trial emulation framework for causal inference. Relevant background for interpreting observational performance differences between human and AI annotators without randomised assignment.
- **Key Contributions:**
  - Target trial emulation design for non-randomised data
  - Bias sources in observational comparisons
  - Counterfactual reasoning in evaluating annotator accuracy

### 33. Project Presentation (WIP)
- **File:** `final_optics_wip_presentation.pptx`
- **Relevance:** Work-in-progress project presentation slides summarising study design, methods, and early results. Useful for communicating project scope to collaborators and stakeholders.

### 34. Bot Question File Reference
- **File:** `file_names_bot_questions.txt`
- **Relevance:** Text file documenting the file naming conventions and question structures used when querying the LLM bot during structured extraction. Serves as a quick reference for understanding the input format expected by the summarisation pipeline.

### 35. Reference Image
- **File:** `IMG_6047.PNG`
- **Relevance:** Supporting reference image captured during project development. Content likely documents a workflow step, output, or system configuration relevant to the extraction pipeline.

---

## Marginal / Background Only (Not Formally Citable)

### 44. Deep Learning Neural Networks Explained (Medium — Sanjay Singh)
- **File:** `Deep Learning Neural Networks Explained_ ANN, CNN, RNN, and Transformers (Basic Understanding) _ by Sanjay Singh _ Medium.pdf`
- **Relevance:** General-audience Medium article covering ANN, CNN, RNN, and Transformer architectures. Background reading only — not formally citable in a manuscript. Useful for onboarding or explaining concepts to non-technical collaborators.
- **Status:** 🟡 Background/tutorial — do not cite in manuscript

### 45. RNN vs LSTM vs GRU vs Transformers (GeeksforGeeks)
- **File:** `RNN vs LSTM vs GRU vs Transformers - GeeksforGeeks.pdf`
- **Relevance:** GeeksforGeeks tutorial comparing sequential model architectures. Background reading only. Provides historical context for why transformer-based LLMs supersede earlier RNN/LSTM approaches for clinical text extraction.
- **Status:** 🟡 Background/tutorial — do not cite in manuscript

---

## Future Value (Not Currently Needed)

### 46. Semiparametric Accelerated Failure Time Partial Linear Model
- **File:** `Semiparametric Accelerated Failure Time Partial Linear Model.pdf`
- **Relevance:** Statistical methodology paper on semiparametric AFT models. Not relevant to the current extraction/evaluation pipeline. **Future value** if the project is extended to include time-to-event outcomes (e.g., survival analysis, treatment delay modelling) or causal inference framing.
- **Status:** 🔵 Keep — future value only

---

## Internal Project Documents (Not References)

### 47. Presentation Outline (Internal)
- **File:** `presentation outline.docx`
- **Relevance:** Internal project presentation draft/outline. Not a research reference. Consider moving to `docs/` or `conferences/` if relevant to a specific presentation.
- **Status:** ❌ Not a reference — misplaced in references/

---

## Resolved Archives

### 36. Windows Local References Archive (Extracted)
- **File:** `references_from_windows_local_pending_sort.7z`
- **Status:** ✅ **Extracted and indexed** — all 11 files have been extracted to `references/` and indexed as entries 37–47 above. The `.7z` archive may now be deleted or retained as a backup.

---

## Notes

- `How_can_artificial_intelligence_decrease_cognitive (1).pdf` is a duplicate of `How_can_artificial_intelligence_decrease_cognitive.pdf` — consider removing one.
- All API keys in `environment_setup.md` have been redacted to placeholder values for repository safety.
- `validaton/` directory from the source project references folder was **intentionally excluded** from this repository — it contains patient-identifiable data (MRNs, validation tables) and must not be committed to version control.
- `complete guide to ann.pdf` and `Complete Guide to Artificial Neural Network Concepts & Models.pdf` appear to be duplicates — consider consolidating.
- Entry 1 filename corrected: actual file on disk has a leading underscore (`_framework to assess...`); this differs from earlier versions of the index.
- `ocr_deblur_metrics_prepost.py` is a **Python script** found in `references/` — this does not belong here and should be moved to `src/` or `tools/`. It is not a reference.
- `presentation outline.docx` (entry 47) is an internal project document — consider moving to `docs/` or a `conferences/` subfolder.
- Entries 44–45 (Medium, GeeksforGeeks) are background tutorials only — **do not cite in manuscript**.
