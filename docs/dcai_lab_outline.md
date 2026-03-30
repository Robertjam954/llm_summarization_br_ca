# Outline: dcai-course/dcai-lab Repository

**Source:** https://github.com/dcai-course/dcai-lab/tree/master  
**Course:** [Introduction to Data-Centric AI](https://dcai.csail.mit.edu/) (MIT CSAIL)

---

## Repository Overview

This repository contains 9 lab assignments for MIT's *Introduction to Data-Centric AI* course. Each lab is a Jupyter notebook that illustrates a core data-centric AI concept — improving model performance by improving the data rather than the model architecture. Labs come with associated datasets, solution notebooks, and (where applicable) supporting Python files.

---

## Repository Structure

```
dcai-lab/
├── data_centric_model_centric/   # Lab 1
├── label_errors/                 # Lab 2
├── dataset_curation/             # Lab 3
├── data_centric_evaluation/      # Lab 4
├── outliers/                     # Lab 5
├── growing_datasets/             # Lab 6
├── interpretable_features/       # Lab 7
├── prompt_engineering/           # Lab 8
├── membership_inference/         # Lab 9
└── README.md
```

---

## Lab 1 — Data-Centric AI vs Model-Centric AI

**Notebook:** `data_centric_model_centric/Lab - Data-Centric AI vs Model-Centric AI.ipynb`  
**Data:** `reviews_train.csv`, `reviews_test.csv` (magazine product reviews labelled good/bad)  
**Dependencies:** `scikit-learn`, `pandas`

### Goal
Build a binary text classifier for product reviews and demonstrate that improving the *data* (removing noisy HTML-corrupted samples) outperforms purely model-centric tuning.

### Pipeline
`CountVectorizer → TfidfTransformer → SGDClassifier` (scikit-learn `Pipeline`)

### Functions

| Function | Signature | Description |
|---|---|---|
| `evaluate` | `evaluate(clf)` | Predicts on test set and prints `Accuracy: XX.X%` |
| `is_bad_data` | `is_bad_data(review: str) -> bool` | Student-implemented heuristic — returns `True` if a review contains HTML artifacts |

### Outputs

| Output | Value |
|---|---|
| Baseline accuracy (noisy data) | ~76.5% |
| Accuracy after removing bad data | Significantly higher (exercise-dependent) |
| Printed format | `Accuracy: 76.5%` |

### Exercises
1. Train a more accurate model without changing the data (try different classifiers / hyperparameter search / ensembles)
2. Inspect training data for patterns in bad examples
3. Implement `is_bad_data()` to filter HTML-corrupted records; retrain and evaluate

---

## Lab 2 — Label Errors (Confident Learning)

**Notebook:** `label_errors/Lab - Label Errors.ipynb`  
**Data:** `student-grades.csv` — 3 exam scores, notes, and a noisy letter grade (20% intentionally corrupted)  
**Dependencies:** `xgboost`, `scikit-learn`, `pandas`, `cleanlab`

### Goal
Automatically identify mislabeled data using Confident Learning, remove the errors, and show the resulting accuracy improvement without changing the model.

### Model
`XGBClassifier(tree_method="hist", enable_categorical=True)`

### Functions

| Function | Signature | Description |
|---|---|---|
| `compute_class_thresholds` | `compute_class_thresholds(pred_probs: np.ndarray, labels: np.ndarray) -> np.ndarray` | Student-implemented. Returns array of length K with per-class thresholds (average self-confidence per class) |
| `compute_confident_joint` | `compute_confident_joint(pred_probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray` | Student-implemented. Returns K×K matrix counting label co-occurrences above threshold |
| `cross_val_predict` | `cross_val_predict(model, train_data, train_labels, method='predict_proba', cv=10)` | Scikit-learn utility — produces out-of-sample predicted probabilities (N×5 matrix) |
| `find_label_issues` | `cleanlab.filter.find_label_issues(train_labels, pred_probs, return_indices_ranked_by='self_confidence')` | One-line Cleanlab alternative to the full student exercise |

### Outputs

| Output | Example Value |
|---|---|
| Estimated noise rate | `Estimated noise rate: 19.X%` |
| Percentage of true errors found | `Percentage of errors found: 83.X%` |
| Accuracy with original (noisy) data | `Accuracy with original data: 79.2%` |
| Accuracy after removing label errors | `Accuracy with errors removed: 86.X%` |
| Error reduction | `Reduction in error: 35.X%` |

### Exercises
1. Compute out-of-sample predicted probabilities using 10-fold cross-validation
2. Implement `compute_class_thresholds()`
3. Implement `compute_confident_joint()`
4. Count number of label issues from the off-diagonal sum of the confident joint
5. Rank data by self-confidence; extract the top `num_label_issues` indices; remove and retrain

---

## Lab 3 — Dataset Creation and Curation (Multiple Annotators)

**Notebook:** `dataset_curation/Lab - Dataset Curation.ipynb`  
**Dependencies:** `scikit-learn`, `pandas`, `numpy`, `cleanlab`

### Goal
Analyze a classification dataset labeled by multiple crowd-sourced annotators. Compare naive majority-vote aggregation against the CROWDLAB algorithm for consensus label estimation.

### Functions

| Function | Signature | Description |
|---|---|---|
| `make_data` | `make_data(sample_size=300)` | Generates a synthetic 3-class, 2D classification dataset with 50 simulated noisy annotators. Returns features `X`, ground truth, and annotator label matrix |
| `train_model` | `train_model(labels_to_fit)` | Trains a K-Nearest Neighbours classifier via cross-validation; returns out-of-sample predicted class probabilities |

### Outputs
- Annotator label matrix (rows = examples, columns = annotators, values = class int or `NA`)
- Majority-vote consensus label accuracy vs. ground truth
- CROWDLAB consensus label accuracy vs. ground truth
- Per-annotator quality estimates

### Exercises
1. Compute majority-vote consensus labels; evaluate accuracy; estimate annotator quality
2. Apply CROWDLAB (`cleanlab.multiannotator`) for improved consensus labels and annotator quality estimates

---

## Lab 4 — Data-Centric Evaluation of ML Models

**Notebook:** `data_centric_evaluation/Lab - Data-Centric Evaluation.ipynb`  
**Data:** `train.csv`, `test.csv` — 3-class classification task with features `x1`–`x5` and target `y`  
**Dependencies:** `cleanlab`, `matplotlib`, `scikit-learn`, `pandas`, `numpy`

### Goal
Improve a fixed neural network model's accuracy *solely* by improving the training data — without altering model architecture or hyperparameters.

### Functions

| Function | Signature | Description |
|---|---|---|
| `train_evaluate_model` | `train_evaluate_model(X, y, X_test, y_test)` | Trains `MLPClassifier(early_stopping=True, random_state=SEED)`, prints `Balanced accuracy = X.XX`, returns predictions |

### Outputs

| Output | Example |
|---|---|
| Balanced accuracy (baseline) | `Balanced accuracy = 0.XX` |
| Scatter plot | 2D scatter of first two features, coloured by class |
| Goal accuracy | >80% balanced accuracy achievable without model changes |

### Exercise
Modify `X` and `y` (e.g. remove label errors, fix feature scaling, add/drop features) so that `train_evaluate_model(my_X, my_y, X_test, y_test)` reports higher performance. Apply identical feature transformations to `X_test`.

---

## Lab 5 — Class Imbalance, Outliers, and Distribution Shift

**Notebook:** `outliers/Lab - Outliers.ipynb`  
**Solution:** `outliers/Solution - Outliers.ipynb`  
**Dependencies:** `torch`, `torchvision`, `scikit-learn`, image processing libraries

### Goal
Implement and compare multiple outlier/anomaly detection methods on a dog-image dataset. Training data contains only clean dog images; evaluation data contains outliers (non-dogs).

### Methods Compared (student choice)
- Isolation Forest
- One-Class SVM
- Local Outlier Factor
- Autoencoder-based reconstruction error
- Embedding-distance methods (e.g. using pre-trained CNN features)

### Outputs
- Per-method outlier scores for each evaluation image
- Detection performance metrics (AUC, precision/recall)
- Visual display of images ranked by outlier score

---

## Lab 6 — Growing or Compressing Datasets (Active Learning)

**Notebook:** `growing_datasets/Lab - Growing Datasets.ipynb`  
**Dependencies:** `scikit-learn`, `numpy`, `matplotlib`

### Goal
Implement an active learning loop that selects the most informative unlabelled data points to request labels for, demonstrating that a small, strategically selected dataset can match or outperform a much larger randomly selected dataset.

### Key Concepts Implemented
- Uncertainty sampling (least-confidence, margin, entropy)
- Query-by-committee
- Comparison against random baseline

### Outputs
- Learning curves: model accuracy vs. number of labelled examples
- Plots comparing active learning strategies to random sampling

---

## Lab 7 — Interpretability in Data-Centric ML

**Notebook:** `interpretable_features/Lab - Interpretable Features.ipynb`  
**Solution:** `interpretable_features/Solution - Interpretable Features.ipynb`  
**Data:** `X_train_ft.csv`, `X_test_ft.csv`, `train_data.csv`, `test_data.csv`, `feature_descriptions.csv`, `data_description.txt`  
**Dependencies:** `shap`, `scikit-learn`, `pandas`, `matplotlib`

### Goal
Use interpretability tools (SHAP) to identify problematic features in a dataset — features that are inconsistently defined, leak information, or are spuriously correlated — thereby improving data quality.

### Outputs
- SHAP value plots (beeswarm, bar, force plots) for individual and global feature importance
- Identification of suspect features (e.g. features with implausibly high importance)
- Example explanation visual (`example_explanation.png`)

---

## Lab 8 — Encoding Human Priors: Prompt Engineering

**Notebook:** `prompt_engineering/Lab_Prompt_Engineering.ipynb`  
**Solution:** `prompt_engineering/Solution_Lab_Prompt_Engineering.ipynb`  
**Data:** `movie_club.txt`, `running_club.txt`, `math_club.txt` (sample past emails)  
**Dependencies:** `powerml-app==0.0.41`, Google Colab (`google.colab.auth`)  
**Also available on:** [Google Colab](https://colab.research.google.com/drive/1cipH-u6Jz0EH-6Cd9MPYgY4K0sJZwRJq)

### Goal
Learn to engineer data (prompts/context) fed to LLMs to produce better outputs, demonstrating that small amounts of targeted data can dramatically change LLM behaviour.

### Functions

| Function | Signature | Description |
|---|---|---|
| `authenticate_powerml` | `authenticate_powerml()` | Authenticates with Google via `google.colab.auth`; exchanges GCloud token for PowerML API token |
| `ContextTemplate` | `ContextTemplate(context: str, args: list[str])` | Creates a reusable prompt template with `{{variable}}` placeholders |
| `LLM` | `LLM(config: dict)` | Instantiates an LLM backed by the PowerML API |
| `llm.fit` | `llm.fit(template: ContextTemplate)` | Binds the LLM to a context template |
| `llm.predict` | `llm.predict(**kwargs) -> str` | Fills template variables and returns LLM-generated text |

### Outputs
- Generated club-announcement emails (one per club per exercise)
- Printed separator (`--------------`) between outputs
- Qualitative comparison of zero-shot vs. few-shot vs. styled outputs

### Exercises
1. **Reusable context:** Modify the context template so the email subject can be specified as a variable alongside `club_name`
2. **Impact of data:** Load past emails from `.txt` files and add them as few-shot examples in the context; compare outputs
3. **Style transfer:** Engineer the prompt to make the LLM write in a funnier style than the examples

---

## Lab 9 — Data Privacy and Security (Membership Inference)

**Notebook:** `membership_inference/Lab - Membership Inference.ipynb`  
**Solution:** `membership_inference/Solution - Membership Inference.ipynb`  
**Files:** `target_model.pt` (pre-trained PyTorch model), `target_model.py` (model definition)  
**Dependencies:** `torch`, `torchvision`, `scikit-learn`, `numpy`

### Goal
Implement a membership inference attack: given black-box prediction access to a trained ML model, determine whether a given data point was part of the model's training set.

### Key Concepts
- Shadow model training
- Confidence score thresholding
- Comparing training vs. non-training loss/confidence distributions

### Outputs
- Membership inference attack accuracy and/or AUC
- Confidence score distributions for members vs. non-members
- Comparison of attack strategies

---

## Cross-Cutting Summary

| Lab | Core Concept | Model Used | Key Output |
|---|---|---|---|
| 1 | Data cleaning vs. model tuning | SVM (SGD) | Accuracy improvement via data cleaning |
| 2 | Confident learning / label errors | XGBoost | ~35% error reduction after removing mislabelled data |
| 3 | Multi-annotator consensus | K-NN | CROWDLAB > majority-vote accuracy |
| 4 | Data-centric evaluation | MLP (fixed) | >80% balanced accuracy via data improvement only |
| 5 | Outlier / anomaly detection | Various | Outlier detection AUC comparison |
| 6 | Active learning | Various | Learning curves: accuracy vs. labels acquired |
| 7 | Interpretable features (SHAP) | Random Forest | Identification of suspect features |
| 8 | Prompt engineering | LLM (PowerML) | Better-fitting generated emails from few-shot context |
| 9 | Membership inference attack | Shadow models | Attack AUC on pre-trained target model |

---

*Generated from: https://github.com/dcai-course/dcai-lab/tree/master (2024 master branch)*
