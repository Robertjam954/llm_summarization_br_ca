# HCAT Safety Metrics — Selection List
**Memorial Sloan Kettering | Goel Lab**  
**Date:** March 2026

---

## Overview

This document lists all relevant HCAT (HIPAA Compliance Assessment Tool) metrics for assessing the safety and effectiveness of our deidentification pipeline (NB01). Please review and select which metrics to implement in NB11.

---

## Category 1: Core Deidentification Performance Metrics

### 1.1 PHI Detection & Redaction Accuracy

| Metric | Definition | Formula | Priority |
|--------|------------|---------|----------|
| **PHI Recall (Sensitivity)** | % of true PHI instances successfully detected and redacted | `TP / (TP + FN)` | **HIGH** |
| **PHI Precision** | % of redactions that were actually PHI (not false positives) | `TP / (TP + FP)` | **HIGH** |
| **PHI F1-Score** | Harmonic mean of precision and recall | `2 × (Precision × Recall) / (Precision + Recall)` | **HIGH** |
| **False Negative Rate (FNR)** | % of PHI instances missed by redaction | `FN / (TP + FN)` = `1 - Recall` | **HIGH** |
| **False Positive Rate (FPR)** | % of non-PHI text incorrectly redacted | `FP / (FP + TN)` | MEDIUM |
| **Over-redaction Rate** | % of total text unnecessarily redacted | `FP_chars / total_chars` | MEDIUM |

**Notes:**
- TP = True Positives (PHI correctly redacted)
- FP = False Positives (non-PHI incorrectly redacted)
- FN = False Negatives (PHI missed)
- TN = True Negatives (non-PHI correctly preserved)

---

### 1.2 PHI Type-Specific Performance

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Per-Type Recall** | Recall computed separately for each PHI type (EMAIL, SSN, MRN, etc.) | `TP_type / (TP_type + FN_type)` for each type | **HIGH** |
| **Per-Type Precision** | Precision computed separately for each PHI type | `TP_type / (TP_type + FP_type)` for each type | **HIGH** |
| **Critical PHI Miss Rate** | % of high-severity PHI (SSN, MRN, NAME) missed | `FN_critical / total_critical_phi` | **HIGH** |
| **Low-Risk PHI Miss Rate** | % of low-severity PHI (ZIP, DATE) missed | `FN_low_risk / total_low_risk_phi` | LOW |

---

## Category 2: Residual PHI Risk Assessment

### 2.1 Weighted Risk Scores

| Metric | Definition | Formula | Priority |
|--------|------------|---------|----------|
| **Residual PHI Risk Score** | Weighted sum of missed PHI by severity | `Σ(FN_count_type × severity_weight_type)` | **HIGH** |
| **Document-Level Risk Score** | Max risk score across all documents | `max(residual_risk_per_doc)` | **HIGH** |
| **Average Risk Per Document** | Mean residual risk across all documents | `mean(residual_risk_per_doc)` | MEDIUM |
| **High-Risk Document Count** | # of documents with risk score > threshold | `count(risk_score > threshold)` | MEDIUM |

**Severity Weights:**
```
SSN: 10.0          (Highest risk)
MRN: 8.0
NAME: 7.0
DOB: 6.0
ADDRESS: 5.0
PHONE: 4.0
EMAIL: 3.0
DATE: 2.0
ZIP: 1.5
CONTEXT_NUMERIC: 1.0  (Lowest risk)
```

---

### 2.2 Re-identification Risk Metrics

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **k-Anonymity Score** | Minimum group size for any quasi-identifier combination | `min(group_size)` across all QI combos | MEDIUM |
| **l-Diversity Score** | Minimum diversity of sensitive attributes within each group | `min(unique_sensitive_values)` per group | LOW |
| **Uniqueness Risk** | % of records with unique quasi-identifier combinations | `unique_QI_combos / total_records` | MEDIUM |
| **Prosecutor Risk** | Probability of re-identification given background knowledge | Computed via probabilistic model | LOW |
| **Journalist Risk** | Probability of re-identification via linkage attack | Computed via linkage simulation | LOW |

**Notes:**
- Quasi-identifiers (QI) = combinations of non-direct identifiers that could enable re-identification (e.g., age + ZIP + diagnosis date)
- Requires manual annotation of QI fields in validation dataset

---

## Category 3: Operational & Quality Metrics

### 3.1 Processing Efficiency

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Redaction Throughput** | PDFs processed per hour | `total_pdfs / total_hours` | LOW |
| **Average Redactions Per Page** | Mean # of redactions per PDF page | `total_redactions / total_pages` | LOW |
| **OCR Confidence Score** | Mean OCR confidence across all tokens | `mean(ocr_conf)` | MEDIUM |
| **Error Rate** | % of PDFs that failed processing | `failed_pdfs / total_pdfs` | MEDIUM |

---

### 3.2 Redaction Quality

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Bounding Box Coverage** | % of PHI text fully covered by black boxes | `fully_covered_phi / total_phi` | **HIGH** |
| **Partial Redaction Rate** | % of PHI with incomplete coverage (edges visible) | `partial_redactions / total_redactions` | **HIGH** |
| **Redaction Uniformity** | Consistency of padding across redactions | `std(padding_px)` | LOW |
| **Visual Inspection Pass Rate** | % of redacted PDFs passing manual visual QC | `passed_visual_qc / total_sampled` | MEDIUM |

---

## Category 4: Contextual Redaction Performance

### 4.1 Context-Based Detection

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Context Label Detection Rate** | % of PHI labels (e.g., "MRN:", "Patient:") detected | `detected_labels / total_labels` | MEDIUM |
| **Context Window Accuracy** | % of tokens after labels correctly classified as PHI/non-PHI | `correct_context_classifications / total_context_tokens` | MEDIUM |
| **Context False Positive Rate** | % of non-PHI tokens after labels incorrectly redacted | `FP_context / total_context_tokens` | MEDIUM |
| **Optimal Window Size** | Window size (# tokens after label) that maximizes F1 | Tuned via grid search | LOW |

---

## Category 5: Compliance & Audit Metrics

### 5.1 HIPAA Safe Harbor Compliance

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Safe Harbor Element Coverage** | % of 18 HIPAA Safe Harbor identifiers addressed | `covered_elements / 18` | **HIGH** |
| **Geographic Subdivision Compliance** | All geographic units smaller than state redacted? | Boolean (yes/no) | **HIGH** |
| **Date Precision Compliance** | All dates more precise than year redacted? | Boolean (yes/no) | **HIGH** |
| **Age Compliance** | All ages >89 redacted or aggregated? | Boolean (yes/no) | MEDIUM |

**18 HIPAA Safe Harbor Identifiers:**
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year)
4. Telephone numbers
5. Fax numbers
6. Email addresses
7. Social Security numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers
13. Device identifiers/serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photos
18. Any other unique identifying number/code

---

### 5.2 Audit Trail Metrics

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Mapping Completeness** | % of deidentified files with case_id mapping | `mapped_files / total_files` | **HIGH** |
| **Log Completeness** | % of redactions logged with metadata | `logged_redactions / total_redactions` | MEDIUM |
| **Traceability Score** | % of observations traceable back to source | `traceable_obs / total_obs` | **HIGH** |
| **Audit Log Integrity** | No missing/corrupted log entries | Boolean (yes/no) | MEDIUM |

---

## Category 6: Comparative & Benchmark Metrics

### 6.1 Method Comparison

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **OCR vs Manual Annotation Agreement** | Cohen's kappa between OCR redaction and manual PHI annotation | `kappa(ocr, manual)` | **HIGH** |
| **Regex vs ML Redaction Comparison** | F1 score difference between regex and ML-based redaction | `F1_ml - F1_regex` | LOW |
| **Inter-Annotator Agreement** | Agreement between multiple manual PHI annotators | `Fleiss' kappa` or `Krippendorff's alpha` | MEDIUM |

---

### 6.2 Benchmark Against Standards

| Metric | Definition | Calculation | Priority |
|--------|------------|-------------|----------|
| **Industry Benchmark Comparison** | Our recall vs published deidentification benchmarks | `our_recall - benchmark_recall` | MEDIUM |
| **MSK Internal Standard Compliance** | Meets MSK IRB deidentification requirements? | Boolean (yes/no) | **HIGH** |
| **FDA Guidance Compliance** | Meets FDA guidance for deidentified clinical data? | Boolean (yes/no) | MEDIUM |

---

## Recommended Metrics for NB11 Implementation

Based on project priorities (safety, traceability, HIPAA compliance), I recommend implementing the following **core set**:

### Tier 1: Essential (Must Implement)
1. ✅ **PHI Recall (Sensitivity)** — Primary safety metric
2. ✅ **PHI Precision** — Avoid over-redaction
3. ✅ **PHI F1-Score** — Overall performance
4. ✅ **False Negative Rate** — Critical for safety
5. ✅ **Per-Type Recall** — Identify weak PHI types
6. ✅ **Residual PHI Risk Score** — Weighted risk assessment
7. ✅ **Critical PHI Miss Rate** — Focus on high-severity PHI
8. ✅ **Bounding Box Coverage** — Ensure complete redaction
9. ✅ **Safe Harbor Element Coverage** — HIPAA compliance
10. ✅ **Mapping Completeness** — Traceability
11. ✅ **OCR vs Manual Annotation Agreement** — Validation quality

### Tier 2: Important (Should Implement)
12. ⚠️ **Over-redaction Rate** — Balance utility vs safety
13. ⚠️ **Document-Level Risk Score** — Identify high-risk cases
14. ⚠️ **Partial Redaction Rate** — Quality control
15. ⚠️ **Context Label Detection Rate** — Contextual redaction performance
16. ⚠️ **Traceability Score** — Audit capability

### Tier 3: Optional (Nice to Have)
17. 🔵 **k-Anonymity Score** — Advanced privacy metric
18. 🔵 **Inter-Annotator Agreement** — If multiple annotators
19. 🔵 **Redaction Throughput** — Operational efficiency
20. 🔵 **Industry Benchmark Comparison** — External validation

---

## Selection Instructions

**Please review the metrics above and indicate:**

1. **Which Tier 1 metrics to implement?** (Recommend all 11)
2. **Which Tier 2 metrics to add?** (Select 2-5)
3. **Any Tier 3 metrics of interest?** (Optional)
4. **Any custom metrics not listed?** (Please specify)

**Response format:**
```
Tier 1: [list metric numbers to implement, e.g., 1-11 or specific subset]
Tier 2: [list metric numbers, e.g., 12, 13, 15]
Tier 3: [list metric numbers or "none"]
Custom: [describe any additional metrics]
```

Once you confirm, I will implement the selected metrics in NB11.
