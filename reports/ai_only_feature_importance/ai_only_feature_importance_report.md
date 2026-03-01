# H2O AI-Only Feature Importance Analysis Report
============================================================

## Summary
- Analysis focuses on AI-only features (human columns excluded)
- Targets analyzed: 4
- Models used: Random Forest, XGBoost, Gradient Boosting

## AI Feature Categories
- Agreement: 13 features
- Performance Diff: 10 features
- Element AI: 7 features
- AI-specific: 2 features
- Other: 1 features

## Ai Performance

### Random Forest

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 20.6399 |
| 2 | Lesion Size_perf_diff | Performance Diff | 5.5930 |
| 3 | Lesion Location_agreement | Agreement | 5.2810 |
| 4 | Histologic Diagnosis_agreement | Agreement | 1.9687 |
| 5 | Lesion Size_ai_correct | AI-specific | 1.9647 |
| 6 | Lesion Size_agreement | Agreement | 1.4026 |
| 7 | Histologic Diagnosis_perf_diff | Performance Diff | 1.3225 |
| 8 | Workup Recommendation_agreement | Agreement | 1.3053 |
| 9 | Receptor Status_agreement | Agreement | 1.2292 |
| 10 | Lesion Location_perf_diff | Performance Diff | 1.1706 |

### XGBoost

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 1.2194 |
| 2 | Lesion Location_agreement | Agreement | 0.2563 |
| 3 | Lesion Size_ai_correct | AI-specific | 0.1694 |
| 4 | Histologic Diagnosis_ai_correct | Element AI | 0.1637 |
| 5 | Workup Recommendation_perf_diff | Performance Diff | 0.1628 |
| 6 | Receptor Status_agreement | Agreement | 0.0733 |
| 7 | Histologic Diagnosis_agreement | Agreement | 0.0384 |
| 8 | Lymph Node_agreement | Agreement | 0.0366 |
| 9 | Chronology Preserved_ai_correct | Element AI | 0.0335 |
| 10 | Receptor Status_ai_correct | Element AI | 0.0258 |

### Gradient Boosting

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 1.9893 |
| 2 | Lesion Size_agreement | Agreement | 0.6435 |
| 3 | Workup Recommendation_agreement | Agreement | 0.2991 |
| 4 | Receptor Status_ai_correct | Element AI | 0.1488 |
| 5 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 0.1394 |
| 6 | Receptor Status_agreement | Agreement | 0.1282 |
| 7 | Workup Recommendation_perf_diff | Performance Diff | 0.0999 |
| 8 | Lymph Node_agreement | Agreement | 0.0422 |
| 9 | Invasive Component Size (Pathology)_agreement | Agreement | 0.0268 |
| 10 | Accurate Clip Placement_agreement | Agreement | 0.0227 |

## Ai Total Correct

### Random Forest

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 3941.3113 |
| 2 | Lesion Location_agreement | Agreement | 1037.3662 |
| 3 | Lesion Size_perf_diff | Performance Diff | 950.0955 |
| 4 | Lesion Size_ai_correct | AI-specific | 539.6971 |
| 5 | Histologic Diagnosis_agreement | Agreement | 349.4773 |
| 6 | Lesion Size_agreement | Agreement | 272.9792 |
| 7 | Histologic Diagnosis_perf_diff | Performance Diff | 271.2551 |
| 8 | Receptor Status_agreement | Agreement | 264.7563 |
| 9 | Workup Recommendation_agreement | Agreement | 260.4468 |
| 10 | Chronology Preserved_agreement | Agreement | 226.7274 |

### XGBoost

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 213.0183 |
| 2 | Lesion Size_ai_correct | AI-specific | 23.4363 |
| 3 | Calcifications / Asymmetry_agreement | Agreement | 8.2916 |
| 4 | Workup Recommendation_perf_diff | Performance Diff | 7.0962 |
| 5 | Histologic Diagnosis_ai_correct | Element AI | 7.0022 |
| 6 | Biopsy Method_agreement | Agreement | 6.0595 |
| 7 | Receptor Status_agreement | Agreement | 5.8569 |
| 8 | Lymph Node_agreement | Agreement | 5.3711 |
| 9 | Lymph Node_ai_correct | Element AI | 5.2421 |
| 10 | Lesion Location_agreement | Agreement | 5.1924 |

### Gradient Boosting

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Unnamed: 0 | Other | 389.7212 |
| 2 | Lesion Size_agreement | Agreement | 125.9996 |
| 3 | Workup Recommendation_agreement | Agreement | 58.9862 |
| 4 | Receptor Status_ai_correct | Element AI | 31.9549 |
| 5 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 27.1835 |
| 6 | Receptor Status_agreement | Agreement | 25.2604 |
| 7 | Workup Recommendation_perf_diff | Performance Diff | 19.9068 |
| 8 | Lymph Node_agreement | Agreement | 8.4238 |
| 9 | Invasive Component Size (Pathology)_agreement | Agreement | 4.6901 |
| 10 | Accurate Clip Placement_agreement | Agreement | 4.4472 |

## Human Advantage

### Random Forest

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Chronology Preserved_perf_diff | Performance Diff | 18.6911 |
| 2 | Lesion Size_perf_diff | Performance Diff | 15.9365 |
| 3 | Accurate Clip Placement_perf_diff | Performance Diff | 13.7953 |
| 4 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 9.0911 |
| 5 | Chronology Preserved_agreement | Agreement | 5.9348 |
| 6 | Lesion Location_perf_diff | Performance Diff | 5.1812 |
| 7 | Lesion Size_ai_correct | AI-specific | 4.3088 |
| 8 | Workup Recommendation_perf_diff | Performance Diff | 4.0695 |
| 9 | Lymph Node_perf_diff | Performance Diff | 4.0443 |
| 10 | Workup Recommendation_agreement | Agreement | 4.0342 |

### XGBoost

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Lesion Location_ai_correct | AI-specific | 0.6848 |
| 2 | Lesion Size_perf_diff | Performance Diff | 0.6512 |
| 3 | Chronology Preserved_perf_diff | Performance Diff | 0.5579 |
| 4 | Accurate Clip Placement_perf_diff | Performance Diff | 0.4038 |
| 5 | Workup Recommendation_perf_diff | Performance Diff | 0.3525 |
| 6 | Lymph Node_perf_diff | Performance Diff | 0.2662 |
| 7 | Lesion Size_ai_correct | AI-specific | 0.2130 |
| 8 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 0.2081 |
| 9 | Additional Enhancement (MRI)_agreement | Agreement | 0.1608 |
| 10 | Receptor Status_ai_correct | Element AI | 0.1096 |

### Gradient Boosting

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 1.5075 |
| 2 | Unnamed: 0 | Other | 1.3700 |
| 3 | Workup Recommendation_perf_diff | Performance Diff | 1.2470 |
| 4 | Accurate Clip Placement_agreement | Agreement | 0.7784 |
| 5 | Receptor Status_perf_diff | Performance Diff | 0.6769 |
| 6 | Workup Recommendation_agreement | Agreement | 0.5837 |
| 7 | Lymph Node_perf_diff | Performance Diff | 0.5240 |
| 8 | Lymph Node_agreement | Agreement | 0.3045 |
| 9 | Invasive Component Size (Pathology)_agreement | Agreement | 0.0486 |
| 10 | Receptor Status_agreement | Agreement | 0.0468 |

## Agreement Rate

### Random Forest

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Accurate Clip Placement_agreement | Agreement | 18.8748 |
| 2 | Accurate Clip Placement_perf_diff | Performance Diff | 14.0992 |
| 3 | Calcifications / Asymmetry_agreement | Agreement | 7.7319 |
| 4 | Lesion Size_perf_diff | Performance Diff | 7.0572 |
| 5 | Chronology Preserved_perf_diff | Performance Diff | 6.5296 |
| 6 | Workup Recommendation_agreement | Agreement | 6.4983 |
| 7 | Invasive Component Size (Pathology)_agreement | Agreement | 4.1773 |
| 8 | Chronology Preserved_agreement | Agreement | 3.8721 |
| 9 | Lesion Size_agreement | Agreement | 3.0364 |
| 10 | Unnamed: 0 | Other | 2.8930 |

### XGBoost

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Accurate Clip Placement_agreement | Agreement | 1.3604 |
| 2 | Chronology Preserved_perf_diff | Performance Diff | 0.8435 |
| 3 | Lesion Size_perf_diff | Performance Diff | 0.7377 |
| 4 | Calcifications / Asymmetry_agreement | Agreement | 0.5466 |
| 5 | Workup Recommendation_agreement | Agreement | 0.3859 |
| 6 | Invasive Component Size (Pathology)_agreement | Agreement | 0.1934 |
| 7 | Lymph Node_agreement | Agreement | 0.1167 |
| 8 | Lesion Size_agreement | Agreement | 0.1141 |
| 9 | Unnamed: 0 | Other | 0.1054 |
| 10 | Histologic Diagnosis_agreement | Agreement | 0.0956 |

### Gradient Boosting

| Rank | Feature | Category | Relative Importance |
|------|---------|----------|--------------------|
| 1 | Accurate Clip Placement_agreement | Agreement | 2.3394 |
| 2 | Lesion Size_agreement | Agreement | 1.2766 |
| 3 | Workup Recommendation_agreement | Agreement | 0.9476 |
| 4 | Invasive Component Size (Pathology)_agreement | Agreement | 0.5510 |
| 5 | Unnamed: 0 | Other | 0.5128 |
| 6 | Lymph Node_agreement | Agreement | 0.2099 |
| 7 | Receptor Status_agreement | Agreement | 0.2033 |
| 8 | Invasive Component Size (Pathology)_perf_diff | Performance Diff | 0.0504 |
| 9 | Lymph Node_perf_diff | Performance Diff | 0.0466 |
| 10 | Receptor Status_perf_diff | Performance Diff | 0.0261 |
