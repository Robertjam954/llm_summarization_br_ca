# H2O Feature Importance Analysis Report
==================================================

## Summary
- Targets analyzed: 4
- Models used: Random Forest, XGBoost, Gradient Boosting

## Human Performance

### Random Forest

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 72.1988 |
| 2 | Workup Recommendation_human_correct | 19.5159 |
| 3 | Chronology Preserved_perf_diff | 16.8385 |
| 4 | total_agreements | 9.6383 |
| 5 | Chronology Preserved_human_correct | 7.6079 |
| 6 | ai_total_correct | 4.8914 |
| 7 | Unnamed: 0 | 4.5822 |
| 8 | Accurate Clip Placement_human_correct | 4.3929 |
| 9 | Invasive Component Size (Pathology)_perf_diff | 2.6385 |
| 10 | Accurate Clip Placement_perf_diff | 2.1783 |

### XGBoost

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 9.8653 |
| 2 | Lesion Size_human_correct | 0.4114 |
| 3 | Unnamed: 0 | 0.1439 |
| 4 | Lymph Node_human_correct | 0.0002 |
| 5 | Lesion Location_human_correct | 0.0002 |

### Gradient Boosting

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 9.8383 |
| 2 | Accurate Clip Placement_agreement | 0.2349 |
| 3 | Unnamed: 0 | 0.1917 |
| 4 | Lesion Size_agreement | 0.1703 |
| 5 | ai_total_correct | 0.0780 |
| 6 | Workup Recommendation_human_correct | 0.0603 |
| 7 | total_agreements | 0.0544 |
| 8 | Workup Recommendation_agreement | 0.0422 |
| 9 | Receptor Status_agreement | 0.0324 |
| 10 | Invasive Component Size (Pathology)_human_correct | 0.0145 |

## Ai Performance

### Random Forest

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | ai_total_correct | 39.9079 |
| 2 | human_total_correct | 6.6463 |
| 3 | total_agreements | 3.9784 |
| 4 | Lesion Location_agreement | 2.7285 |
| 5 | Lesion Size_perf_diff | 1.8316 |
| 6 | Unnamed: 0 | 1.1298 |
| 7 | Lesion Size_ai_correct | 0.9332 |
| 8 | Lesion Size_agreement | 0.6099 |
| 9 | Lesion Location_perf_diff | 0.5291 |
| 10 | Receptor Status_ai_correct | 0.5123 |

### XGBoost

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | ai_total_correct | 3.0202 |

### Gradient Boosting

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | ai_total_correct | 4.2003 |
| 2 | total_agreements | 0.0627 |
| 3 | Lesion Size_agreement | 0.0547 |
| 4 | Unnamed: 0 | 0.0325 |
| 5 | Workup Recommendation_agreement | 0.0106 |
| 6 | Invasive Component Size (Pathology)_agreement | 0.0085 |
| 7 | human_total_correct | 0.0060 |
| 8 | Receptor Status_ai_correct | 0.0046 |
| 9 | Lymph Node_agreement | 0.0036 |
| 10 | Workup Recommendation_perf_diff | 0.0027 |

## Human Advantage

### Random Forest

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 27.8304 |
| 2 | total_agreements | 21.0234 |
| 3 | Chronology Preserved_perf_diff | 8.2031 |
| 4 | Chronology Preserved_human_correct | 7.5961 |
| 5 | Lesion Size_perf_diff | 5.1093 |
| 6 | Lesion Location_ai_correct | 4.0900 |
| 7 | Accurate Clip Placement_perf_diff | 3.9153 |
| 8 | Lesion Size_ai_correct | 3.8110 |
| 9 | Invasive Component Size (Pathology)_perf_diff | 3.8013 |
| 10 | Accurate Clip Placement_human_correct | 3.6214 |

### XGBoost

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 2.1825 |
| 2 | Lesion Location_ai_correct | 0.5601 |
| 3 | Workup Recommendation_human_correct | 0.2344 |
| 4 | ai_total_correct | 0.1592 |
| 5 | Lesion Size_ai_correct | 0.1519 |
| 6 | total_agreements | 0.1216 |
| 7 | Invasive Component Size (Pathology)_perf_diff | 0.1080 |
| 8 | Receptor Status_ai_correct | 0.0792 |
| 9 | Additional Enhancement (MRI)_agreement | 0.0636 |
| 10 | Lymph Node_perf_diff | 0.0623 |

### Gradient Boosting

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | human_total_correct | 4.4049 |
| 2 | ai_total_correct | 1.7262 |
| 3 | total_agreements | 0.4602 |
| 4 | Workup Recommendation_agreement | 0.4452 |
| 5 | Unnamed: 0 | 0.3369 |
| 6 | Workup Recommendation_human_correct | 0.2882 |
| 7 | Accurate Clip Placement_agreement | 0.1707 |
| 8 | Receptor Status_perf_diff | 0.0909 |
| 9 | Lymph Node_human_correct | 0.0798 |
| 10 | Lymph Node_perf_diff | 0.0713 |

## Agreement Rate

### Random Forest

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | total_agreements | 44.0496 |
| 2 | human_total_correct | 13.9278 |
| 3 | Accurate Clip Placement_perf_diff | 7.2952 |
| 4 | Accurate Clip Placement_agreement | 6.3689 |
| 5 | Accurate Clip Placement_human_correct | 4.1937 |
| 6 | Chronology Preserved_perf_diff | 3.7691 |
| 7 | Workup Recommendation_agreement | 2.0082 |
| 8 | Chronology Preserved_agreement | 1.7485 |
| 9 | Lesion Size_perf_diff | 1.6813 |
| 10 | Invasive Component Size (Pathology)_agreement | 1.6080 |

### XGBoost

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | total_agreements | 4.5562 |
| 2 | Lesion Size_human_correct | 1.0257 |
| 3 | human_total_correct | 0.0268 |

### Gradient Boosting

| Rank | Feature | Relative Importance |
|------|---------|--------------------|
| 1 | total_agreements | 3.9324 |
| 2 | Accurate Clip Placement_agreement | 0.8912 |
| 3 | human_total_correct | 0.7932 |
| 4 | Lesion Size_agreement | 0.4131 |
| 5 | Unnamed: 0 | 0.1094 |
| 6 | ai_total_correct | 0.0931 |
| 7 | Workup Recommendation_human_correct | 0.0319 |
| 8 | Receptor Status_agreement | 0.0297 |
| 9 | Workup Recommendation_agreement | 0.0238 |
| 10 | Lymph Node_human_correct | 0.0201 |
