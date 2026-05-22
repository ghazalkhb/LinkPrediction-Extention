# Fair Overlap Comparison Report

## 1. Objective
This experiment compares overlap and non-overlap windowing fairly, including the reviewer-requested comparison between 50 ms non-overlap and 100 ms with 50% overlap.

## 2. Compared Settings
- 50 ms non-overlap
- 100 ms non-overlap
- 100 ms with 50% overlap
- 200 ms non-overlap
- 500 ms non-overlap

## 3. Shared Evaluation Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Forecasting direction: G_t -> E_{t+1}
- Train time end: 7000
- Test time start: 7000
- Test negative ratio: 1:10
- Precision@K uses K=100

## 4. Results
### 50ms_non_overlap
- Train windows: 140
- Test windows: 60
- Average edges per window: 4197.9
- Duplicated edge percentage from overlap: 0.00%
- AUC: 0.9737 +- 0.0127
- F1: 0.2659 +- 0.0035
- PR-AUC: 0.9460 +- 0.0200
- Precision@K: 0.9978 +- 0.0049
- Runtime train/eval (sec): 225.00 / 8.11

### 100ms_non_overlap
- Train windows: 70
- Test windows: 30
- Average edges per window: 8395.8
- Duplicated edge percentage from overlap: 0.00%
- AUC: 0.9717 +- 0.0101
- F1: 0.2676 +- 0.0024
- PR-AUC: 0.9381 +- 0.0171
- Precision@K: 0.9952 +- 0.0062
- Runtime train/eval (sec): 166.99 / 5.82

### 100ms_overlap_50pct
- Train windows: 139
- Test windows: 60
- Average edges per window: 8398.0
- Duplicated edge percentage from overlap: 49.78%
- AUC: 0.9745 +- 0.0103
- F1: 0.2603 +- 0.0027
- PR-AUC: 0.9384 +- 0.0148
- Precision@K: 0.9953 +- 0.0067
- Runtime train/eval (sec): 188.45 / 12.12

### 200ms_non_overlap
- Train windows: 35
- Test windows: 15
- Average edges per window: 16791.7
- Duplicated edge percentage from overlap: 0.00%
- AUC: 0.9547 +- 0.0123
- F1: 0.2628 +- 0.0035
- PR-AUC: 0.8746 +- 0.0225
- Precision@K: 0.9914 +- 0.0091
- Runtime train/eval (sec): 91.38 / 5.62

### 500ms_non_overlap
- Train windows: 14
- Test windows: 6
- Average edges per window: 41979.2
- Duplicated edge percentage from overlap: 0.00%
- AUC: 0.9141 +- 0.0128
- F1: 0.2650 +- 0.0034
- PR-AUC: 0.7965 +- 0.0174
- Precision@K: 1.0000 +- 0.0000
- Runtime train/eval (sec): 40.15 / 4.46

## 5. Interpretation
This comparison distinguishes improvements from true temporal resolution effects versus overlap-induced duplication and larger effective sample reuse.