# Full Report: Revision Experiments

Generated at: 2026-05-18 16:12:08

This file consolidates all per-experiment markdown reports generated in revision/results.

---

# Rolling Forecasting Evaluation Report (G_t -> G_{t+1})

## 1. Objective
This experiment evaluates temporal link forecasting where each prediction uses graph G_t as input and scores links that appear in G_{t+1}.

## 2. Protocol
- Model: GAT
- Dataset: Data\Alibaba 2022\CallGraph_0.csv
- Window size: 100
- Fixed training windows: G_0 ... G_69
- Validation windows (inside training horizon): G_60 ... G_68
- Test windows: G_70 ... G_1799
- Test target at each step: edges from G_{t+1}
- Negatives at each step: sampled node pairs not present in G_{t+1}

## 3. Run Summary
- Training pairs used: 60
- Validation pairs used: 9
- Test pairs evaluated: 1729
- Training time (sec): 448.25
- Evaluation time (sec): 196.61

## 4. Primary Forecasting Results (Average Over Rolling Pairs)
- AUC: mean=0.9707, std=0.0121, min=0.8361, max=0.9887
- Precision: mean=0.6479, std=0.0041, min=0.6095, max=0.6581
- Recall: mean=0.9738, std=0.0130, min=0.8186, max=0.9917
- F1: mean=0.7781, std=0.0067, min=0.6987, max=0.7904
- Accuracy: mean=0.7223, std=0.0072, min=0.6471, max=0.7377
- MRR: mean=0.9983, std=0.0294, min=0.5000, max=1.0000

## 5. Saved Artifacts
- Per-pair rolling metrics: E:\LinkPrediction-Extention-main\revision\results\rolling_per_pair_metrics.csv
- JSON summary: E:\LinkPrediction-Extention-main\revision\results\rolling_summary.json

## 6. Interpretation
These values are the primary forecasting result because they strictly follow the temporal direction G_t -> G_{t+1} and avoid evaluating links inside the same test graph window.

---

# Training-Testing Gap Experiment Report

## Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Window size: 100ms
- Training: G_0..G_59 -> E_1..E_60 (60 pairs)
- Validation: G_60..G_68 -> E_61..E_69 (9 pairs)
- Gaps evaluated: [0, 5, 10, 20, 50, 100, 200, 500]
- Eval window per gap: 30 consecutive pairs
- Seeds: 3
- Negative sampling: degree-aware (alpha=0.1)

## Results

| Gap (windows) | Gap (ms) | AUC | F1 | PR-AUC | Precision | Recall |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.9749±0.0086 | 0.7805±0.0050 | 0.9835±0.0053 | 0.6491 | 0.9786 |
| 5 | 500 | 0.9733±0.0096 | 0.7811±0.0051 | 0.9826±0.0059 | 0.6509 | 0.9766 |
| 10 | 1000 | 0.9722±0.0102 | 0.7793±0.0061 | 0.9818±0.0064 | 0.6485 | 0.9762 |
| 20 | 2000 | 0.9620±0.0274 | 0.7731±0.0164 | 0.9757±0.0156 | 0.6451 | 0.9648 |
| 50 | 5000 | 0.9744±0.0085 | 0.7804±0.0056 | 0.9836±0.0052 | 0.6496 | 0.9773 |
| 100 | 10000 | 0.9714±0.0069 | 0.7780±0.0055 | 0.9815±0.0042 | 0.6472 | 0.9750 |
| 200 | 20000 | 0.9740±0.0065 | 0.7789±0.0040 | 0.9829±0.0040 | 0.6476 | 0.9772 |
| 500 | 50000 | 0.9667±0.0100 | 0.7741±0.0053 | 0.9785±0.0058 | 0.6437 | 0.9706 |

## Interpretation
As the gap increases, the model must predict further into the future using
a fixed training horizon. Degradation reflects temporal drift in the call graph.

---

# Recurring Link Activity Experiment Report

## Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Window size: 100ms
- Training: G_0..G_59 (60 pairs), Val: G_60..G_68 (9 pairs)
- Negative sampling: degree-aware (alpha=0.1)
- Seeds: 3
- Sampled 50 windows per density category (Q25=6943, Q75=7735 edges)

## Results

| Scenario | AUC | F1 | PR-AUC | Precision | Recall | Avg Input Edges | Avg Target Edges |
|---|---|---|---|---|---|---|---|
| low_density | 0.9729±0.0078 | 0.7800±0.0049 | 0.9828 | 0.6495 | 0.9761 | 6669 | 7177 |
| high_density | 0.9700±0.0174 | 0.7789±0.0081 | 0.9808 | 0.6489 | 0.9740 | 8195 | 7724 |

## Interpretation
This experiment tests whether the model can generalize across load conditions.
Low-density windows have fewer edges (bottom quartile) while high-density
windows have more (top quartile). Comparable AUC across conditions indicates
robustness to load fluctuations; a large gap indicates load-dependent fragility.

---

# Multi-Horizon Forecasting Report (G_t -> G_{t+k})

## 1. Objective
This experiment evaluates whether future-link prediction remains operationally useful when forecasting multiple steps ahead using G_t to predict E_{t+k}.

## 2. Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Window size: 100 ms
- Horizons: 1,5,10,30
- Training horizon anchor: G_0 ... G_69
- Validation windows: selected from later trainable pairs for each k
- Test start input: G_70
- Top-K for Precision@K and Recall@K: 100

## 3. Horizon Mapping
- k=1: 100 ms ahead
- k=5: 500 ms ahead
- k=10: 1000 ms ahead
- k=30: 3000 ms ahead

## 4. Primary Results By Horizon
### k=1 (100 ms ahead)
- Evaluated test pairs: 1729
- AUC: 0.9677 +- 0.0127
- Average Precision (PR-AUC): 0.9797 +- 0.0072
- Precision@K: 0.9995 +- 0.0024
- Recall@K: 0.0137 +- 0.0011
- F1 (secondary): 0.7758 +- 0.0069
- Inference time per prediction (ms): 107.48 +- 26.66

### k=5 (500 ms ahead)
- Evaluated test pairs: 1725
- AUC: 0.9666 +- 0.0123
- Average Precision (PR-AUC): 0.9792 +- 0.0071
- Precision@K: 0.9998 +- 0.0013
- Recall@K: 0.0137 +- 0.0011
- F1 (secondary): 0.7725 +- 0.0068
- Inference time per prediction (ms): 103.16 +- 20.12

### k=10 (1000 ms ahead)
- Evaluated test pairs: 1720
- AUC: 0.9644 +- 0.0162
- Average Precision (PR-AUC): 0.9778 +- 0.0093
- Precision@K: 0.9997 +- 0.0016
- Recall@K: 0.0137 +- 0.0011
- F1 (secondary): 0.7671 +- 0.0096
- Inference time per prediction (ms): 104.18 +- 19.07

### k=30 (3000 ms ahead)
- Evaluated test pairs: 1700
- AUC: 0.9614 +- 0.0123
- Average Precision (PR-AUC): 0.9761 +- 0.0071
- Precision@K: 0.9998 +- 0.0014
- Recall@K: 0.0137 +- 0.0011
- F1 (secondary): 0.7723 +- 0.0069
- Inference time per prediction (ms): 99.70 +- 19.15

## 5. Train/Validation Metadata
- k=1: train_pairs=55, val_pairs=14, best_epoch=31, training_time_sec=420.58
- k=5: train_pairs=52, val_pairs=13, best_epoch=32, training_time_sec=402.81
- k=10: train_pairs=48, val_pairs=12, best_epoch=45, training_time_sec=484.53
- k=30: train_pairs=32, val_pairs=8, best_epoch=29, training_time_sec=231.50

## 6. Saved Artifacts
- horizon_per_pair_metrics.csv
- horizon_summary.csv
- horizon_summary.json

---

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

---

# Imbalanced Negative Test Evaluation Report

## 1. Objective
This experiment re-evaluates forecasting under increasingly imbalanced test negatives to measure how balanced 1:1 testing may inflate F1, precision, and accuracy.

## 2. Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Window size: 100
- Training negatives: 1:1 (kept unchanged)
- Test forecasting direction: G_t -> E_{t+1}
- Test negative ratios: 1,5,10,50
- Test start input: G_70
- Best training epoch: 48
- Training time (sec): 310.36

## 3. Main Results By Test Imbalance
### 1 positive : 1 negatives
- Evaluated pairs: 1729
- AUC: 0.9712 +- 0.0114
- Precision: 0.6432 +- 0.0040
- Recall: 0.9743 +- 0.0121
- F1: 0.7749 +- 0.0064
- Accuracy: 0.7170 +- 0.0070
- Avg positive edges/window: 7364.3
- Avg negative edges/window: 7364.3

### 1 positive : 5 negatives
- Evaluated pairs: 1729
- AUC: 0.9711 +- 0.0115
- Precision: 0.2649 +- 0.0028
- Recall: 0.9743 +- 0.0121
- F1: 0.4166 +- 0.0044
- Accuracy: 0.5452 +- 0.0034
- Avg positive edges/window: 7364.3
- Avg negative edges/window: 36821.4

### 1 positive : 10 negatives
- Evaluated pairs: 1729
- AUC: 0.9711 +- 0.0115
- Precision: 0.1527 +- 0.0018
- Recall: 0.9743 +- 0.0121
- F1: 0.2640 +- 0.0031
- Accuracy: 0.5061 +- 0.0028
- Avg positive edges/window: 7364.3
- Avg negative edges/window: 73642.9

### 1 positive : 50 negatives
- Evaluated pairs: 300
- AUC: 0.9727 +- 0.0132
- Precision: 0.0349 +- 0.0005
- Recall: 0.9759 +- 0.0138
- F1: 0.0674 +- 0.0009
- Accuracy: 0.4708 +- 0.0021
- Avg positive edges/window: 7726.6
- Avg negative edges/window: 386329.7

## 4. All-Non-Edges Evaluation Feasibility
All-non-edges testing is not computationally feasible under current dataset scale or configured limits. Feasibility thresholds: max_nodes=3000, max_negatives=2000000.

## 5. Interpretation
As the test negative ratio increases, precision and F1 are expected to decrease while recall may remain less affected, directly exposing optimistic bias from balanced 1:1 test distributions.

---

# Longer Temporal Slice / Sensitivity Experiment

## 1. Objective
This experiment tests temporal-slice stability under the corrected rolling protocol to address reviewer concerns that short slices may distort conclusions.

## 2. Protocol
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Model: GAT
- Window size: 100
- Rolling direction: G_t -> E_(t+1)
- Evaluation negative ratio: 1:10
- Precision@K: K=100
- Sensitivity mode used: segments

## 3. Slice Results
### early_third (ok)
- Rows: 4588825
- Services: 13597
- Windows: 601
- Mean/median edges per window: 7635.32 / 7584.00
- Edge density: 0.00022878
- AUC: 0.9835 +- 0.0057
- F1: 0.2692 +- 0.0018
- PR-AUC: 0.9498 +- 0.0088
- Precision@K: 0.9966 +- 0.0056
- Runtime (train sec): 501.14

### middle_third (ok)
- Rows: 4370710
- Services: 13520
- Windows: 601
- Mean/median edges per window: 7272.40 / 7207.00
- Edge density: 0.00022834
- AUC: 0.9833 +- 0.0066
- F1: 0.2772 +- 0.0020
- PR-AUC: 0.9512 +- 0.0094
- Precision@K: 0.9975 +- 0.0050
- Runtime (train sec): 462.66

### late_third (ok)
- Rows: 4372821
- Services: 13185
- Windows: 601
- Mean/median edges per window: 7275.91 / 7238.00
- Edge density: 0.00023262
- AUC: 0.9827 +- 0.0058
- F1: 0.2732 +- 0.0019
- PR-AUC: 0.9431 +- 0.0093
- Precision@K: 0.9970 +- 0.0054
- Runtime (train sec): 397.19

## 4. Stability Comment
Observed metric spread across evaluated slices: AUC span=0.0008, F1 span=0.0080, PR-AUC span=0.0081.

## 5. Limitation
Requested 1/2/4/8-hour slices were not feasible under inferred timestamp span/units for this file, so a structured early/middle/late pilot was run instead. This is a sensitivity proxy and should be followed by true longer-hour slices when full-duration traces are available.

---

# Cross-Dataset Robustness Under Corrected Protocol

## 1. Objective
This experiment applies the same corrected rolling G_t -> G_{t+1} protocol across Alibaba 2022, Alibaba 2021, and Huawei 2021 to test whether weak Huawei performance persists after protocol corrections.

## 2. Shared Protocol
- Model: GAT
- Window size: 100
- Rolling direction: G_t -> E_(t+1)
- Train/val split by window index: train up to G_68, val starts at G_60
- Test starts at G_70
- Negative evaluation ratios: 1,5,10,50

## 3. Dataset Metadata
### Alibaba 2022
- Rows after preprocessing: 13332356
- Services: 17691
- Windows: 1801
- Train/Val pairs: 60/9
- Best epoch: 31
- Training time (sec): 196.59

### Alibaba 2021
- Rows after preprocessing: 6018598
- Services: 7393
- Windows: 3001
- Train/Val pairs: 60/9
- Best epoch: 16
- Training time (sec): 50.61

### Huawei 2021
- Rows after preprocessing: 465933
- Services: 82
- Windows: 2352
- Train/Val pairs: 60/9
- Best epoch: 1
- Training time (sec): 5.19

## 4. Performance by Dataset and Ratio
### Alibaba 2022
- 1:1 | AUC=0.9682, Precision=0.6465, Recall=0.9721, F1=0.7765, Accuracy=0.7202, TestPairs=1729
- 1:5 | AUC=0.9682, Precision=0.2677, Recall=0.9721, F1=0.4197, Accuracy=0.5521, TestPairs=1729
- 1:10 | AUC=0.9682, Precision=0.1545, Recall=0.9721, F1=0.2666, Accuracy=0.5139, TestPairs=1729
- 1:50 | AUC=0.9705, Precision=0.0354, Recall=0.9747, F1=0.0684, Accuracy=0.4794, TestPairs=300

### Alibaba 2021
- 1:1 | AUC=0.9764, Precision=0.6428, Recall=0.9781, F1=0.7757, Accuracy=0.7173, TestPairs=2929
- 1:5 | AUC=0.9764, Precision=0.2646, Recall=0.9781, F1=0.4165, Accuracy=0.5433, TestPairs=2929
- 1:10 | AUC=0.9764, Precision=0.1524, Recall=0.9781, F1=0.2638, Accuracy=0.5036, TestPairs=2929
- 1:50 | AUC=0.9749, Precision=0.0346, Recall=0.9765, F1=0.0668, Accuracy=0.4653, TestPairs=300

### Huawei 2021
- 1:1 | AUC=0.5775, Precision=0.5324, Recall=0.9563, F1=0.6790, Accuracy=0.5591, TestPairs=1096
- 1:5 | AUC=0.5730, Precision=0.1863, Recall=0.9563, F1=0.3104, Accuracy=0.2895, TestPairs=1096
- 1:10 | AUC=0.5755, Precision=0.1035, Recall=0.9563, F1=0.1862, Accuracy=0.2335, TestPairs=1096
- 1:50 | AUC=0.6319, Precision=0.0232, Recall=0.8698, F1=0.0452, Accuracy=0.2783, TestPairs=300

## 5. Huawei Persistence Check
Huawei still underperforms under the corrected protocol if its AUC/F1 remain materially lower than both Alibaba datasets at the same ratio.
At 1:10, AUC: Huawei=0.5755, Alibaba2022=0.9682, Alibaba2021=0.9764.
At 1:10, F1: Huawei=0.1862, Alibaba2022=0.2666, Alibaba2021=0.2638.

---

# Ranking-Based Evaluation Report (G_t -> G_{t+1})

## 1. Objective
This experiment evaluates ranking quality of future link prediction, which is more deployment-relevant than strict binary classification when candidate calls are imbalanced.

## 2. Protocol
- Model: GAT
- Dataset: Data/Alibaba 2022/CallGraph_0.csv
- Window size: 100
- Forecasting direction: G_t -> E_{t+1}
- Training negatives: 1:1
- Ranking candidate negative ratio at test: 1:50
- Candidate set per window = true future edges + sampled non-edges
- Test start input: G_70
- Best training epoch: 39
- Training time (sec): 261.74

## 3. Ranking Metrics (Averaged Over Rolling Test Pairs)
- Evaluated test pairs: 1729
- Hits@10: 1.0000 +- 0.0000
- Hits@50: 1.0000 +- 0.0000
- Hits@100: 1.0000 +- 0.0000
- Precision@10: 0.9836 +- 0.0489
- Precision@50: 0.9904 +- 0.0170
- Precision@100: 0.9923 +- 0.0107
- MRR: 0.9783 +- 0.1051
- PR-AUC: 0.8802 +- 0.0216

## 4. Interpretation
These ranking results indicate whether top-ranked predicted links capture true future service calls, supporting operational triage value even when threshold-based F1 degrades under imbalanced test distributions.

---

# Runtime Measurement Breakdown

## 1. Objective
Refine runtime reporting under the corrected rolling protocol by separating data, graph, training, and evaluation components.

## 2. Shared Runtime Components
- Data loading time: 28.87 sec
- Graph construction time: 0.75 sec
- Node count after cap: 1997
- Transformer attention estimate: 0.24 GB

## 3. Per-Model Runtime Breakdown
### GAT
- Training time per epoch (mean +- std): 2.277 +- 0.244 sec
- Total training time: 61.49 sec
- Model inference time (test loop forward only): 0.27 sec
- End-to-end evaluation time: 4.16 sec
- Evaluation overhead (sampling + scoring + metrics): 3.88 sec
- AUC / PR-AUC: 0.9690 / 0.8845

### Diffusion
- Training time per epoch (mean +- std): 2.457 +- 0.122 sec
- Total training time: 46.68 sec
- Model inference time (test loop forward only): 0.31 sec
- End-to-end evaluation time: 3.11 sec
- Evaluation overhead (sampling + scoring + metrics): 2.80 sec
- AUC / PR-AUC: 0.8954 / 0.7160

### DiffusionGAT
- Training time per epoch (mean +- std): 3.844 +- 0.227 sec
- Total training time: 115.32 sec
- Model inference time (test loop forward only): 0.60 sec
- End-to-end evaluation time: 4.08 sec
- Evaluation overhead (sampling + scoring + metrics): 3.48 sec
- AUC / PR-AUC: 0.9053 / 0.6182

### Transformer
- Training time per epoch (mean +- std): 69.122 +- 1.107 sec
- Total training time: 2073.66 sec
- Model inference time (test loop forward only): 2.43 sec
- End-to-end evaluation time: 7.05 sec
- Evaluation overhead (sampling + scoring + metrics): 4.62 sec
- AUC / PR-AUC: 0.9867 / 0.9623

### TransformerGAT
- Training time per epoch (mean +- std): 70.202 +- 4.513 sec
- Total training time: 1895.46 sec
- Model inference time (test loop forward only): 2.53 sec
- End-to-end evaluation time: 6.60 sec
- Evaluation overhead (sampling + scoring + metrics): 4.07 sec
- AUC / PR-AUC: 0.9679 / 0.8811

## 4. Bottleneck Interpretation
- GAT: pipeline=4.64s vs neural=61.76s -> Model-dominant
- Diffusion: pipeline=3.56s vs neural=46.99s -> Model-dominant
- DiffusionGAT: pipeline=4.23s vs neural=115.92s -> Model-dominant
- Transformer: pipeline=5.37s vs neural=2076.09s -> Model-dominant
- TransformerGAT: pipeline=4.82s vs neural=1897.99s -> Model-dominant

## 5. Note
This is a precision runtime redo only; it does not require rerunning every historical dataset/window combination.

---

# Matched Statistical Tests Report

## 1. Objective
This experiment reruns corrected evaluations and computes paired statistical tests only on matched experimental units.

## 2. Matched Unit Definition
- Primary unit for paired tests: seed + dataset + test target window
- Model comparisons use the same Alibaba 2022 rolling windows under the corrected G_t -> E_(t+1) protocol
- Overlap comparison matches on the same target window start times; overlap rows without a non-overlap target match are excluded
- Alibaba vs Huawei is not forced into a paired test because dataset identity changes the unit itself

## 3. Configuration
- Seeds: 0,1,2,3,4
- Rolling window size: 100
- Rolling split: train_end_index=69, val_start_index=60, test_start_index=70
- Model-comparison negative ratio: 1:1
- Model-comparison max_time cap: 10000
- Model-comparison max_nodes cap: 2000
- Overlap negative ratio: 1:10
- Dataset descriptive negative ratio: 1:10

## 4. Paired Test Summary
### GAT_vs_DiffusionGAT
- AUC: n=145, mean diff (DiffusionGAT - GAT)=-0.072363, paired t p=1.94e-83, Wilcoxon p=1.525e-25, Cohen's d_z=-3.5335
- F1: n=145, mean diff (DiffusionGAT - GAT)=-0.024148, paired t p=1.333e-43, Wilcoxon p=1.764e-25, Cohen's d_z=-1.6682
- Accuracy: n=145, mean diff (DiffusionGAT - GAT)=-0.021623, paired t p=2.488e-35, Wilcoxon p=9.05e-24, Cohen's d_z=-1.3811

### Diffusion_vs_DiffusionGAT
- AUC: n=145, mean diff (DiffusionGAT - Diffusion)=-0.005383, paired t p=0.007371, Wilcoxon p=0.0113, Cohen's d_z=-0.2257
- F1: n=145, mean diff (DiffusionGAT - Diffusion)=-0.021282, paired t p=7.012e-13, Wilcoxon p=3.106e-10, Cohen's d_z=-0.6551
- Accuracy: n=145, mean diff (DiffusionGAT - Diffusion)=-0.038431, paired t p=1.361e-14, Wilcoxon p=1.96e-10, Cohen's d_z=-0.7127

### Transformer_vs_TransformerGAT
- AUC: n=145, mean diff (TransformerGAT - Transformer)=-0.029447, paired t p=1.931e-49, Wilcoxon p=1.525e-25, Cohen's d_z=-1.8858
- F1: n=145, mean diff (TransformerGAT - Transformer)=-0.008124, paired t p=6.054e-21, Wilcoxon p=6.994e-20, Cohen's d_z=-0.9178
- Accuracy: n=145, mean diff (TransformerGAT - Transformer)=-0.007928, paired t p=1.383e-16, Wilcoxon p=4.918e-15, Cohen's d_z=-0.7782

### 100ms_non_overlap_vs_100ms_overlap_50pct
- AUC: n=145, mean diff (100ms_overlap_50pct - 100ms_non_overlap)=0.013500, paired t p=1.036e-23, Wilcoxon p=6.479e-25, Cohen's d_z=1.0054
- F1: n=145, mean diff (100ms_overlap_50pct - 100ms_non_overlap)=-0.006282, paired t p=7.863e-26, Wilcoxon p=4.81e-19, Cohen's d_z=-1.0726
- PR_AUC: n=145, mean diff (100ms_overlap_50pct - 100ms_non_overlap)=0.030732, paired t p=1.585e-38, Wilcoxon p=1.623e-25, Cohen's d_z=1.4895
- PrecisionAt100: n=145, mean diff (100ms_overlap_50pct - 100ms_non_overlap)=0.014345, paired t p=9.138e-08, Wilcoxon p=4.052e-08, Cohen's d_z=0.4675

## 5. Dataset Comparison Note
- Alibaba2022_vs_Huawei2021: Dataset identity is part of the experimental unit, so Alibaba-vs-Huawei observations are not the same seed-window-dataset units required for paired inference. Descriptive summaries are reported instead.

## 6. Descriptive Dataset Means
- Alibaba2022: AUC=0.9664, F1=0.2680, Accuracy=0.5180 over 8645 matched seed-window evaluations
- Huawei2021: AUC=0.7168, F1=0.1902, Accuracy=0.2528 over 5480 matched seed-window evaluations

## 7. Outputs Verified
- `paired_statistical_tests_raw_units.csv`: present
- `paired_statistical_tests_summary.csv`: present
- `paired_statistical_tests_summary.json`: present
- `paired_statistical_tests_report.md`: present

### Stage Execution Status
- model: [PASS] ok
- overlap: [PASS] ok
- dataset: [PASS] ok

---

# Preprocessing Statistics Per Dataset

## 1. Objective
This report quantifies preprocessing effects per dataset, including bad-line removal and dropna filtering, to address reviewer concerns about dataset noise and cleaning impact.

## 2. Metrics Reported
- raw rows
- rows after bad-line removal
- rows after dropna()
- percentage removed
- unique services
- unique directed edges
- number of windows
- mean/median edges per window
- edge density

## 3. Dataset Statistics
### Alibaba 2021
- Source file: Data\MSCallGraph_0.csv
- Window size used: 100.0
- Raw rows: 6088846
- Rows after bad-line removal: 6088846
- Rows after dropna(): 6018598
- Percentage removed: 1.15%
- Unique services: 7393
- Unique directed edges: 16566
- Number of windows: 3000
- Mean edges per window: 2006.20
- Median edges per window: 1917.50
- Edge density: 0.00030313
- Runtime (sec): 39.38

### Alibaba 2022
- Source file: Data\Alibaba 2022\CallGraph_0.csv
- Window size used: 100.0
- Raw rows: 13332356
- Rows after bad-line removal: 13332085
- Rows after dropna(): 13332085
- Percentage removed: 0.00%
- Unique services: 17687
- Unique directed edges: 58583
- Number of windows: 1800
- Mean edges per window: 7406.71
- Median edges per window: 7334.50
- Edge density: 0.00018728
- Runtime (sec): 62.01

### Huawei 2021
- Source file: Data\Huawei\status_1min_20210411.csv
- Window size used: 100.0
- Raw rows: 465933
- Rows after bad-line removal: 465933
- Rows after dropna(): 465933
- Percentage removed: 0.00%
- Unique services: 82
- Unique directed edges: 292
- Number of windows: 2351
- Mean edges per window: 198.19
- Median edges per window: 0.00
- Edge density: 0.04396266
- Runtime (sec): 2.78

## 4. Notes
dropna() is applied after schema normalization to required modeling fields (timestamp, um, dm), matching the paper's unified preprocessing pattern.

---

## Coverage Summary
- Included reports: 12
- Missing reports: 2
  - downstream_operational_proxy_report.md
  - full_trace_drift_report.md