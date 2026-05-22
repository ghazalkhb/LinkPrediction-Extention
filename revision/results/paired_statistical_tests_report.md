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