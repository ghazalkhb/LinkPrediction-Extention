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