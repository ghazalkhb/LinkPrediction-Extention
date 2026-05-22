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