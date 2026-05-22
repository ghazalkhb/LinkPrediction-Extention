# Full-Trace Drift Experiment Report

## 1. Objective
This experiment evaluates metric drift over the full available trace timeline to assess long-term stability instead of only short temporal slices.

## 2. Protocol
- Dataset: /scratch/ghazalkh/alibaba2022_day1/CallGraph_0.csv
- Model: GAT
- Device: cuda
- Window size (ms): 3600000
- Train days: 0.25
- Evaluated days: 1

## 3. Aggregate Results
- AUC mean: 0.9237 +- 0.0024
- PR-AUC mean: 1.0000 +- 0.0000
- F1 mean: 0.9935 +- 0.0004
- Relative AUC drift vs first eval day: 0.00%
- Relative F1 drift vs first eval day: 0.00%

## 4. Day-level Trend
- Day 0: AUC=0.9237, PR-AUC=1.0000, F1=0.9935, pairs=2

## 5. Limitation Statement
This full-trace drift experiment improves long-horizon evidence but is still an offline forecasting analysis. It does not by itself establish online adaptation, retraining policy, or closed-loop operational control.