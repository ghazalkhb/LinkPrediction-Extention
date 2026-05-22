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