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