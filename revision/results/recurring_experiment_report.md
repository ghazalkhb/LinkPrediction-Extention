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