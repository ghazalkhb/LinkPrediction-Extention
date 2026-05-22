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