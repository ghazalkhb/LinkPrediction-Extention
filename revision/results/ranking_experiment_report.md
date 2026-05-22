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