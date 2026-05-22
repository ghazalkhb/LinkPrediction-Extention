# Downstream Operational Proxy Validation Report

## 1. Objective
This experiment evaluates a downstream proxy for operational benefit by mapping link-prediction outputs to anomaly-triage effectiveness.

## 2. Protocol
- Model: GAT
- Dataset: Data/MSCallGraph_0.csv
- Forecasting: G_t -> E_(t+1)
- Top-K triage budget: 100
- Candidate negative ratio: 1:50
- Window anomaly percentile (edge volume/churn): p90.0
- Alarm threshold percentile over validation risk score: p90.0
- Training runtime (sec): 13.35

## 3. Results
- AUC: 0.9632 +- 0.0283
- PR-AUC: 0.8876 +- 0.0519
- Triage Precision@K: 0.9894 +- 0.0138
- Triage Recall(New Edges)@K: 0.0220 +- 0.0476
- Anomaly Alert Precision: 0.1182 +- 0.0000
- Anomaly Alert Recall: 0.3609 +- 0.0000
- Anomaly Alert F1: 0.1781 +- 0.0000
- RT-based SLO proxy recall@K: 1.4668 +- 0.7663

## 4. Limitation Statement
This is a proxy-level operational validation (anomaly triage and optional RT-risk proxy), not a full closed-loop production evaluation of SLO compliance or mitigation outcomes. Full downstream operational validation remains future work.