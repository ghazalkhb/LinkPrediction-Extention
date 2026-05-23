# Link Prediction for Microservice Call Graphs: Temporal Windows and Scalability Tradeoffs

This repository contains the implementation for the paper *"Temporal Graph Models for Predictive Monitoring in Microservices"*. It includes the original model implementations and a full set of revised-paper experiments covering strict temporal evaluation, imbalanced testing, ranking metrics, multi-horizon forecasting, statistical significance, cross-dataset robustness, and runtime analysis.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Revision Experiments](#revision-experiments)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Overview

This project predicts future service interactions in microservice call graphs using temporal graph neural networks. Key aspects:

- **Strict temporal evaluation**: rolling `G_t → G_{t+1}` forecasting protocol (no data leakage)
- **Multiple model architectures**: GAT, Diffusion, Transformer, LSTM, hybrid models, NodeSim
- **Comprehensive evaluation**: AUC, MRR, Precision@K, Recall@K, PR-AUC, F1 under varying imbalance ratios
- **Multi-horizon forecasting**: `G_t → G_{t+k}` for k = 1, 5, 10, 30
- **Three production datasets**: Alibaba 2021, Alibaba 2022, Huawei 2021

## Project Structure

```
├── Code/
│   ├── LinkPrediction.ipynb          # Main experiments notebook
│   ├── evaluate.py                   # Evaluation functions
│   ├── negative_sampling.py          # Negative sampling strategies
│   ├── Models/
│   │   ├── GNN_Model.py              # GAT implementation
│   │   ├── Standalone_Diffusion_Model.py
│   │   ├── Diffusion_GAT_Model.py
│   │   ├── Standalone_Transformer_Model.py
│   │   └── Transformer_GAT_Model.py
│   ├── LSTM/
│   │   └── main.py
│   └── NodeSim/                      # Third-party node similarity module
│
├── revision/                         # Revised-paper experiments (new)
│   ├── rolling_gt_to_gt1_experiment.py
│   ├── horizon_gt_to_gtk_experiment.py
│   ├── imbalanced_negative_evaluation.py
│   ├── ranking_evaluation.py
│   ├── training_testing_gap_experiment.py
│   ├── temporal_slice_sensitivity.py
│   ├── overlap_fair_experiment.py
│   ├── cross_dataset_robustness.py
│   ├── paired_statistical_tests.py
│   ├── runtime_measurement_breakdown.py
│   ├── recurring_activity_experiment.py
│   ├── full_trace_drift_experiment.py
│   ├── downstream_operational_proxy_experiment.py
│   ├── preprocessing_dataset_statistics.py
│   ├── degree_aware_sampling.py
│   ├── run_all_experiments.py
│   ├── build_full_revision_report.py
│   ├── narval/                       # HPC cluster submission scripts (SLURM)
│   └── results/                      # All experiment outputs (CSV, JSON, MD)
│
└── Results/                          # Original paper results
```

## Models Implemented

| Model | File | Notes |
|---|---|---|
| GAT | `Code/Models/GNN_Model.py` | Primary model; best accuracy/speed tradeoff |
| Diffusion | `Code/Models/Standalone_Diffusion_Model.py` | Global diffusion propagation |
| DiffusionGAT | `Code/Models/Diffusion_GAT_Model.py` | Hybrid |
| Transformer | `Code/Models/Standalone_Transformer_Model.py` | Highest AUC, ~30x slower |
| TransformerGAT | `Code/Models/Transformer_GAT_Model.py` | Hybrid |
| LSTM | `Code/LSTM/main.py` | Sequential baseline |
| NodeSim | `Code/NodeSim/` | Third-party, community-aware embeddings |

## Revision Experiments

All scripts are in `revision/`. Each writes its outputs to `revision/results/`.

| Script | What it tests |
|---|---|
| `rolling_gt_to_gt1_experiment.py` | Strict rolling evaluation (1,729 forecast instances on Alibaba 2022) |
| `horizon_gt_to_gtk_experiment.py` | Multi-horizon forecasting (k = 1, 5, 10, 30) |
| `imbalanced_negative_evaluation.py` | Test-time imbalance sensitivity (1:1 to 1:50 negative ratios) |
| `ranking_evaluation.py` | Ranking metrics: Hits@K, Precision@K, MRR, PR-AUC (1:50 degree-aware negatives) |
| `training_testing_gap_experiment.py` | Prediction degradation as train/test gap grows |
| `temporal_slice_sensitivity.py` | Effect of time-window size and overlap ratio |
| `overlap_fair_experiment.py` | Fair comparison: overlapping vs non-overlapping windows |
| `cross_dataset_robustness.py` | Unified protocol across Alibaba 2021, 2022, Huawei 2021 |
| `paired_statistical_tests.py` | Matched paired t-test + Wilcoxon, n=145 (5 seeds x 29 windows) |
| `runtime_measurement_breakdown.py` | Forward pass, training epoch, end-to-end latency breakdown |
| `recurring_activity_experiment.py` | Effect of recurring call patterns on prediction quality |
| `full_trace_drift_experiment.py` | Distribution shift over full trace duration |
| `downstream_operational_proxy_experiment.py` | Proxy for downstream operational value |
| `preprocessing_dataset_statistics.py` | Dataset statistics (nodes, edges, density, window counts) |
| `degree_aware_sampling.py` | Degree-aware negative sampling utility |
| `run_all_experiments.py` | Runs all the above in sequence |
| `build_full_revision_report.py` | Aggregates all results/ into a single markdown report |

### Key Results (Alibaba 2022, corrected rolling protocol)

| Experiment | Key finding |
|---|---|
| Rolling evaluation (GAT, k=1) | AUC = 0.971, MRR = 0.998, F1 = 0.778, 1,729 forecast instances |
| Ranking (1:50 negatives) | PR-AUC = 0.880, Precision@100 = 0.992, MRR = 0.978 |
| Imbalance (1:10) | F1 drops to 0.26; AUC remains stable (~0.97) |
| Multi-horizon (k=30) | AUC = 0.961 (-0.007 from k=1); Precision@100 ~= 1.0 |
| Cross-dataset | Alibaba 2021: AUC = 0.976; Huawei 2021: AUC = 0.577 |
| Statistical tests | GAT > DiffusionGAT: p < 1e-82, d_z = -3.53; n=145 |
| Runtime (GAT) | Forward 9.5 ms/window; end-to-end ~143 ms/forecast instance |

### Running the Revision Experiments

```bash
# Run a single experiment
python revision/rolling_gt_to_gt1_experiment.py --data-path Data/MSCallGraph_0.csv

# Run all experiments
python revision/run_all_experiments.py --data-path Data/MSCallGraph_0.csv

# Build the combined report
python revision/build_full_revision_report.py
```

Use `python revision/<script>.py --help` to see all available arguments.

For HPC/SLURM clusters, see `revision/narval/`.

## Datasets

| Dataset | Source |
|---|---|
| Alibaba Microservices 2022 | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022 |
| Alibaba Microservices 2021 | https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021 |
| Huawei Cloud Trace 2021 | https://zenodo.org/record/5638238 |


## Usage

**Original notebook (Code/):**
```bash
jupyter notebook Code/LinkPrediction.ipynb
```

**Revision experiments (revision/):**
```bash
python revision/rolling_gt_to_gt1_experiment.py --data-path Data/MSCallGraph_0.csv
python revision/horizon_gt_to_gtk_experiment.py --data-path Data/MSCallGraph_0.csv
python revision/run_all_experiments.py --data-path Data/MSCallGraph_0.csv
```

## Results

Detailed per-experiment results are in `revision/results/` as `.csv`, `.json`, and `.md` files. `revision/FULL_EXPERIMENT_REPORT.md` aggregates all findings.

Original partial results are in `Results/`.
