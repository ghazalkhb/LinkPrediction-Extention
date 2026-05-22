# Revision Experiments

This folder contains new paper-revision experiments only.

## Added Experiment

- `rolling_gt_to_gt1_experiment.py`
  - Implements strict temporal forecasting evaluation:
    - Train horizon: `G_0 ... G_69`
    - Validation: later training pairs (default `G_60 -> E_61 ... G_68 -> E_69`)
    - Test: `G_70 -> E_71`, `G_71 -> E_72`, `G_72 -> E_73`, ...
  - Labels are always built from `G_{t+1}`.
  - Negatives are sampled from node pairs that are not edges in `G_{t+1}`.
  - Saves:
    - `results/rolling_per_pair_metrics.csv`
    - `results/rolling_summary.json`
    - `results/rolling_experiment_report.md`

- `horizon_gt_to_gtk_experiment.py`
  - Implements multi-horizon forecasting evaluation:
    - Input graph at time `t`: `G_t`
    - Target edges at future horizon `k`: `E_{t+k}`
    - Default horizons: `k = 1, 5, 10, 30`
  - Reports and saves per-horizon:
    - AUC
    - Average Precision (PR-AUC)
    - Precision@K and Recall@K
    - F1 (secondary)
    - Inference time per prediction horizon
  - Saves:
    - `results/horizon_per_pair_metrics.csv`
    - `results/horizon_summary.csv`
    - `results/horizon_summary.json`
    - `results/horizon_experiment_report.md`

- `imbalanced_negative_evaluation.py`
  - Re-evaluates `G_t -> E_{t+1}` under imbalanced test negatives:
    - `1:1`, `1:5`, `1:10`, `1:50` (positive:negative)
  - Keeps training at 1:1 negatives.
  - Reports how test distribution affects:
    - Precision
    - Recall
    - F1
    - Accuracy
    - AUC
  - Includes feasibility check for all-non-edges evaluation on smaller datasets/windows.
  - Saves:
    - `results/imbalanced_per_pair_metrics.csv`
    - `results/imbalanced_summary.csv`
    - `results/imbalanced_summary.json`
    - `results/imbalanced_experiment_report.md`

- `ranking_evaluation.py`
  - Adds ranking-oriented evaluation for `G_t -> E_{t+1}`.
  - For each rolling test step, candidate service pairs are scored and ranked.
  - Reports:
    - Hits@10, Hits@50, Hits@100
    - Precision@10, Precision@50, Precision@100
    - MRR
    - PR-AUC
  - Saves:
    - `results/ranking_per_pair_metrics.csv`
    - `results/ranking_summary.csv`
    - `results/ranking_summary.json`
    - `results/ranking_experiment_report.md`

- `overlap_fair_experiment.py`
  - Fairly compares overlap/non-overlap windowing settings:
    - 50 ms non-overlap
    - 100 ms non-overlap
    - 100 ms with 50% overlap
    - 200 ms non-overlap
    - 500 ms non-overlap
  - Reports per setting:
    - number of train/test windows
    - average edges per window
    - duplicated edge percentage from overlap
    - AUC, F1, PR-AUC, Precision@K
    - runtime
  - Saves:
    - `results/overlap_fair_summary.csv`
    - `results/overlap_fair_summary.json`
    - `results/overlap_fair_experiment_report.md`

- `preprocessing_dataset_statistics.py`
  - Computes preprocessing statistics per dataset (Alibaba 2021, Alibaba 2022, Huawei 2021).
  - Reports:
    - raw rows
    - rows after bad-line removal
    - rows after dropna() on required columns (`timestamp`, `um`, `dm`)
    - percentage removed
    - unique services
    - unique directed edges
    - number of windows
    - mean/median edges per window
    - edge density
  - Saves:
    - `results/preprocessing_dataset_stats.csv`
    - `results/preprocessing_dataset_stats.json`
    - `results/preprocessing_dataset_stats_report.md`

- `cross_dataset_robustness.py`
  - Runs the corrected final protocol on all three datasets with shared settings:
    - Alibaba 2022
    - Alibaba 2021
    - Huawei 2021
  - Shared protocol components:
    - same model
    - same window size
    - same rolling `G_t -> E_{t+1}` evaluation setup
    - same negative evaluation ratios
  - Reports whether Huawei underperformance persists under the corrected protocol.
  - Saves:
    - `results/cross_dataset_robustness_summary.csv`
    - `results/cross_dataset_robustness_summary.json`
    - `results/cross_dataset_robustness_report.md`

- `temporal_slice_sensitivity.py`
  - Adds longer temporal slice sensitivity analysis for reviewer concerns on short slices.
  - Supports two modes:
    - hour-based pilot (`1h, 2h, 4h, 8h`) when timeline span supports it
    - fallback early/middle/late slice pilot with explicit limitation note
  - Reports slice-level stability of:
    - graph statistics (services, windows, edges/window, density)
    - metrics (AUC, F1, PR-AUC, Precision@K)
    - runtime
  - Saves:
    - `results/temporal_slice_sensitivity_summary.csv`
    - `results/temporal_slice_sensitivity_summary.json`
    - `results/temporal_slice_sensitivity_report.md`

- `runtime_measurement_breakdown.py`
  - Redoes runtime reporting under the corrected rolling protocol with finer-grained timing.
  - Separates:
    - data loading time
    - graph construction time
    - training time per epoch
    - total training time
    - model inference time
    - end-to-end evaluation time
  - Covers all five models:
    - GAT
    - Diffusion
    - DiffusionGAT
    - Transformer
    - TransformerGAT
  - Includes a bottleneck interpretation comparing pipeline overhead against neural-model time.
  - Saves:
    - `results/runtime_breakdown_summary.csv`
    - `results/runtime_breakdown_epoch_times.csv`
    - `results/runtime_breakdown_summary.json`
    - `results/runtime_breakdown_report.md`

- `downstream_operational_proxy_experiment.py`
  - Adds downstream operational proxy validation requested by reviewers.
  - Links `G_t -> E_{t+1}` prediction quality to operational triage behavior:
    - anomaly-alert precision/recall/F1 using edge-volume/churn anomalies
    - triage precision@K and recall of new edges@K
    - optional RT-based SLO-risk proxy recall@K when `rt` exists in dataset
  - Includes explicit limitation text: this is proxy validation, not full closed-loop SLO/anomaly mitigation in production.
  - Saves:
    - `results/downstream_operational_proxy_per_pair.csv`
    - `results/downstream_operational_proxy_summary.csv`
    - `results/downstream_operational_proxy_summary.json`
    - `results/downstream_operational_proxy_report.md`

- `full_trace_drift_experiment.py`
  - Adds long-horizon drift analysis over the full available trace timeline.
  - Trains on early days and evaluates day-by-day metric stability on later days.
  - Reports drift of AUC/F1 relative to first evaluation day.
  - This directly addresses reviewer concern that slice sensitivity does not replace full-trace drift analysis.
  - Saves:
    - `results/full_trace_drift_per_pair_metrics.csv`
    - `results/full_trace_drift_daily_summary.csv`
    - `results/full_trace_drift_summary.csv`
    - `results/full_trace_drift_summary.json`
    - `results/full_trace_drift_report.md`

- `download_alibaba2022_microservices.py`
  - Downloads Alibaba v2022 callgraph archives from the official source and builds normalized `CallGraph_0.csv`.
  - Normalizes schema to required columns (`timestamp`, `um`, `dm`) and optionally keeps `rt`.

- `build_full_revision_report.py`
  - Builds a single consolidated markdown report from all result reports.
  - Output default: `revision/FULL_EXPERIMENT_REPORT.md`

## Run

From the project root:

```powershell
python revision/rolling_gt_to_gt1_experiment.py --dataset-path CallGraph_0.csv --model GAT
```

If your dataset is elsewhere, provide the full path:

```powershell
python revision/rolling_gt_to_gt1_experiment.py --dataset-path E:/path/to/CallGraph_0.csv --model GAT
```

The generated markdown report is the complete saved write-up for this experiment.

## Multi-Horizon Run

```powershell
python revision/horizon_gt_to_gtk_experiment.py --dataset-path Data/CallGraph_0.csv --model GAT --horizons 1,5,10,30 --time-window-size 100 --top-k 100
```

## Imbalanced Negative Test Run

```powershell
python revision/imbalanced_negative_evaluation.py --dataset-path Data/CallGraph_0.csv --model GAT --eval-ratios 1,5,10,50 --time-window-size 100
```

## Ranking Evaluation Run

```powershell
python revision/ranking_evaluation.py --dataset-path Data/CallGraph_0.csv --model GAT --time-window-size 100 --candidate-neg-ratio 50
```

## Fair Overlap Comparison Run

```powershell
python revision/overlap_fair_experiment.py --dataset-path Data/CallGraph_0.csv --model GAT --train-time-end 7000 --test-time-start 7000 --max-time 10000 --top-k 100
```

## Preprocessing Statistics Run

```powershell
python revision/preprocessing_dataset_statistics.py --results-dir revision/results
```

## Cross-Dataset Robustness Run

```powershell
python revision/cross_dataset_robustness.py --results-dir revision/results --model GAT --window-size 100 --eval-ratios 1,5,10,50
```

## Temporal Slice Sensitivity Run

```powershell
python revision/temporal_slice_sensitivity.py --dataset-path "Data/Alibaba 2022/CallGraph_0.csv" --results-dir revision/results --model GAT --window-size 100 --slice-mode auto
```

## Runtime Measurement Breakdown Run

```powershell
python revision/runtime_measurement_breakdown.py --dataset-path "Data/Alibaba 2022/CallGraph_0.csv" --results-dir revision/results
```

## Downstream Operational Proxy Validation Run

```powershell
python revision/downstream_operational_proxy_experiment.py --dataset-path "Data/Alibaba 2022/CallGraph_0.csv" --results-dir revision/results --model GAT --top-k 100 --candidate-neg-ratio 50
```

## Full-Trace Drift Run

```powershell
python revision/full_trace_drift_experiment.py --dataset-path "Data/Alibaba 2022/CallGraph_0.csv" --results-dir revision/results --model GAT --window-size 60000 --train-days 2
```

## Download Alibaba 2022 Data Run

```powershell
python revision/download_alibaba2022_microservices.py --start-date 0d0 --end-date 0d1 --output-dir "Data/Alibaba 2022" --output-csv CallGraph_0.csv --include-rt
```

## Build Consolidated Full Report

```powershell
python revision/build_full_revision_report.py --results-dir revision/results --output revision/FULL_EXPERIMENT_REPORT.md
```

## Narval GPU Submission

See `revision/narval/README.md` for `sbatch` commands and GPU setup.
