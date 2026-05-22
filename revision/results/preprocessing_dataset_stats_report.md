# Preprocessing Statistics Per Dataset

## 1. Objective
This report quantifies preprocessing effects per dataset, including bad-line removal and dropna filtering, to address reviewer concerns about dataset noise and cleaning impact.

## 2. Metrics Reported
- raw rows
- rows after bad-line removal
- rows after dropna()
- percentage removed
- unique services
- unique directed edges
- number of windows
- mean/median edges per window
- edge density

## 3. Dataset Statistics
### Alibaba 2021
- Source file: Data\MSCallGraph_0.csv
- Window size used: 100.0
- Raw rows: 6088846
- Rows after bad-line removal: 6088846
- Rows after dropna(): 6018598
- Percentage removed: 1.15%
- Unique services: 7393
- Unique directed edges: 16566
- Number of windows: 3000
- Mean edges per window: 2006.20
- Median edges per window: 1917.50
- Edge density: 0.00030313
- Runtime (sec): 39.38

### Alibaba 2022
- Source file: Data\Alibaba 2022\CallGraph_0.csv
- Window size used: 100.0
- Raw rows: 13332356
- Rows after bad-line removal: 13332085
- Rows after dropna(): 13332085
- Percentage removed: 0.00%
- Unique services: 17687
- Unique directed edges: 58583
- Number of windows: 1800
- Mean edges per window: 7406.71
- Median edges per window: 7334.50
- Edge density: 0.00018728
- Runtime (sec): 62.01

### Huawei 2021
- Source file: Data\Huawei\status_1min_20210411.csv
- Window size used: 100.0
- Raw rows: 465933
- Rows after bad-line removal: 465933
- Rows after dropna(): 465933
- Percentage removed: 0.00%
- Unique services: 82
- Unique directed edges: 292
- Number of windows: 2351
- Mean edges per window: 198.19
- Median edges per window: 0.00
- Edge density: 0.04396266
- Runtime (sec): 2.78

## 4. Notes
dropna() is applied after schema normalization to required modeling fields (timestamp, um, dm), matching the paper's unified preprocessing pattern.