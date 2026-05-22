# Cross-Dataset Robustness Under Corrected Protocol

## 1. Objective
This experiment applies the same corrected rolling G_t -> G_{t+1} protocol across Alibaba 2022, Alibaba 2021, and Huawei 2021 to test whether weak Huawei performance persists after protocol corrections.

## 2. Shared Protocol
- Model: GAT
- Window size: 100
- Rolling direction: G_t -> E_(t+1)
- Train/val split by window index: train up to G_68, val starts at G_60
- Test starts at G_70
- Negative evaluation ratios: 1,5,10,50

## 3. Dataset Metadata
### Alibaba 2022
- Rows after preprocessing: 13332356
- Services: 17691
- Windows: 1801
- Train/Val pairs: 60/9
- Best epoch: 31
- Training time (sec): 196.59

### Alibaba 2021
- Rows after preprocessing: 6018598
- Services: 7393
- Windows: 3001
- Train/Val pairs: 60/9
- Best epoch: 16
- Training time (sec): 50.61

### Huawei 2021
- Rows after preprocessing: 465933
- Services: 82
- Windows: 2352
- Train/Val pairs: 60/9
- Best epoch: 1
- Training time (sec): 5.19

## 4. Performance by Dataset and Ratio
### Alibaba 2022
- 1:1 | AUC=0.9682, Precision=0.6465, Recall=0.9721, F1=0.7765, Accuracy=0.7202, TestPairs=1729
- 1:5 | AUC=0.9682, Precision=0.2677, Recall=0.9721, F1=0.4197, Accuracy=0.5521, TestPairs=1729
- 1:10 | AUC=0.9682, Precision=0.1545, Recall=0.9721, F1=0.2666, Accuracy=0.5139, TestPairs=1729
- 1:50 | AUC=0.9705, Precision=0.0354, Recall=0.9747, F1=0.0684, Accuracy=0.4794, TestPairs=300

### Alibaba 2021
- 1:1 | AUC=0.9764, Precision=0.6428, Recall=0.9781, F1=0.7757, Accuracy=0.7173, TestPairs=2929
- 1:5 | AUC=0.9764, Precision=0.2646, Recall=0.9781, F1=0.4165, Accuracy=0.5433, TestPairs=2929
- 1:10 | AUC=0.9764, Precision=0.1524, Recall=0.9781, F1=0.2638, Accuracy=0.5036, TestPairs=2929
- 1:50 | AUC=0.9749, Precision=0.0346, Recall=0.9765, F1=0.0668, Accuracy=0.4653, TestPairs=300

### Huawei 2021
- 1:1 | AUC=0.5775, Precision=0.5324, Recall=0.9563, F1=0.6790, Accuracy=0.5591, TestPairs=1096
- 1:5 | AUC=0.5730, Precision=0.1863, Recall=0.9563, F1=0.3104, Accuracy=0.2895, TestPairs=1096
- 1:10 | AUC=0.5755, Precision=0.1035, Recall=0.9563, F1=0.1862, Accuracy=0.2335, TestPairs=1096
- 1:50 | AUC=0.6319, Precision=0.0232, Recall=0.8698, F1=0.0452, Accuracy=0.2783, TestPairs=300

## 5. Huawei Persistence Check
Huawei still underperforms under the corrected protocol if its AUC/F1 remain materially lower than both Alibaba datasets at the same ratio.
At 1:10, AUC: Huawei=0.5755, Alibaba2022=0.9682, Alibaba2021=0.9764.
At 1:10, F1: Huawei=0.1862, Alibaba2022=0.2666, Alibaba2021=0.2638.