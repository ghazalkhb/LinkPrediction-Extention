# Runtime Measurement Breakdown

## 1. Objective
Refine runtime reporting under the corrected rolling protocol by separating data, graph, training, and evaluation components.

## 2. Shared Runtime Components
- Data loading time: 28.87 sec
- Graph construction time: 0.75 sec
- Node count after cap: 1997
- Transformer attention estimate: 0.24 GB

## 3. Per-Model Runtime Breakdown
### GAT
- Training time per epoch (mean +- std): 2.277 +- 0.244 sec
- Total training time: 61.49 sec
- Model inference time (test loop forward only): 0.27 sec
- End-to-end evaluation time: 4.16 sec
- Evaluation overhead (sampling + scoring + metrics): 3.88 sec
- AUC / PR-AUC: 0.9690 / 0.8845

### Diffusion
- Training time per epoch (mean +- std): 2.457 +- 0.122 sec
- Total training time: 46.68 sec
- Model inference time (test loop forward only): 0.31 sec
- End-to-end evaluation time: 3.11 sec
- Evaluation overhead (sampling + scoring + metrics): 2.80 sec
- AUC / PR-AUC: 0.8954 / 0.7160

### DiffusionGAT
- Training time per epoch (mean +- std): 3.844 +- 0.227 sec
- Total training time: 115.32 sec
- Model inference time (test loop forward only): 0.60 sec
- End-to-end evaluation time: 4.08 sec
- Evaluation overhead (sampling + scoring + metrics): 3.48 sec
- AUC / PR-AUC: 0.9053 / 0.6182

### Transformer
- Training time per epoch (mean +- std): 69.122 +- 1.107 sec
- Total training time: 2073.66 sec
- Model inference time (test loop forward only): 2.43 sec
- End-to-end evaluation time: 7.05 sec
- Evaluation overhead (sampling + scoring + metrics): 4.62 sec
- AUC / PR-AUC: 0.9867 / 0.9623

### TransformerGAT
- Training time per epoch (mean +- std): 70.202 +- 4.513 sec
- Total training time: 1895.46 sec
- Model inference time (test loop forward only): 2.53 sec
- End-to-end evaluation time: 6.60 sec
- Evaluation overhead (sampling + scoring + metrics): 4.07 sec
- AUC / PR-AUC: 0.9679 / 0.8811

## 4. Bottleneck Interpretation
- GAT: pipeline=4.64s vs neural=61.76s -> Model-dominant
- Diffusion: pipeline=3.56s vs neural=46.99s -> Model-dominant
- DiffusionGAT: pipeline=4.23s vs neural=115.92s -> Model-dominant
- Transformer: pipeline=5.37s vs neural=2076.09s -> Model-dominant
- TransformerGAT: pipeline=4.82s vs neural=1897.99s -> Model-dominant

## 5. Note
This is a precision runtime redo only; it does not require rerunning every historical dataset/window combination.