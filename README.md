
# Link Prediction for Microservice Call Graphs: Temporal Windows and Scalability Tradeoffs

This repository contains an implementation of link prediction models for microservice call graphs, emphasizing temporal dynamics and scalability. It implements multiple deep learning architectures, includes a third-party NodeSim module from previous research, and evaluates performance using real-world datasets.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Datasets](#datasets)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Paper](#paper)
- [Contributing](#contributing)

## Overview

This project addresses the challenge of predicting future connections in microservice call graphs through temporal network analysis. Key aspects include:

- **Multiple Model Architectures**: GAT, Diffusion, Transformer, LSTM, hybrid models, and NodeSim.
- **Temporal Window Processing**: Dynamic segmentation of call graphs based on timestamps.
- **Advanced Negative Sampling**: Sophisticated strategies to enhance training.
- **Evaluation Metrics**: AUC, MRR, Precision, Recall, F1-score, Accuracy.

## Project Structure

```
LinkPrediction-Extention-main/
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
│   └── NodeSim/                      # External module from another paper
│       ├── src/
│       ├── Input/
│       ├── Output/
│       └── Adjusted/
├── Paper/
│   └── Link_Prediction_for_Microservice_Call_Graphs_Temporal_Windows_and_Scalability_Tradeoffs.pdf
└── Results/
    ├── Results final.pdf             # Partial evaluation results
    ├── StandaloneDiffusion.zip
    └── transformer.zip
```

## Models Implemented

### 1. **Graph Attention Network (GAT)**
- **File**: `Code/Models/GNN_Model.py`
- **Features**: Multi-head attention, edge-aware.

### 2. **Diffusion Model**
- **File**: `Code/Models/Standalone_Diffusion_Model.py`
- **Features**: Global diffusion-based information propagation.

### 3. **Transformer Model**
- **File**: `Code/Models/Standalone_Transformer_Model.py`
- **Features**: Self-attention, positional encoding.

### 4. **LSTM Model**
- **File**: `Code/LSTM/main.py`
- **Features**: Sequential modeling, temporal dependencies.

### 5. **Hybrid Models**
- **Diffusion + GAT**: `Diffusion_GAT_Model.py`
- **Transformer + GAT**: `Transformer_GAT_Model.py`
- **Features**: Combines multiple architectures' strengths.

### 6. **NodeSim** *(Third-party Integration)*
- **Directory**: `Code/NodeSim/`
- **Note**: Adapted from existing work for community-aware embeddings.
- **Use Case**: Node similarity enhancement.

## Datasets

The experiments use publicly available datasets:
- [Alibaba Microservices 2022](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022)
- [Alibaba Microservices 2021](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2021)
- [Huawei Cloud Trace](https://zenodo.org/record/5638238)

## Features

- **Temporal Processing**: Configurable time windows, dynamic graph handling.
- **Advanced Training**: Negative sampling, cross-validation, multi-seed experiments.
- **Evaluation Metrics**: Robust performance measurement with multiple criteria.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- TensorFlow 2.x
- NetworkX, NumPy, Pandas, Scikit-learn

### Setup
```bash
git clone <repository-url>
cd LinkPrediction-Extention-main

# PyTorch Geometric
pip install torch_geometric

# TensorFlow for LSTM
pip install tensorflow

# Other dependencies
pip install networkx numpy pandas scikit-learn matplotlib
```

## Usage

### Main Notebook
Run experiments via the provided notebook:

```bash
jupyter notebook Code/LinkPrediction.ipynb
```

### Model Selection
In the notebook, select your model:

```python
selected_model = "GAT"  # Options: "GAT", "Diffusion", "DiffusionGAT", "Transformer", "TransformerGAT", "LSTM"
```

### NodeSim Module Usage

```bash
cd Code/NodeSim/src
python main.py --input Input/sample.txt --output Output/sample.emb --dimensions 128
```

## Results

The provided results (`Results final.pdf`) are a subset demonstrating key findings and metrics, as full results are extensive. Additional detailed outcomes are compressed into zip files for specific models.

## Paper

The accompanying research paper offers a comprehensive analysis, including methodology, experiments, temporal windowing effects, and scalability trade-offs.

See: [Paper PDF](Paper/Link_Prediction_for_Microservice_Call_Graphs_Temporal_Windows_and_Scalability_Tradeoffs.pdf)

## Contributing

Contributions, suggestions, and improvements are welcome. Please fork the repository, create your branch, and submit a pull request.
