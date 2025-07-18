# Link Prediction for Microservice Call Graphs: Temporal Windows and Scalability Tradeoffs

This repository contains a comprehensive implementation of link prediction models for microservice call graphs, focusing on temporal dynamics and scalability considerations. The project implements multiple deep learning architectures and evaluates their performance on temporal network data.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Paper](#paper)
- [Contributing](#contributing)

## 🎯 Overview

This project addresses the challenge of predicting future connections in microservice call graphs using temporal network analysis. The implementation includes:

- **Multiple Model Architectures**: GAT, Diffusion, Transformer, LSTM, and hybrid models
- **Temporal Window Processing**: Time-based graph segmentation for dynamic analysis
- **Advanced Negative Sampling**: Sophisticated sampling strategies for training
- **Comprehensive Evaluation**: Multiple metrics including AUC, MRR, Precision, Recall, and F1-score
- **NodeSim Integration**: Community-aware node similarity for enhanced embeddings

## 📁 Project Structure

```
LinkPrediction-Extention-main/
├── Code/
│   ├── LinkPrediction.ipynb          # Main Jupyter notebook for experiments
│   ├── evaluate.py                   # Evaluation metrics and functions
│   ├── negative_sampling.py          # Advanced negative sampling strategies
│   ├── Models/
│   │   ├── GNN_Model.py             # Graph Attention Network (GAT) implementation
│   │   ├── Standalone_Diffusion_Model.py  # Diffusion-based model
│   │   ├── Diffusion_GAT_Model.py   # Hybrid Diffusion + GAT model
│   │   ├── Standalone_Transformer_Model.py  # Transformer-only model
│   │   └── Transformer_GAT_Model.py # Hybrid Transformer + GAT model
│   ├── LSTM/
│   │   └── main.py                  # LSTM-based link prediction
│   └── NodeSim/
│       ├── src/                     # NodeSim source code
│       ├── Input/                   # Input data files
│       ├── Output/                  # Generated embeddings
│       └── Adjusted/                # Modified NodeSim implementations
├── Paper/
│   └── Link_Prediction_for_Microservice_Call_Graphs_Temporal_Windows_and_Scalability_Tradeoffs.pdf
└── Results/
    ├── Results final.pdf            # Final evaluation results
    ├── StandaloneDiffusion.zip      # Diffusion model results
    └── transformer.zip              # Transformer model results
```

## 🤖 Models Implemented

### 1. **Graph Attention Network (GAT)**
- **File**: `Code/Models/GNN_Model.py`
- **Features**: Multi-head attention mechanism, edge-aware processing
- **Use Case**: Capturing complex node relationships in call graphs

### 2. **Diffusion Model**
- **File**: `Code/Models/Standalone_Diffusion_Model.py`
- **Features**: APPNP-based diffusion, global information propagation
- **Use Case**: Long-range dependency modeling

### 3. **Transformer Model**
- **File**: `Code/Models/Standalone_Transformer_Model.py`
- **Features**: Self-attention mechanism, positional embeddings
- **Use Case**: Sequence-aware link prediction

### 4. **LSTM Model**
- **File**: `Code/LSTM/main.py`
- **Features**: Sequential processing, temporal dependencies
- **Use Case**: Time-series based prediction

### 5. **Hybrid Models**
- **Diffusion + GAT**: `Code/Models/Diffusion_GAT_Model.py`
- **Transformer + GAT**: `Code/Models/Transformer_GAT_Model.py`
- **Features**: Combines benefits of multiple architectures

### 6. **NodeSim Integration**
- **Directory**: `Code/NodeSim/`
- **Features**: Community-aware node similarity, enhanced embeddings
- **Use Case**: Improved node representation learning

## ✨ Features

### Temporal Processing
- **Time Windows**: Configurable temporal segmentation
- **Dynamic Graphs**: Time-evolving network structures
- **Temporal Features**: Timestamp-aware edge attributes

### Advanced Training
- **Negative Sampling**: Sophisticated sampling strategies
- **Cross-Validation**: K-fold validation for robust evaluation
- **Early Stopping**: Patience-based training termination
- **Multi-Seed Experiments**: Reproducible results across seeds

### Evaluation Metrics
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **MRR**: Mean Reciprocal Rank
- **Precision/Recall/F1**: Classification performance metrics
- **Accuracy**: Overall prediction accuracy
- **Confusion Matrix**: Detailed error analysis

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- TensorFlow 2.x
- NetworkX
- NumPy, Pandas, Scikit-learn

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd LinkPrediction-Extention-main

# Install PyTorch Geometric
pip install torch_geometric

# Install TensorFlow for LSTM
pip install tensorflow

# Install other dependencies
pip install networkx numpy pandas scikit-learn matplotlib
```

## 📖 Usage

### Main Experiment Notebook
The primary interface is through the Jupyter notebook:

```python
# Open the main notebook
jupyter notebook Code/LinkPrediction.ipynb
```

### Model Selection
In the notebook, you can select different models:

```python
selected_model = "GAT"  # Options: "GAT", "Diffusion", "DiffusionGAT", 
                        # "Transformer", "TransformerGAT", "LSTM"
```

### Configuration Parameters
```python
# Temporal parameters
time_window_size = 100
train_time_end = 7000
test_time_start = 7000
test_time_end = 10000

# Model parameters
embedding_dim = 64
patience = 10
min_epochs = 30
```

### Running Experiments
```python
# The notebook automatically handles:
# 1. Data loading and preprocessing
# 2. Time window creation
# 3. Model training and validation
# 4. Performance evaluation
# 5. Results visualization
```

### NodeSim Usage
```bash
cd Code/NodeSim/src
python main.py --input Input/sample.txt --output Output/sample.emb --dimensions 128
```

## 📊 Results

The project includes comprehensive evaluation results:

- **Model Comparison**: Performance across different architectures
- **Temporal Analysis**: Impact of time windows on prediction accuracy
- **Scalability Study**: Tradeoffs between model complexity and performance
- **Community Effects**: NodeSim integration benefits

Key findings are documented in:
- `Paper/Link_Prediction_for_Microservice_Call_Graphs_Temporal_Windows_and_Scalability_Tradeoffs.pdf`
- `Results/Results final.pdf`

## 📄 Paper

The accompanying research paper provides detailed analysis of:
- Temporal window effects on link prediction
- Scalability tradeoffs in different model architectures
- Community-aware approaches for microservice call graphs
- Experimental methodology and results

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New model implementations
- Performance improvements
- Documentation enhancements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for the graph neural network framework
- NodeSim authors for the community-aware embedding approach
- Research community for foundational work in link prediction

---

**Note**: This project is designed for research purposes and microservice call graph analysis. For production use, additional validation and testing is recommended. 