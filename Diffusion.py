import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve
)

# 1. Data Preparation

# Load call graph data and handle potential parsing errors
df = pd.read_csv('CallGraph_0.csv', on_bad_lines='skip')

# Retrieve all unique nodes from both 'um' and 'dm' columns
all_nodes = pd.concat([df['um'], df['dm']]).unique()

# Create a mapping from node identifiers to numeric indices
node_mapping = {node: idx for idx, node in enumerate(all_nodes)}

# Encode 'um' and 'dm' nodes using the created mapping
df['um_encoded'] = df['um'].map(node_mapping)
df['dm_encoded'] = df['dm'].map(node_mapping)

# Split the data into time windows of 100 ms each
time_window_size = 100  # Window size in milliseconds
max_timestamp = 10000  # Maximum timestamp to consider

def create_time_windows(dataframe, window_size, max_time):
    """Split the data into time windows of a specified size."""
    time_windows = []
    for start_time in range(0, max_time, window_size):
        window_df = dataframe[
            (dataframe['timestamp'] >= start_time) & (dataframe['timestamp'] < start_time + window_size)
        ]
        time_windows.append(window_df)
    return time_windows

time_windows = create_time_windows(df, time_window_size, max_timestamp)

# Define training and testing ranges based on timestamps
train_time_end = 7000  # Training until 7000 ms
test_time_start = 7000  # Testing starts from 7000 ms
test_time_end = 10000  # Testing ends at 10000 ms

# Separate time windows into training and testing sets
train_windows = [window for window in time_windows if window['timestamp'].max() < train_time_end]
test_windows = [window for window in time_windows if test_time_start <= window['timestamp'].min() < test_time_end]

# 2. Graph Construction

def create_graph(dataframe):
    """Construct a graph from a DataFrame of edges and timestamps."""
    um_values = np.array(dataframe['um_encoded'].values)
    dm_values = np.array(dataframe['dm_encoded'].values)

    # Create edge index from 'um' and 'dm' encoded values
    edge_index = torch.tensor(np.vstack([um_values, dm_values]), dtype=torch.long)

    # Use timestamp as edge attribute
    edge_attr = torch.tensor(dataframe['timestamp'].values, dtype=torch.float).unsqueeze(-1)

    return Data(edge_index=edge_index, edge_attr=edge_attr)

# Create graphs for all training and testing windows
train_graphs = [create_graph(window) for window in train_windows]
test_graphs = [create_graph(window) for window in test_windows]

# Add identity matrix as node features
num_nodes = len(all_nodes)  # Total number of unique nodes
for graph in train_graphs + test_graphs:
    graph.x = torch.eye(num_nodes)

# Transfer graphs to the available device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_graphs = [graph.to(device) for graph in train_graphs]
test_graphs = [graph.to(device) for graph in test_graphs]

# 3. Diffusion Model Definition

class DiffusionModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, diffusion_steps=10, alpha=0.5):
        super(DiffusionModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha

        # Linear transformation layers
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(out_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        adjacency = torch.zeros((x.size(0), x.size(0)), device=x.device)
        adjacency[edge_index[0], edge_index[1]] = 1

        diffusion_matrix = adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-6)
        h = self.lin1(x)

        for _ in range(self.diffusion_steps):
            h = self.alpha * torch.matmul(diffusion_matrix, h) + (1 - self.alpha) * h

        h = self.lin2(h)
        return h

# Initialize model parameters and create the Diffusion model
in_channels = num_nodes  # Number of input features (unique nodes)
out_channels = 16  # Dimension of the output features
model = DiffusionModel(in_channels=in_channels, out_channels=out_channels).to(device)

# Use Adam optimizer with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

# 4. Advanced Negative Sampling

def advanced_negative_sampling(edge_index, num_nodes, existing_edges, alpha=0.1):
    """Perform degree-based negative sampling, avoiding existing edges."""
    num_edges = edge_index.size(1)
    degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
    degree_prob = (degrees / degrees.sum()).pow(alpha)
    degree_prob = degree_prob / degree_prob.sum()  # Normalize probabilities

    neg_edges = []
    existing_set = set([tuple(edge) for edge in existing_edges.T.tolist()])

    for _ in range(num_edges):
        while True:
            src = torch.multinomial(degree_prob, 1).item()  # Sample source node
            dest = torch.randint(0, num_nodes, (1,)).item()  # Sample destination node
            if (src, dest) not in existing_set and (dest, src) not in existing_set:  # Ensure edge is unique
                neg_edges.append([src, dest])
                break

    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()  # Return as tensor

# 5. Training Function

def train(model, data):
    """Train the model on a single graph."""
    model.train()
    optimizer.zero_grad()
    out = model(data)

    # Compute positive edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1))  # Apply sigmoid to scores

    # Generate and predict negative edges
    neg_edge_index = advanced_negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1))  # Apply sigmoid to scores

    # Compute binary cross-entropy loss for both positive and negative edges
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss

    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    return loss.item()

# 6. Training Loop

for epoch in range(200):
    for i, graph in enumerate(train_graphs):
        loss = train(model, graph)
        print(f'Epoch {epoch}, Training Window {i}, Loss: {loss}')

# 7. Evaluation Function

def mean_reciprocal_rank(y_true, y_pred):
    """Compute Mean Reciprocal Rank (MRR) for binary classification."""
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order]
    ranks = np.where(y_true_sorted == 1)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0

def evaluate(model, data):
    """Evaluate the model on a single graph."""
    model.eval()
    out = model(data)

    # Compute positive and negative edge predictions
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1)).detach().cpu().numpy()

    neg_edge_index = advanced_negative_sampling(
        edge_index=data.edge_index, num_nodes=data.num_nodes, existing_edges=data.edge_index
    )
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1)).detach().cpu().numpy()

    # Concatenate predictions and ground truth labels
    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    y_pred = np.concatenate([pos_pred, neg_pred])

    # Calculate metrics
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, (y_pred > 0.5).astype(int), average='binary')
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    mrr = mean_reciprocal_rank(y_true, y_pred)

    return auc, precision, recall, f1, acc, mrr

# Evaluate the model on the test set
for i, graph in enumerate(test_graphs):
    auc, precision, recall, f1, acc, mrr = evaluate(model, graph)
    print(f'Test Window {i}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}, MRR: {mrr:.4f}')
