import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    accuracy_score
)

# Load Data
df = pd.read_csv('CallGraph_0.csv', on_bad_lines='skip')
all_nodes = pd.concat([df['um'], df['dm']]).unique()
node_mapping = {node: idx for idx, node in enumerate(all_nodes)}
df['um_encoded'] = df['um'].map(node_mapping)
df['dm_encoded'] = df['dm'].map(node_mapping)

time_window_size = 100
max_timestamp = 10000

def create_time_windows(dataframe, window_size, max_time):
    time_windows = []
    for start_time in range(0, max_time, window_size):
        window_df = dataframe[
            (dataframe['timestamp'] >= start_time) & (dataframe['timestamp'] < start_time + window_size)
        ]
        time_windows.append(window_df)
    return time_windows

time_windows = create_time_windows(df, time_window_size, max_timestamp)
train_time_end = 7000
test_time_start = 7000
test_time_end = 10000
train_windows = [w for w in time_windows if w['timestamp'].max() < train_time_end]
test_windows = [w for w in time_windows if test_time_start <= w['timestamp'].min() < test_time_end]

def create_graph(dataframe):
    um_values = np.array(dataframe['um_encoded'].values)
    dm_values = np.array(dataframe['dm_encoded'].values)
    edge_index = torch.tensor(np.vstack([um_values, dm_values]), dtype=torch.long)
    edge_attr = torch.tensor(dataframe['timestamp'].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr)

train_graphs = [create_graph(window) for window in train_windows]
test_graphs = [create_graph(window) for window in test_windows]
num_nodes = len(all_nodes)
for graph in train_graphs + test_graphs:
    graph.x = torch.eye(num_nodes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_graphs = [graph.to(device) for graph in train_graphs]
test_graphs = [graph.to(device) for graph in test_graphs]

class TransformerModel(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim=16, num_heads=2, num_layers=2):
        super(TransformerModel, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)
        encoder_layers = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, data):
        x = self.embedding(torch.arange(data.x.shape[0], device=device))
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

model = TransformerModel(num_nodes=num_nodes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

def advanced_negative_sampling(edge_index, num_nodes, existing_edges, alpha=0.1):
    num_edges = edge_index.size(1)
    degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
    degree_prob = (degrees / degrees.sum()).pow(alpha)
    degree_prob = degree_prob / degree_prob.sum()
    neg_edges = []
    existing_set = set([tuple(edge) for edge in existing_edges.T.tolist()])
    for _ in range(num_edges):
        while True:
            src = torch.multinomial(degree_prob, 1).item()
            dest = torch.randint(0, num_nodes, (1,)).item()
            if (src, dest) not in existing_set and (dest, src) not in existing_set:
                neg_edges.append([src, dest])
                break
    return torch.tensor(neg_edges, dtype=torch.long).t().contiguous()

def train(model, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1))
    neg_edge_index = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1))
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss = pos_loss + neg_loss
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(200):
    for i, graph in enumerate(train_graphs):
        loss = train(model, graph)
        print(f'Epoch {epoch}, Training Window {i}, Loss: {loss}')

def evaluate(model, data):
    model.eval()
    out = model(data)
    src, dest = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dest]).sum(dim=1)).detach().cpu().numpy()
    neg_edge_index = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg_src, neg_dest = neg_edge_index
    neg_pred = torch.sigmoid((out[neg_src] * out[neg_dest]).sum(dim=1)).detach().cpu().numpy()
    y_true = np.concatenate([np.ones_like(pos_pred), np.zeros_like(neg_pred)])
    y_pred = np.concatenate([pos_pred, neg_pred])
    auc = roc_auc_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, (y_pred > 0.5).astype(int), average='binary')
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    return auc, precision, recall, f1, acc

for i, graph in enumerate(test_graphs):
    auc, precision, recall, f1, acc = evaluate(model, graph)
    print(f'Test Window {i}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}')
