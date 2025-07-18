import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ===== TransformerOnly Model =====
class TransformerOnly(nn.Module):
    def __init__(self, num_nodes, in_channels, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(num_nodes, in_channels))
        encoder_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x + self.pos_embed[:x.size(0)].to(x.device)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        return x.squeeze(0)

# ===== Train Step =====
def train_transformer(model, data, optimizer, embedding_layer, num_nodes, device):
    model.train()
    optimizer.zero_grad()
    x = embedding_layer(torch.arange(num_nodes, device=device))
    data.x = x.detach()
    out = model(data.x)

    src, dst = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dst]).sum(dim=1))

    neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))

    loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()
    return loss.item()

# ===== Validation Step =====
def val_transformer(model, data, embedding_layer, num_nodes, device):
    model.eval()
    with torch.no_grad():
        x = embedding_layer(torch.arange(num_nodes, device=device))
        data.x = x.detach()
        out = model(data.x)

        src, dst = data.edge_index
        pos_pred = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
        neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
        neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))

        loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
               F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return loss.item()
