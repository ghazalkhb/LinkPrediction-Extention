import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from negative_sampling import advanced_negative_sampling

class TransformerOnly(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.randn(num_nodes, in_channels))
        encoder_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='relu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x + self.pos_embed.to(x.device)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        return x.squeeze(0)

def compute_loss(model, data, embedding_layer, num_nodes, device):
    data.x = embedding_layer(torch.arange(num_nodes, device=device))
    emb = model(data.x)
    src, dst = data.edge_index
    pos = torch.sigmoid((emb[src] * emb[dst]).sum(dim=1))
    neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg = torch.sigmoid((emb[neg_ei[0]] * emb[neg_ei[1]]).sum(dim=1))
    return F.binary_cross_entropy(pos, torch.ones_like(pos)) + \
           F.binary_cross_entropy(neg, torch.zeros_like(neg))

def train_step(model, data, optimizer, embedding_layer, num_nodes, device):
    model.train()
    optimizer.zero_grad()
    loss = compute_loss(model, data, embedding_layer, num_nodes, device)
    loss.backward()
    optimizer.step()
    return loss.item()

def val_step(model, data, embedding_layer, num_nodes, device):
    model.eval()
    with torch.no_grad():
        return compute_loss(model, data, embedding_layer, num_nodes, device).item()
