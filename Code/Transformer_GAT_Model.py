import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import GATConv
from negative_sampling import advanced_negative_sampling

class TransformerGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, nhead=4, num_layers=2):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.randn(10000, in_channels))  # <-- make sure 10000 covers your num_nodes
        encoder_layer = TransformerEncoderLayer(
            d_model=in_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels,
            activation='relu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=2,
                             concat=True, edge_dim=1)
        self.conv2 = GATConv(hidden_channels * 2, hidden_channels,
                             heads=1, concat=False, edge_dim=1)
        self.attn1 = None

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        x = x + self.pos_embed[:x.size(0)].to(x.device)  # clip to actual node size
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x, (ei1, attn1) = self.conv1(x, ei, ea, return_attention_weights=True)
        x = F.elu(x)
        x, _ = self.conv2(x, ei, ea, return_attention_weights=True)
        self.attn1 = attn1
        return x

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    src, dst = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
    neg_ei = advanced_negative_sampling(data.edge_index,
                                        data.num_nodes,
                                        data.edge_index)
    neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))
    loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()
    return loss.item()

def compute_loss(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        src, dst = data.edge_index
        pos_pred = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
        neg_ei = advanced_negative_sampling(data.edge_index,
                                            data.num_nodes,
                                            data.edge_index)
        neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))
        loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
               F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return loss.item()
