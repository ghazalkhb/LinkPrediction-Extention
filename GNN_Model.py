import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from negative_sampling import advanced_negative_sampling

class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=2):
        super().__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads,
                             concat=True, edge_dim=1)
        self.conv2 = GATConv(out_channels * heads, out_channels,
                             heads=1, concat=False, edge_dim=1)
        self.attn1 = None

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
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
    neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))
    loss = (F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) +
            F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred)))
    loss.backward()
    optimizer.step()
    return loss.item()
