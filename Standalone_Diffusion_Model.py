import torch
import torch.nn.functional as F
from torch_geometric.nn import APPNP
from negative_sampling import advanced_negative_sampling

class Diffusion(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, K=10, alpha=0.1):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.appnp = APPNP(K, alpha)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.lin1(x))
        x = self.appnp(x, ei)
        return x

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    src, dst = data.edge_index
    pos_pred = torch.sigmoid((out[src] * out[dst]).sum(dim=1))
    neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
    neg_pred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1))
    loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
           F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    loss.backward()
    optimizer.step()
    return loss.item()
