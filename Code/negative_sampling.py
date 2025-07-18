import torch

def advanced_negative_sampling(edge_index, num_nodes, existing_edges, alpha=0.1):
    num_edges = edge_index.size(1)
    degrees = torch.bincount(edge_index.flatten(), minlength=num_nodes)
    prob = (degrees / degrees.sum()).pow(alpha)
    prob = prob / prob.sum()
    existing = set([tuple(e) for e in existing_edges.T.tolist()])

    neg = []
    for _ in range(num_edges):
        while True:
            s = torch.multinomial(prob, 1).item()
            d = torch.randint(0, num_nodes, (1,)).item()
            if (s, d) not in existing and (d, s) not in existing:
                neg.append([s, d])
                break
    return torch.tensor(neg, dtype=torch.long).t().contiguous()
