from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

def mean_reciprocal_rank(y_true, y_pred):
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order]
    ranks = np.where(y_sorted == 1)[0]
    return 1.0 / (ranks[0] + 1) if len(ranks) > 0 else 0.0

def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        src, dst = data.edge_index
        pp = torch.sigmoid((out[src] * out[dst]).sum(dim=1)).cpu().numpy()
        neg_ei = advanced_negative_sampling(data.edge_index, data.num_nodes, data.edge_index)
        npred = torch.sigmoid((out[neg_ei[0]] * out[neg_ei[1]]).sum(dim=1)).cpu().numpy()
        y_true = np.concatenate([np.ones_like(pp), np.zeros_like(npred)])
        y_pred = np.concatenate([pp, npred])
        auc = roc_auc_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, (y_pred > 0.5).astype(int), average='binary')
        acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
        mrr = mean_reciprocal_rank(y_true, y_pred)
        return auc, precision, recall, f1, acc, mrr
