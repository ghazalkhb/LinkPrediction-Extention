"""
Recurring Link Activity Experiment (Corrected Protocol)
========================================================
Evaluates whether the model can generalize across load conditions by:
1. Identifying low-density and high-density windows in the test range
2. Training on the standard training horizon (G_0..G_59)
3. Testing on pairs that cross from low-density to high-density windows
   and vice versa, as well as same-density pairs.

Protocol:
- Degree-aware negative sampling (alpha=0.1)
- GAT model, seed-averaged over 3 seeds
- Alibaba 2022, 100ms windows
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch_geometric.data import Data

ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from GNN_Model import GAT  # noqa: E402

REVISION_DIR = Path(__file__).resolve().parent
if str(REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(REVISION_DIR))
from degree_aware_sampling import sample_negative_edges as _degree_sample  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_time_windows(df, window_size, max_time):
    windows = []
    for start in range(0, max_time, window_size):
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < start + window_size)]
        windows.append(w)
    return windows


def create_graph(window_df, num_nodes):
    if window_df.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            [window_df["um_encoded"].values, window_df["dm_encoded"].values], dtype=torch.long,
        )
        edge_attr = torch.tensor(window_df["timestamp"].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def edge_tensor_to_set(edge_index):
    if edge_index.numel() == 0:
        return set()
    return set((int(s), int(d)) for s, d in edge_index.t().tolist())


def sample_negative_edges(num_nodes, count, forbidden, seed, edge_index=None):
    return _degree_sample(num_nodes, count, forbidden, seed, edge_index=edge_index, alpha=0.1)


def attach_node_features(graphs, num_nodes, embedding_dim, device):
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for g in graphs:
        g.x = x
    return x


def pair_loss(model, graph_t, graph_t1, neg_seed):
    z = model(graph_t)
    pos_ei = graph_t1.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)
    pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
    neg_ei = sample_negative_edges(
        graph_t1.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()
    ).to(z.device)
    neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))
    loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
    loss = loss + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return loss


def evaluate_pair(model, graph_t, graph_t1, neg_seed):
    model.eval()
    with torch.no_grad():
        z = model(graph_t)
        pos_ei = graph_t1.edge_index
        if pos_ei.numel() == 0:
            return None
        pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
        neg_ei = sample_negative_edges(
            graph_t1.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()
        ).to(z.device)
        neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()
        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_pred = np.concatenate([pos_score, neg_score])
        y_hat = (y_pred > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
        return {
            "AUC": float(roc_auc_score(y_true, y_pred)),
            "PR_AUC": float(average_precision_score(y_true, y_pred)),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "Accuracy": float(accuracy_score(y_true, y_hat)),
            "PosEdges": int(pos_ei.size(1)),
        }


def train_model(model, optimizer, graphs, train_input, val_input, args, seed):
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idx in train_input:
            optimizer.zero_grad()
            loss = pair_loss(model, graphs[idx], graphs[idx + 1], seed * 100000 + epoch * 1000 + idx)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for idx in val_input:
                loss = pair_loss(model, graphs[idx], graphs[idx + 1], seed * 100000 + epoch * 1000 + 50000 + idx)
                val_losses.append(float(loss.item()))

        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | TrainLoss={np.mean(losses):.6f} | ValLoss={avg_val:.6f}")

        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    dataset_path = ROOT_DIR / args.dataset_path
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path, on_bad_lines="skip")
    for col in ["um", "dm", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df["um"] = df["um"].astype(str)
    df["dm"] = df["dm"].astype(str)
    all_nodes = pd.concat([df["um"], df["dm"]]).unique()
    node_mapping = {node: i for i, node in enumerate(all_nodes)}
    df["um_encoded"] = df["um"].map(node_mapping)
    df["dm_encoded"] = df["dm"].map(node_mapping)

    max_time = args.max_time if args.max_time else int(df["timestamp"].max()) + args.time_window_size
    windows = create_time_windows(df, args.time_window_size, max_time)
    num_nodes = len(all_nodes)
    graphs = [create_graph(w, num_nodes) for w in windows]

    node_features = attach_node_features(graphs, num_nodes, args.embedding_dim, device)
    graphs = [g.to(device) for g in graphs]
    print(f"Created {len(graphs)} windows, {num_nodes} nodes")

    # Compute edge density per window in the test range
    test_start = args.test_start_index
    test_end = len(graphs) - 1  # last window that can serve as input
    edge_counts = []
    for i in range(test_start, test_end + 1):
        ec = graphs[i].edge_index.size(1) if graphs[i].edge_index.numel() > 0 else 0
        edge_counts.append((i, ec))

    edge_counts_sorted = sorted(edge_counts, key=lambda x: x[1])
    median_edges = np.median([e for _, e in edge_counts])
    low_density = [(i, e) for i, e in edge_counts if e < median_edges and e > 0 and i + 1 <= test_end]
    high_density = [(i, e) for i, e in edge_counts if e >= median_edges and i + 1 <= test_end]

    print(f"Test range: G_{test_start}..G_{test_end}")
    print(f"Median edges/window: {median_edges:.0f}")
    print(f"Low-density windows: {len(low_density)}, High-density windows: {len(high_density)}")

    # Select representative windows
    # Pick bottom-quartile as "low" and top-quartile as "high"
    q25 = np.percentile([e for _, e in edge_counts if e > 0], 25)
    q75 = np.percentile([e for _, e in edge_counts if e > 0], 75)

    low_windows = [(i, e) for i, e in edge_counts if 0 < e <= q25 and i + 1 <= test_end]
    high_windows = [(i, e) for i, e in edge_counts if e >= q75 and i + 1 <= test_end]

    # Also need target windows to have edges
    low_windows = [(i, e) for i, e in low_windows
                   if graphs[i + 1].edge_index.numel() > 0]
    high_windows = [(i, e) for i, e in high_windows
                    if graphs[i + 1].edge_index.numel() > 0]

    # Sample up to N from each category
    n_sample = min(args.sample_per_category, len(low_windows), len(high_windows))
    if n_sample == 0:
        raise RuntimeError("Not enough valid windows for low/high density comparison.")

    rng = np.random.RandomState(42)
    low_sample = [low_windows[i] for i in rng.choice(len(low_windows), n_sample, replace=False)]
    high_sample = [high_windows[i] for i in rng.choice(len(high_windows), n_sample, replace=False)]

    print(f"Sampled {n_sample} low-density windows (edges <= {q25:.0f})")
    print(f"Sampled {n_sample} high-density windows (edges >= {q75:.0f})")

    train_input = list(range(0, args.val_start_index))
    val_input = list(range(args.val_start_index, args.train_end_index))
    seeds = list(range(args.num_seeds))

    all_per_pair = []
    scenario_results = []

    for scenario_name, sample_windows in [("low_density", low_sample), ("high_density", high_sample)]:
        test_pairs = [(i, i + 1) for i, _ in sample_windows]
        scenario_rows = []

        for seed in seeds:
            set_seed(seed)
            print(f"\nScenario={scenario_name}, Seed={seed}")

            model = GAT(args.embedding_dim, 16).to(device)
            node_features = attach_node_features(graphs, num_nodes, args.embedding_dim, device)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + [node_features],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            best_epoch = train_model(model, optimizer, graphs, train_input, val_input, args, seed)

            for inp_idx, tgt_idx in test_pairs:
                result = evaluate_pair(model, graphs[inp_idx], graphs[tgt_idx], seed * 200000 + inp_idx)
                if result is None:
                    continue
                result["InputGraph"] = inp_idx
                result["TargetGraph"] = tgt_idx
                result["InputEdges"] = int(graphs[inp_idx].edge_index.size(1)) if graphs[inp_idx].edge_index.numel() > 0 else 0
                result["TargetEdges"] = int(graphs[tgt_idx].edge_index.size(1))
                result["Scenario"] = scenario_name
                result["Seed"] = seed
                result["BestEpoch"] = best_epoch
                scenario_rows.append(result)
                all_per_pair.append(result)

        if scenario_rows:
            df_sc = pd.DataFrame(scenario_rows)
            scenario_results.append({
                "Scenario": scenario_name,
                "Pairs": len(scenario_rows),
                "Seeds": len(seeds),
                "AUC_Mean": float(df_sc["AUC"].mean()),
                "AUC_Std": float(df_sc["AUC"].std(ddof=0)),
                "PR_AUC_Mean": float(df_sc["PR_AUC"].mean()),
                "F1_Mean": float(df_sc["F1"].mean()),
                "F1_Std": float(df_sc["F1"].std(ddof=0)),
                "Precision_Mean": float(df_sc["Precision"].mean()),
                "Recall_Mean": float(df_sc["Recall"].mean()),
                "Accuracy_Mean": float(df_sc["Accuracy"].mean()),
                "AvgInputEdges": float(df_sc["InputEdges"].mean()),
                "AvgTargetEdges": float(df_sc["TargetEdges"].mean()),
            })
            print(f"  {scenario_name}: AUC={df_sc['AUC'].mean():.4f} F1={df_sc['F1'].mean():.4f}")

    # Save
    pd.DataFrame(all_per_pair).to_csv(results_dir / "recurring_per_pair_metrics.csv", index=False)
    pd.DataFrame(scenario_results).to_csv(results_dir / "recurring_summary.csv", index=False)
    (results_dir / "recurring_summary.json").write_text(
        json.dumps(scenario_results, indent=2), encoding="utf-8"
    )

    # Report
    lines = [
        "# Recurring Link Activity Experiment Report",
        "",
        "## Protocol",
        f"- Model: GAT",
        f"- Dataset: {args.dataset_path}",
        f"- Window size: {args.time_window_size}ms",
        f"- Training: G_0..G_{args.val_start_index-1} (60 pairs), Val: G_{args.val_start_index}..G_{args.train_end_index-1} (9 pairs)",
        f"- Negative sampling: degree-aware (alpha=0.1)",
        f"- Seeds: {args.num_seeds}",
        f"- Sampled {n_sample} windows per density category (Q25={q25:.0f}, Q75={q75:.0f} edges)",
        "",
        "## Results",
        "",
        "| Scenario | AUC | F1 | PR-AUC | Precision | Recall | Avg Input Edges | Avg Target Edges |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in scenario_results:
        lines.append(
            f"| {r['Scenario']} | {r['AUC_Mean']:.4f}±{r['AUC_Std']:.4f} "
            f"| {r['F1_Mean']:.4f}±{r['F1_Std']:.4f} | {r['PR_AUC_Mean']:.4f} "
            f"| {r['Precision_Mean']:.4f} | {r['Recall_Mean']:.4f} "
            f"| {r['AvgInputEdges']:.0f} | {r['AvgTargetEdges']:.0f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "This experiment tests whether the model can generalize across load conditions.",
        "Low-density windows have fewer edges (bottom quartile) while high-density",
        "windows have more (top quartile). Comparable AUC across conditions indicates",
        "robustness to load fluctuations; a large gap indicates load-dependent fragility.",
    ])

    (results_dir / "recurring_experiment_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Recurring Link Activity Experiment")
    parser.add_argument("--dataset-path", default="Data/Alibaba 2022/CallGraph_0.csv")
    parser.add_argument("--results-dir", default="revision/results")
    parser.add_argument("--time-window-size", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--train-end-index", type=int, default=69)
    parser.add_argument("--val-start-index", type=int, default=60)
    parser.add_argument("--test-start-index", type=int, default=70)
    parser.add_argument("--sample-per-category", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
