"""
Training-Testing Gap Experiment (Corrected Protocol)
=====================================================
Evaluates how increasing the temporal gap between the training horizon and the
test window affects predictive quality.

Protocol:
- Train on G_0..G_59 -> E_1..E_60  (fixed, 60 pairs)
- Validate on G_60..G_68 -> E_61..E_69  (fixed, 9 pairs)
- For each gap g in [0, 5, 10, 20, 50, 100, 200, 500]:
    test_start = 70 + g
    evaluate on a sliding window of `eval_window` consecutive pairs starting
    at G_{test_start}.
- Degree-aware negative sampling (alpha=0.1)
- GAT model, seed-averaged over 3 seeds
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


def create_time_windows(df: pd.DataFrame, window_size: int, max_time: int) -> list[pd.DataFrame]:
    windows = []
    for start in range(0, max_time, window_size):
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < start + window_size)]
        windows.append(w)
    return windows


def create_graph(window_df: pd.DataFrame, num_nodes: int) -> Data:
    if window_df.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(
            [window_df["um_encoded"].values, window_df["dm_encoded"].values],
            dtype=torch.long,
        )
        edge_attr = torch.tensor(window_df["timestamp"].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def edge_tensor_to_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
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
        graph_t1.num_nodes, pos_ei.size(1), forbidden, neg_seed,
        edge_index=pos_ei.detach().cpu(),
    ).to(z.device)
    neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))
    loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
    loss = loss + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return loss


def evaluate_pairs(model, graphs, pair_indices, seed_offset):
    """Evaluate on a list of (input_idx, target_idx) pairs."""
    model.eval()
    rows = []
    with torch.no_grad():
        for inp_idx, tgt_idx in pair_indices:
            z = model(graphs[inp_idx])
            pos_ei = graphs[tgt_idx].edge_index
            if pos_ei.numel() == 0:
                continue
            pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
            forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
            neg_ei = sample_negative_edges(
                graphs[tgt_idx].num_nodes, pos_ei.size(1), forbidden,
                seed_offset + inp_idx, edge_index=pos_ei.detach().cpu(),
            ).to(z.device)
            neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()
            y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
            y_pred = np.concatenate([pos_score, neg_score])
            y_hat = (y_pred > 0.5).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
            rows.append({
                "InputGraph": int(inp_idx),
                "TargetGraph": int(tgt_idx),
                "AUC": float(roc_auc_score(y_true, y_pred)),
                "PR_AUC": float(average_precision_score(y_true, y_pred)),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1": float(f1),
                "Accuracy": float(accuracy_score(y_true, y_hat)),
                "PosEdges": int(pos_ei.size(1)),
            })
    return rows


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

    train_input = list(range(0, args.val_start_index))  # 0..59
    val_input = list(range(args.val_start_index, args.train_end_index))  # 60..68

    gaps = [int(g) for g in args.gaps.split(",")]
    eval_window = args.eval_window
    seeds = list(range(args.num_seeds))

    all_results = []

    for gap in gaps:
        test_start = args.test_start_index + gap
        test_end = test_start + eval_window
        # Need test_end as target, so need test_end index to exist
        if test_end >= len(graphs):
            print(f"Gap={gap}: test range G_{test_start}..G_{test_end-1} exceeds available windows ({len(graphs)}), skipping.")
            continue

        test_pairs = [(i, i + 1) for i in range(test_start, test_end)]
        gap_ms = gap * args.time_window_size  # gap in ms

        seed_rows = []
        for seed in seeds:
            set_seed(seed)
            print(f"\nGap={gap} ({gap_ms}ms), Seed={seed}: test G_{test_start}..G_{test_end-1} -> E_{test_start+1}..E_{test_end}")

            model = GAT(args.embedding_dim, 16).to(device)
            # Re-attach features (new embedding per seed)
            node_features = attach_node_features(graphs, num_nodes, args.embedding_dim, device)
            optimizer = torch.optim.Adam(
                list(model.parameters()) + [node_features],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_epoch = train_model(model, optimizer, graphs, train_input, val_input, args, seed)
            rows = evaluate_pairs(model, graphs, test_pairs, seed * 200000)
            if rows:
                for r in rows:
                    r["Gap"] = gap
                    r["GapMs"] = gap_ms
                    r["Seed"] = seed
                    r["BestEpoch"] = best_epoch
                seed_rows.extend(rows)

        if seed_rows:
            df_gap = pd.DataFrame(seed_rows)
            auc_mean = df_gap["AUC"].mean()
            pr_mean = df_gap["PR_AUC"].mean()
            f1_mean = df_gap["F1"].mean()
            prec_mean = df_gap["Precision"].mean()
            rec_mean = df_gap["Recall"].mean()
            acc_mean = df_gap["Accuracy"].mean()
            print(f"\n  Gap={gap} ({gap_ms}ms) summary: AUC={auc_mean:.4f} PR_AUC={pr_mean:.4f} F1={f1_mean:.4f}")

            all_results.append({
                "Gap": gap,
                "GapMs": gap_ms,
                "TestStart": test_start,
                "TestEnd": test_end - 1,
                "Pairs": len(seed_rows),
                "Seeds": len(seeds),
                "AUC_Mean": float(df_gap["AUC"].mean()),
                "AUC_Std": float(df_gap["AUC"].std(ddof=0)),
                "PR_AUC_Mean": float(df_gap["PR_AUC"].mean()),
                "PR_AUC_Std": float(df_gap["PR_AUC"].std(ddof=0)),
                "Precision_Mean": float(df_gap["Precision"].mean()),
                "Recall_Mean": float(df_gap["Recall"].mean()),
                "F1_Mean": float(df_gap["F1"].mean()),
                "F1_Std": float(df_gap["F1"].std(ddof=0)),
                "Accuracy_Mean": float(df_gap["Accuracy"].mean()),
            })

    # Save
    summary_df = pd.DataFrame(all_results)
    summary_csv = results_dir / "gap_experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = results_dir / "gap_experiment_summary.json"
    summary_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Report
    report_lines = [
        "# Training-Testing Gap Experiment Report",
        "",
        "## Protocol",
        f"- Model: GAT",
        f"- Dataset: {args.dataset_path}",
        f"- Window size: {args.time_window_size}ms",
        f"- Training: G_0..G_{args.val_start_index-1} -> E_1..E_{args.val_start_index} ({len(train_input)} pairs)",
        f"- Validation: G_{args.val_start_index}..G_{args.train_end_index-1} -> E_{args.val_start_index+1}..E_{args.train_end_index} ({len(val_input)} pairs)",
        f"- Gaps evaluated: {gaps}",
        f"- Eval window per gap: {eval_window} consecutive pairs",
        f"- Seeds: {args.num_seeds}",
        f"- Negative sampling: degree-aware (alpha=0.1)",
        "",
        "## Results",
        "",
    ]
    report_lines.append("| Gap (windows) | Gap (ms) | AUC | F1 | PR-AUC | Precision | Recall |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        report_lines.append(
            f"| {r['Gap']} | {r['GapMs']} | {r['AUC_Mean']:.4f}±{r['AUC_Std']:.4f} "
            f"| {r['F1_Mean']:.4f}±{r['F1_Std']:.4f} | {r['PR_AUC_Mean']:.4f}±{r['PR_AUC_Std']:.4f} "
            f"| {r['Precision_Mean']:.4f} | {r['Recall_Mean']:.4f} |"
        )
    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("As the gap increases, the model must predict further into the future using")
    report_lines.append("a fixed training horizon. Degradation reflects temporal drift in the call graph.")

    report_path = results_dir / "gap_experiment_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nResults saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Training-Testing Gap Experiment")
    parser.add_argument("--dataset-path", default="Data/Alibaba 2022/CallGraph_0.csv")
    parser.add_argument("--results-dir", default="revision/results")
    parser.add_argument("--time-window-size", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--train-end-index", type=int, default=69)
    parser.add_argument("--val-start-index", type=int, default=60)
    parser.add_argument("--test-start-index", type=int, default=70)
    parser.add_argument("--gaps", type=str, default="0,5,10,20,50,100,200,500")
    parser.add_argument("--eval-window", type=int, default=30, help="Number of consecutive test pairs per gap")
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
