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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.data import Data


# Import project models without modifying original project files.
ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from GNN_Model import GAT  # noqa: E402
from Standalone_Diffusion_Model import Diffusion  # noqa: E402
from Diffusion_GAT_Model import DiffusionGAT  # noqa: E402
from Standalone_Transformer_Model import TransformerOnly  # noqa: E402
from Transformer_GAT_Model import TransformerGAT  # noqa: E402

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


def mean_reciprocal_rank(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    hit_positions = np.where(y_sorted == 1)[0]
    if len(hit_positions) == 0:
        return 0.0
    return 1.0 / float(hit_positions[0] + 1)


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

    g = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
    return g


def edge_tensor_to_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    if edge_index.numel() == 0:
        return set()
    return set((int(s), int(d)) for s, d in edge_index.t().tolist())


def sample_negative_edges(
    num_nodes: int,
    count: int,
    forbidden: set[tuple[int, int]],
    seed: int,
    edge_index: torch.Tensor = None,
) -> torch.Tensor:
    """Degree-aware negative sampling (alpha=0.1)."""
    return _degree_sample(num_nodes, count, forbidden, seed, edge_index=edge_index, alpha=0.1)


def get_model(model_name: str, embedding_dim: int, num_nodes: int, device: torch.device) -> torch.nn.Module:
    name = model_name.lower()
    if name == "gat":
        model = GAT(embedding_dim, 16)
    elif name == "diffusion":
        model = Diffusion(embedding_dim, 16)
    elif name == "diffusiongat":
        model = DiffusionGAT(embedding_dim, 16)
    elif name == "transformer":
        model = TransformerOnly(num_nodes, embedding_dim)
    elif name == "transformergat":
        model = TransformerGAT(embedding_dim, 16)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose from: "
            "GAT, Diffusion, DiffusionGAT, Transformer, TransformerGAT"
        )
    return model.to(device)


def model_forward_embeddings(model: torch.nn.Module, model_name: str, graph_t: Data) -> torch.Tensor:
    if model_name.lower() == "transformer":
        return model(graph_t.x)
    return model(graph_t)


def pair_loss(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    neg_seed: int,
) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, graph_t)

    pos_ei = graph_t1.edge_index
    if pos_ei.numel() == 0:
        # Keep training numerically stable if a target window has no links.
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))

    forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
    neg_ei = sample_negative_edges(graph_t1.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
    loss = loss + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return loss


def evaluate_pair(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    neg_seed: int,
) -> dict[str, float] | None:
    model.eval()
    with torch.no_grad():
        z = model_forward_embeddings(model, model_name, graph_t)
        pos_ei = graph_t1.edge_index
        if pos_ei.numel() == 0:
            return None

        pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
        neg_ei = sample_negative_edges(graph_t1.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()).to(z.device)
        neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_pred = np.concatenate([pos_score, neg_score])
        y_hat = (y_pred > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_hat,
            average="binary",
            zero_division=0,
        )

        return {
            "AUC": float(roc_auc_score(y_true, y_pred)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Accuracy": float(accuracy_score(y_true, y_hat)),
            "MRR": float(mean_reciprocal_rank(y_true, y_pred)),
            "PosEdges": int(pos_ei.size(1)),
            "NegEdges": int(neg_ei.size(1)),
        }


def attach_node_features(
    graphs: list[Data], num_nodes: int, embedding_dim: int, device: torch.device
) -> torch.nn.Parameter:
    # Shared trainable node features across all windows.
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for g in graphs:
        g.x = x
    return x


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_names = ["AUC", "Precision", "Recall", "F1", "Accuracy", "MRR"]
    out: dict[str, dict[str, float]] = {}
    for name in metric_names:
        vals = np.array([r[name] for r in rows], dtype=float)
        out[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def write_report(
    report_path: Path,
    summary_path: Path,
    per_pair_csv: Path,
    config: dict,
    train_pair_count: int,
    val_pair_count: int,
    evaluated_rows: list[dict[str, float]],
    training_time_sec: float,
    eval_time_sec: float,
) -> None:
    agg = aggregate_metrics(evaluated_rows)

    lines = []
    lines.append("# Rolling Forecasting Evaluation Report (G_t -> G_{t+1})")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment evaluates temporal link forecasting where each prediction uses graph G_t as input and "
        "scores links that appear in G_{t+1}."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Model: {config['model']}")
    lines.append(f"- Dataset: {config['dataset_path']}")
    lines.append(f"- Window size: {config['time_window_size']}")
    lines.append(f"- Fixed training windows: G_0 ... G_{config['train_end_index']}")
    lines.append(
        f"- Validation windows (inside training horizon): G_{config['val_start_index']} ... G_{config['train_end_index'] - 1}"
    )
    lines.append(f"- Test windows: G_{config['test_start_index']} ... G_{config['last_test_input_index']}")
    lines.append("- Test target at each step: edges from G_{t+1}")
    lines.append("- Negatives at each step: sampled node pairs not present in G_{t+1}")
    lines.append("")
    lines.append("## 3. Run Summary")
    lines.append(f"- Training pairs used: {train_pair_count}")
    lines.append(f"- Validation pairs used: {val_pair_count}")
    lines.append(f"- Test pairs evaluated: {len(evaluated_rows)}")
    lines.append(f"- Training time (sec): {training_time_sec:.2f}")
    lines.append(f"- Evaluation time (sec): {eval_time_sec:.2f}")
    lines.append("")
    lines.append("## 4. Primary Forecasting Results (Average Over Rolling Pairs)")
    for metric in ["AUC", "Precision", "Recall", "F1", "Accuracy", "MRR"]:
        stat = agg[metric]
        lines.append(
            f"- {metric}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, "
            f"min={stat['min']:.4f}, max={stat['max']:.4f}"
        )
    lines.append("")
    lines.append("## 5. Saved Artifacts")
    lines.append(f"- Per-pair rolling metrics: {per_pair_csv}")
    lines.append(f"- JSON summary: {summary_path}")
    lines.append("")
    lines.append("## 6. Interpretation")
    lines.append(
        "These values are the primary forecasting result because they strictly follow the temporal direction G_t -> G_{t+1} "
        "and avoid evaluating links inside the same test graph window."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    start_total = time.time()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Provide --dataset-path to a valid CSV file."
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path, on_bad_lines="skip")
    for col in ["um", "dm", "timestamp"]:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain column '{col}'.")

    df["um"] = df["um"].astype(str)
    df["dm"] = df["dm"].astype(str)
    all_nodes = pd.concat([df["um"], df["dm"]]).unique()
    node_mapping = {node: i for i, node in enumerate(all_nodes)}
    df["um_encoded"] = df["um"].map(node_mapping)
    df["dm_encoded"] = df["dm"].map(node_mapping)

    if args.max_time is None:
        max_time = int(df["timestamp"].max()) + args.time_window_size
    else:
        max_time = args.max_time

    windows = create_time_windows(df, args.time_window_size, max_time)
    num_nodes = len(all_nodes)
    graphs = [create_graph(w, num_nodes) for w in windows]

    node_features = attach_node_features(graphs, num_nodes, args.embedding_dim, device)
    graphs = [g.to(device) for g in graphs]

    if len(graphs) <= args.train_end_index + 1:
        raise ValueError(
            f"Need at least {args.train_end_index + 2} windows, but only {len(graphs)} were created."
        )
    if args.test_start_index >= len(graphs) - 1:
        raise ValueError(
            "test_start_index must be <= number_of_windows - 2 to allow G_t -> G_{t+1} evaluation."
        )

    all_train_input = list(range(0, args.train_end_index))  # 0..68 when train_end_index=69
    val_input = [i for i in all_train_input if i >= args.val_start_index]
    train_input = [i for i in all_train_input if i < args.val_start_index]

    if len(train_input) == 0:
        raise ValueError("No training pairs left after validation split. Lower val_start_index.")
    if len(val_input) == 0:
        raise ValueError("No validation pairs selected. Increase train_end_index or lower val_start_index.")

    test_last_input = len(graphs) - 2
    test_input = list(range(args.test_start_index, test_last_input + 1))

    model = get_model(args.model, args.embedding_dim, num_nodes, device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [node_features],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for idx in train_input:
            optimizer.zero_grad()
            loss = pair_loss(
                model,
                args.model,
                graphs[idx],
                graphs[idx + 1],
                neg_seed=args.seed * 100000 + epoch * 1000 + idx,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for idx in val_input:
                loss = pair_loss(
                    model,
                    args.model,
                    graphs[idx],
                    graphs[idx + 1],
                    neg_seed=args.seed * 100000 + epoch * 1000 + 50000 + idx,
                )
                val_losses.append(float(loss.item()))

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        print(f"Epoch {epoch:03d} | TrainLoss={avg_train:.6f} | ValLoss={avg_val:.6f}")

        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time_sec = time.time() - train_start

    eval_start = time.time()
    per_pair_rows: list[dict[str, float]] = []
    for idx in test_input:
        row = evaluate_pair(
            model,
            args.model,
            graphs[idx],
            graphs[idx + 1],
            neg_seed=args.seed * 200000 + idx,
        )
        if row is None:
            print(f"Skipping G_{idx} -> E_{idx + 1} because E_{idx + 1} is empty.")
            continue

        row["InputGraph"] = int(idx)
        row["TargetGraph"] = int(idx + 1)
        per_pair_rows.append(row)
        print(
            f"G_{idx} -> E_{idx + 1} | AUC={row['AUC']:.4f} P={row['Precision']:.4f} "
            f"R={row['Recall']:.4f} F1={row['F1']:.4f} Acc={row['Accuracy']:.4f} MRR={row['MRR']:.4f}"
        )

    eval_time_sec = time.time() - eval_start

    if not per_pair_rows:
        raise RuntimeError("No test rolling pairs were evaluated. Check data coverage and test_start_index.")

    per_pair_df = pd.DataFrame(per_pair_rows)
    per_pair_csv = results_dir / "rolling_per_pair_metrics.csv"
    per_pair_df.to_csv(per_pair_csv, index=False)

    agg = aggregate_metrics(per_pair_rows)
    summary = {
        "config": {
            "model": args.model,
            "dataset_path": str(dataset_path),
            "time_window_size": args.time_window_size,
            "embedding_dim": args.embedding_dim,
            "train_end_index": args.train_end_index,
            "val_start_index": args.val_start_index,
            "test_start_index": args.test_start_index,
            "last_test_input_index": int(test_input[-1]),
            "epochs": args.epochs,
            "seed": args.seed,
        },
        "counts": {
            "num_windows": len(graphs),
            "num_nodes": num_nodes,
            "train_pairs": len(train_input),
            "val_pairs": len(val_input),
            "test_pairs_evaluated": len(per_pair_rows),
        },
        "timing_sec": {
            "training": round(train_time_sec, 4),
            "evaluation": round(eval_time_sec, 4),
            "total": round(time.time() - start_total, 4),
        },
        "aggregate_metrics": agg,
    }

    summary_path = results_dir / "rolling_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = results_dir / "rolling_experiment_report.md"
    write_report(
        report_path=report_path,
        summary_path=summary_path,
        per_pair_csv=per_pair_csv,
        config=summary["config"],
        train_pair_count=len(train_input),
        val_pair_count=len(val_input),
        evaluated_rows=per_pair_rows,
        training_time_sec=train_time_sec,
        eval_time_sec=eval_time_sec,
    )

    print("\nPrimary forecasting result (mean over rolling test pairs):")
    for metric in ["AUC", "Precision", "Recall", "F1", "Accuracy", "MRR"]:
        print(f"  {metric}: {agg[metric]['mean']:.4f}")
    print(f"\nSaved: {per_pair_csv}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rolling evaluation with strict G_t -> G_{t+1} protocol for temporal link forecasting."
    )
    parser.add_argument("--dataset-path", type=str, default="CallGraph_0.csv")
    parser.add_argument("--results-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--model", type=str, default="GAT")
    parser.add_argument("--time-window-size", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)

    parser.add_argument("--train-end-index", type=int, default=69)
    parser.add_argument("--val-start-index", type=int, default=60)
    parser.add_argument("--test-start-index", type=int, default=70)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
