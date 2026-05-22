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
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data


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
from degree_aware_sampling import sample_negative_edges_fast as _degree_sample_fast  # noqa: E402


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
        um = window_df["um_encoded"].to_numpy(dtype=np.int64, copy=False)
        dm = window_df["dm_encoded"].to_numpy(dtype=np.int64, copy=False)
        edge_np = np.vstack([um, dm])
        edge_index = torch.from_numpy(edge_np).to(dtype=torch.long)
        edge_attr = torch.tensor(window_df["timestamp"].values, dtype=torch.float).unsqueeze(-1)
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


def edge_tensor_to_keys(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    if edge_index.numel() == 0:
        return np.empty((0,), dtype=np.int64)
    arr = edge_index.detach().cpu().numpy().astype(np.int64)
    src = arr[0]
    dst = arr[1]
    keys = src * num_nodes + dst
    rev_keys = dst * num_nodes + src
    return np.unique(np.concatenate([keys, rev_keys]))


def sample_negative_edges_fast(
    num_nodes: int,
    count: int,
    forbidden_keys: np.ndarray,
    seed: int,
    edge_index: torch.Tensor = None,
) -> torch.Tensor:
    """Degree-aware negative sampling (alpha=0.1)."""
    return _degree_sample_fast(num_nodes, count, forbidden_keys, seed, edge_index=edge_index, alpha=0.1)


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


def pair_train_loss(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    neg_seed: int,
) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, graph_t)
    pos_ei = graph_t1.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden_keys = edge_tensor_to_keys(pos_ei, graph_t1.num_nodes)
    neg_ei = sample_negative_edges_fast(graph_t1.num_nodes, pos_ei.size(1), forbidden_keys, neg_seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
    loss = loss + F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
    return loss


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    sorted_true = y_true[order]
    pos = np.where(sorted_true == 1)[0]
    if len(pos) == 0:
        return 0.0
    return 1.0 / float(pos[0] + 1)


def hits_at_k_window(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k_eval = int(max(1, min(k, len(y_score))))
    top_idx = np.argsort(y_score)[::-1][:k_eval]
    return 1.0 if np.any(y_true[top_idx] == 1) else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k_eval = int(max(1, min(k, len(y_score))))
    top_idx = np.argsort(y_score)[::-1][:k_eval]
    return float(np.sum(y_true[top_idx])) / float(k_eval)


def evaluate_pair_ranking(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    candidate_neg_ratio: int,
    seed: int,
) -> dict[str, float] | None:
    model.eval()
    with torch.no_grad():
        z = model_forward_embeddings(model, model_name, graph_t)
        pos_ei = graph_t1.edge_index
        if pos_ei.numel() == 0:
            return None

        pos_count = int(pos_ei.size(1))
        neg_count = int(pos_count * candidate_neg_ratio)

        pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden_keys = edge_tensor_to_keys(pos_ei, graph_t1.num_nodes)
        neg_ei = sample_negative_edges_fast(graph_t1.num_nodes, neg_count, forbidden_keys, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
        neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])

    return {
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "MRR": float(mrr_score(y_true, y_score)),
        "HitsAt10": float(hits_at_k_window(y_true, y_score, 10)),
        "HitsAt50": float(hits_at_k_window(y_true, y_score, 50)),
        "HitsAt100": float(hits_at_k_window(y_true, y_score, 100)),
        "PrecisionAt10": float(precision_at_k(y_true, y_score, 10)),
        "PrecisionAt50": float(precision_at_k(y_true, y_score, 50)),
        "PrecisionAt100": float(precision_at_k(y_true, y_score, 100)),
        "PosEdges": float(pos_count),
        "NegEdges": float(neg_count),
    }


def attach_node_features(
    graphs: list[Data], num_nodes: int, embedding_dim: int, device: torch.device
) -> torch.nn.Parameter:
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for g in graphs:
        g.x = x
    return x


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    metrics = [
        "PR_AUC",
        "MRR",
        "HitsAt10",
        "HitsAt50",
        "HitsAt100",
        "PrecisionAt10",
        "PrecisionAt50",
        "PrecisionAt100",
    ]
    out: dict[str, float] = {"Pairs": float(len(rows))}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        out[f"{m}_Mean"] = float(vals.mean())
        out[f"{m}_Std"] = float(vals.std(ddof=0))
    out["AvgPosEdges"] = float(np.mean([r["PosEdges"] for r in rows]))
    out["AvgNegEdges"] = float(np.mean([r["NegEdges"] for r in rows]))
    return out


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    summary: dict[str, float],
    best_epoch: int,
    train_time_sec: float,
) -> None:
    lines: list[str] = []
    lines.append("# Ranking-Based Evaluation Report (G_t -> G_{t+1})")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment evaluates ranking quality of future link prediction, which is more deployment-relevant "
        "than strict binary classification when candidate calls are imbalanced."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append(f"- Window size: {args.time_window_size}")
    lines.append("- Forecasting direction: G_t -> E_{t+1}")
    lines.append(f"- Training negatives: 1:1")
    lines.append(f"- Ranking candidate negative ratio at test: 1:{args.candidate_neg_ratio}")
    lines.append("- Candidate set per window = true future edges + sampled non-edges")
    lines.append(f"- Test start input: G_{args.test_start_index}")
    lines.append(f"- Best training epoch: {best_epoch}")
    lines.append(f"- Training time (sec): {train_time_sec:.2f}")
    lines.append("")
    lines.append("## 3. Ranking Metrics (Averaged Over Rolling Test Pairs)")
    lines.append(f"- Evaluated test pairs: {int(summary['Pairs'])}")
    lines.append(f"- Hits@10: {summary['HitsAt10_Mean']:.4f} +- {summary['HitsAt10_Std']:.4f}")
    lines.append(f"- Hits@50: {summary['HitsAt50_Mean']:.4f} +- {summary['HitsAt50_Std']:.4f}")
    lines.append(f"- Hits@100: {summary['HitsAt100_Mean']:.4f} +- {summary['HitsAt100_Std']:.4f}")
    lines.append(f"- Precision@10: {summary['PrecisionAt10_Mean']:.4f} +- {summary['PrecisionAt10_Std']:.4f}")
    lines.append(f"- Precision@50: {summary['PrecisionAt50_Mean']:.4f} +- {summary['PrecisionAt50_Std']:.4f}")
    lines.append(f"- Precision@100: {summary['PrecisionAt100_Mean']:.4f} +- {summary['PrecisionAt100_Std']:.4f}")
    lines.append(f"- MRR: {summary['MRR_Mean']:.4f} +- {summary['MRR_Std']:.4f}")
    lines.append(f"- PR-AUC: {summary['PR_AUC_Mean']:.4f} +- {summary['PR_AUC_Std']:.4f}")
    lines.append("")
    lines.append("## 4. Interpretation")
    lines.append(
        "These ranking results indicate whether top-ranked predicted links capture true future service calls, "
        "supporting operational triage value even when threshold-based F1 degrades under imbalanced test distributions."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

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
            f"Need at least {args.train_end_index + 2} windows, but only {len(graphs)} windows exist."
        )

    all_train_inputs = list(range(0, args.train_end_index))
    val_inputs = [i for i in all_train_inputs if i >= args.val_start_index]
    train_inputs = [i for i in all_train_inputs if i < args.val_start_index]
    if not train_inputs or not val_inputs:
        raise ValueError("Invalid train/val split. Adjust train_end_index and val_start_index.")

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
        tr_losses = []
        for idx in train_inputs:
            optimizer.zero_grad()
            loss = pair_train_loss(
                model,
                args.model,
                graphs[idx],
                graphs[idx + 1],
                neg_seed=args.seed * 100000 + epoch * 1000 + idx,
            )
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = []
            for idx in val_inputs:
                v_loss = pair_train_loss(
                    model,
                    args.model,
                    graphs[idx],
                    graphs[idx + 1],
                    neg_seed=args.seed * 100000 + epoch * 1000 + 50000 + idx,
                )
                val_losses.append(float(v_loss.item()))

        avg_train = float(np.mean(tr_losses))
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | TrainLoss={avg_train:.6f} | ValLoss={avg_val:.6f}")

        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time_sec = time.time() - train_start

    test_last_input = len(graphs) - 2
    test_inputs = list(range(args.test_start_index, test_last_input + 1))

    rows: list[dict[str, float]] = []
    for idx in test_inputs:
        row = evaluate_pair_ranking(
            model,
            args.model,
            graphs[idx],
            graphs[idx + 1],
            candidate_neg_ratio=args.candidate_neg_ratio,
            seed=args.seed * 200000 + idx,
        )
        if row is None:
            continue
        row["InputGraph"] = float(idx)
        row["TargetGraph"] = float(idx + 1)
        rows.append(row)

    if not rows:
        raise RuntimeError("No ranking rows were produced.")

    per_pair_df = pd.DataFrame(rows)
    per_pair_path = results_dir / "ranking_per_pair_metrics.csv"
    per_pair_df.to_csv(per_pair_path, index=False)

    summary = summarize_rows(rows)
    summary_df = pd.DataFrame([summary])
    summary_csv = results_dir / "ranking_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = {
        "config": {
            "model": args.model,
            "dataset_path": str(args.dataset_path),
            "time_window_size": args.time_window_size,
            "train_end_index": args.train_end_index,
            "val_start_index": args.val_start_index,
            "test_start_index": args.test_start_index,
            "candidate_neg_ratio": args.candidate_neg_ratio,
            "seed": args.seed,
        },
        "best_epoch": best_epoch,
        "training_time_sec": float(train_time_sec),
        "summary": summary,
    }
    summary_json_path = results_dir / "ranking_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    report_path = results_dir / "ranking_experiment_report.md"
    write_report(report_path, args, summary, best_epoch, train_time_sec)

    print("\n=== Final Ranking Summary ===")
    print(
        f"Hits@10={summary['HitsAt10_Mean']:.4f}, Hits@50={summary['HitsAt50_Mean']:.4f}, "
        f"Hits@100={summary['HitsAt100_Mean']:.4f}"
    )
    print(
        f"P@10={summary['PrecisionAt10_Mean']:.4f}, P@50={summary['PrecisionAt50_Mean']:.4f}, "
        f"P@100={summary['PrecisionAt100_Mean']:.4f}"
    )
    print(f"MRR={summary['MRR_Mean']:.4f}, PR-AUC={summary['PR_AUC_Mean']:.4f}")

    print(f"Saved: {per_pair_path}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ranking-based G_t -> G_{t+1} evaluation with Hits@K, Precision@K, MRR, and PR-AUC."
    )
    parser.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
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

    parser.add_argument("--candidate-neg-ratio", type=int, default=50)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
