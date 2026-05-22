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
    both = np.concatenate([keys, rev_keys])
    return np.unique(both)


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


def evaluate_pair_with_ratio(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    neg_ratio: int,
    seed: int,
) -> dict[str, float] | None:
    model.eval()
    with torch.no_grad():
        z = model_forward_embeddings(model, model_name, graph_t)
        pos_ei = graph_t1.edge_index
        if pos_ei.numel() == 0:
            return None

        pos_count = int(pos_ei.size(1))
        neg_count = int(pos_count * neg_ratio)

        pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden_keys = edge_tensor_to_keys(pos_ei, graph_t1.num_nodes)
        neg_ei = sample_negative_edges_fast(graph_t1.num_nodes, neg_count, forbidden_keys, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
        neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_pred = np.concatenate([pos_scores, neg_scores])
        y_hat = (y_pred > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_hat,
            average="binary",
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_hat)

        return {
            "AUC": float(roc_auc_score(y_true, y_pred)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Accuracy": float(acc),
            "PosEdges": float(pos_count),
            "NegEdges": float(neg_count),
        }


def evaluate_pair_all_nonedges(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_t1: Data,
    max_nodes: int,
    max_negatives: int,
) -> dict[str, float] | None:
    if graph_t1.num_nodes > max_nodes:
        return None

    pos_ei = graph_t1.edge_index
    if pos_ei.numel() == 0:
        return None

    n = int(graph_t1.num_nodes)
    total_directed_no_self = n * (n - 1)
    forbidden_keys = edge_tensor_to_keys(pos_ei, n)
    estimated_nonedges = total_directed_no_self - int(forbidden_keys.size // 2)
    if estimated_nonedges > max_negatives:
        return None

    model.eval()
    with torch.no_grad():
        z = model_forward_embeddings(model, model_name, graph_t)

        pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()

        src = np.repeat(np.arange(n, dtype=np.int64), n)
        dst = np.tile(np.arange(n, dtype=np.int64), n)
        mask = src != dst
        src = src[mask]
        dst = dst[mask]
        keys = src * n + dst
        keep = ~np.isin(keys, forbidden_keys, assume_unique=False)
        src = src[keep]
        dst = dst[keep]

        neg_ei = torch.from_numpy(np.vstack([src, dst])).to(dtype=torch.long, device=z.device)
        neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_pred = np.concatenate([pos_scores, neg_scores])
        y_hat = (y_pred > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_hat,
            average="binary",
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_hat)

        return {
            "AUC": float(roc_auc_score(y_true, y_pred)),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Accuracy": float(acc),
            "PosEdges": float(pos_scores.shape[0]),
            "NegEdges": float(neg_scores.shape[0]),
        }


def attach_node_features(
    graphs: list[Data], num_nodes: int, embedding_dim: int, device: torch.device
) -> torch.nn.Parameter:
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for g in graphs:
        g.x = x
    return x


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    metrics = ["AUC", "Precision", "Recall", "F1", "Accuracy"]
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
    summary_df: pd.DataFrame,
    best_epoch: int,
    train_time_sec: float,
    all_nonedge_note: str,
) -> None:
    lines: list[str] = []
    lines.append("# Imbalanced Negative Test Evaluation Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment re-evaluates forecasting under increasingly imbalanced test negatives to measure how "
        "balanced 1:1 testing may inflate F1, precision, and accuracy."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append(f"- Window size: {args.time_window_size}")
    lines.append("- Training negatives: 1:1 (kept unchanged)")
    lines.append("- Test forecasting direction: G_t -> E_{t+1}")
    lines.append(f"- Test negative ratios: {args.eval_ratios}")
    lines.append(f"- Test start input: G_{args.test_start_index}")
    lines.append(f"- Best training epoch: {best_epoch}")
    lines.append(f"- Training time (sec): {train_time_sec:.2f}")
    lines.append("")
    lines.append("## 3. Main Results By Test Imbalance")
    for _, row in summary_df.iterrows():
        ratio = int(row["NegRatio"])
        lines.append(f"### 1 positive : {ratio} negatives")
        lines.append(f"- Evaluated pairs: {int(row['Pairs'])}")
        lines.append(f"- AUC: {row['AUC_Mean']:.4f} +- {row['AUC_Std']:.4f}")
        lines.append(f"- Precision: {row['Precision_Mean']:.4f} +- {row['Precision_Std']:.4f}")
        lines.append(f"- Recall: {row['Recall_Mean']:.4f} +- {row['Recall_Std']:.4f}")
        lines.append(f"- F1: {row['F1_Mean']:.4f} +- {row['F1_Std']:.4f}")
        lines.append(f"- Accuracy: {row['Accuracy_Mean']:.4f} +- {row['Accuracy_Std']:.4f}")
        lines.append(f"- Avg positive edges/window: {row['AvgPosEdges']:.1f}")
        lines.append(f"- Avg negative edges/window: {row['AvgNegEdges']:.1f}")
        lines.append("")

    lines.append("## 4. All-Non-Edges Evaluation Feasibility")
    lines.append(all_nonedge_note)
    lines.append("")
    lines.append("## 5. Interpretation")
    lines.append(
        "As the test negative ratio increases, precision and F1 are expected to decrease while recall may remain "
        "less affected, directly exposing optimistic bias from balanced 1:1 test distributions."
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
    test_inputs_all = list(range(args.test_start_index, test_last_input + 1))

    ratios = [int(x.strip()) for x in args.eval_ratios.split(",") if x.strip()]
    per_pair_rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for ratio in ratios:
        if ratio == 50 and args.ratio50_max_pairs > 0:
            test_inputs = test_inputs_all[: args.ratio50_max_pairs]
        else:
            test_inputs = test_inputs_all

        rows_ratio: list[dict[str, float]] = []
        print(f"\n=== Evaluating ratio 1:{ratio} on {len(test_inputs)} test pairs ===")
        for idx in test_inputs:
            row = evaluate_pair_with_ratio(
                model,
                args.model,
                graphs[idx],
                graphs[idx + 1],
                neg_ratio=ratio,
                seed=args.seed * 200000 + ratio * 10000 + idx,
            )
            if row is None:
                continue
            row["NegRatio"] = float(ratio)
            row["InputGraph"] = float(idx)
            row["TargetGraph"] = float(idx + 1)
            rows_ratio.append(row)
            per_pair_rows.append(row)

        if not rows_ratio:
            continue

        summary = summarize_rows(rows_ratio)
        summary["NegRatio"] = float(ratio)
        summary_rows.append(summary)

        print(
            f"1:{ratio} | AUC={summary['AUC_Mean']:.4f} | "
            f"P={summary['Precision_Mean']:.4f} | R={summary['Recall_Mean']:.4f} | "
            f"F1={summary['F1_Mean']:.4f} | Acc={summary['Accuracy_Mean']:.4f}"
        )

    if not summary_rows:
        raise RuntimeError("No evaluation rows were produced.")

    per_pair_df = pd.DataFrame(per_pair_rows)
    per_pair_path = results_dir / "imbalanced_per_pair_metrics.csv"
    per_pair_df.to_csv(per_pair_path, index=False)

    summary_df = pd.DataFrame(summary_rows).sort_values("NegRatio")
    summary_csv_path = results_dir / "imbalanced_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    all_nonedge_result = evaluate_pair_all_nonedges(
        model,
        args.model,
        graphs[test_inputs_all[0]],
        graphs[test_inputs_all[0] + 1],
        max_nodes=args.all_nonedges_max_nodes,
        max_negatives=args.all_nonedges_max_negatives,
    )

    if all_nonedge_result is None:
        all_nonedge_note = (
            "All-non-edges testing is not computationally feasible under current dataset scale or configured limits. "
            f"Feasibility thresholds: max_nodes={args.all_nonedges_max_nodes}, "
            f"max_negatives={args.all_nonedges_max_negatives}."
        )
    else:
        all_nonedge_note = (
            "All-non-edges test was feasible for a sample pair and was evaluated with: "
            f"AUC={all_nonedge_result['AUC']:.4f}, Precision={all_nonedge_result['Precision']:.4f}, "
            f"Recall={all_nonedge_result['Recall']:.4f}, F1={all_nonedge_result['F1']:.4f}, "
            f"Accuracy={all_nonedge_result['Accuracy']:.4f}."
        )

    summary_json_path = results_dir / "imbalanced_summary.json"
    payload = {
        "config": {
            "model": args.model,
            "dataset_path": str(args.dataset_path),
            "time_window_size": args.time_window_size,
            "train_end_index": args.train_end_index,
            "val_start_index": args.val_start_index,
            "test_start_index": args.test_start_index,
            "eval_ratios": ratios,
            "ratio50_max_pairs": args.ratio50_max_pairs,
        },
        "best_epoch": best_epoch,
        "training_time_sec": float(train_time_sec),
        "summary": summary_df.to_dict(orient="records"),
        "all_nonedge_note": all_nonedge_note,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = results_dir / "imbalanced_experiment_report.md"
    write_report(
        report_path=report_path,
        args=args,
        summary_df=summary_df,
        best_epoch=best_epoch,
        train_time_sec=float(train_time_sec),
        all_nonedge_note=all_nonedge_note,
    )

    print("\n=== Final Summary ===")
    for _, r in summary_df.iterrows():
        ratio = int(r["NegRatio"])
        print(
            f"1:{ratio} | AUC={r['AUC_Mean']:.4f} | Precision={r['Precision_Mean']:.4f} | "
            f"Recall={r['Recall_Mean']:.4f} | F1={r['F1_Mean']:.4f} | Accuracy={r['Accuracy_Mean']:.4f}"
        )

    print(f"Saved: {per_pair_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="G_t -> G_{t+1} evaluation with imbalanced test negative sampling ratios."
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

    parser.add_argument("--eval-ratios", type=str, default="1,5,10,50")
    parser.add_argument("--ratio50-max-pairs", type=int, default=300)

    parser.add_argument("--all-nonedges-max-nodes", type=int, default=3000)
    parser.add_argument("--all-nonedges-max-negatives", type=int, default=2000000)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run(args)
