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
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
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
    graph_target: Data,
    neg_seed: int,
) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, graph_t)

    pos_ei = graph_target.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
    neg_ei = sample_negative_edges(graph_target.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))
    loss = loss + F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
    return loss


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, top_k: int) -> tuple[float, float]:
    if len(y_score) == 0:
        return 0.0, 0.0
    k_eval = int(max(1, min(top_k, len(y_score))))
    idx = np.argsort(y_score)[::-1][:k_eval]
    tp = float(y_true[idx].sum())
    total_pos = float(y_true.sum())
    p_at_k = tp / float(k_eval)
    r_at_k = tp / total_pos if total_pos > 0 else 0.0
    return p_at_k, r_at_k


def evaluate_pair(
    model: torch.nn.Module,
    model_name: str,
    graph_t: Data,
    graph_target: Data,
    neg_seed: int,
    top_k: int,
) -> dict[str, float] | None:
    model.eval()
    with torch.no_grad():
        tic = time.perf_counter()
        z = model_forward_embeddings(model, model_name, graph_t)

        pos_ei = graph_target.edge_index
        if pos_ei.numel() == 0:
            return None

        pos_score = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden = edge_tensor_to_set(pos_ei.detach().cpu())
        neg_ei = sample_negative_edges(graph_target.num_nodes, pos_ei.size(1), forbidden, neg_seed, edge_index=pos_ei.detach().cpu()).to(z.device)
        neg_score = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()
        infer_ms = (time.perf_counter() - tic) * 1000.0

    y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
    y_score = np.concatenate([pos_score, neg_score])
    y_hat = (y_score > 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_hat,
        average="binary",
        zero_division=0,
    )
    p_at_k, r_at_k = precision_recall_at_k(y_true, y_score, top_k)

    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "PR_AUC": float(average_precision_score(y_true, y_score)),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "PrecisionAtK": float(p_at_k),
        "RecallAtK": float(r_at_k),
        "InferenceMs": float(infer_ms),
        "PosEdges": int(pos_ei.size(1)),
        "NegEdges": int(neg_ei.size(1)),
    }


def attach_node_features(
    graphs: list[Data], num_nodes: int, embedding_dim: int, device: torch.device
) -> torch.nn.Parameter:
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for g in graphs:
        g.x = x
    return x


def summarize_horizon(df_h: pd.DataFrame) -> dict[str, float]:
    return {
        "Pairs": int(len(df_h)),
        "AUC_Mean": float(df_h["AUC"].mean()),
        "AUC_Std": float(df_h["AUC"].std(ddof=0)),
        "PR_AUC_Mean": float(df_h["PR_AUC"].mean()),
        "PR_AUC_Std": float(df_h["PR_AUC"].std(ddof=0)),
        "PrecisionAtK_Mean": float(df_h["PrecisionAtK"].mean()),
        "PrecisionAtK_Std": float(df_h["PrecisionAtK"].std(ddof=0)),
        "RecallAtK_Mean": float(df_h["RecallAtK"].mean()),
        "RecallAtK_Std": float(df_h["RecallAtK"].std(ddof=0)),
        "F1_Mean": float(df_h["F1"].mean()),
        "F1_Std": float(df_h["F1"].std(ddof=0)),
        "InferenceMs_Mean": float(df_h["InferenceMs"].mean()),
        "InferenceMs_Std": float(df_h["InferenceMs"].std(ddof=0)),
    }


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    horizon_df: pd.DataFrame,
    train_val_rows: list[dict[str, float]],
) -> None:
    lines: list[str] = []
    lines.append("# Multi-Horizon Forecasting Report (G_t -> G_{t+k})")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment evaluates whether future-link prediction remains operationally useful when forecasting "
        "multiple steps ahead using G_t to predict E_{t+k}."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append(f"- Window size: {args.time_window_size} ms")
    lines.append(f"- Horizons: {args.horizons}")
    lines.append(f"- Training horizon anchor: G_0 ... G_{args.train_end_index}")
    lines.append("- Validation windows: selected from later trainable pairs for each k")
    lines.append(f"- Test start input: G_{args.test_start_index}")
    lines.append(f"- Top-K for Precision@K and Recall@K: {args.top_k}")
    lines.append("")
    lines.append("## 3. Horizon Mapping")
    for k in [int(v.strip()) for v in args.horizons.split(",") if v.strip()]:
        ahead_ms = k * args.time_window_size
        lines.append(f"- k={k}: {ahead_ms} ms ahead")
    lines.append("")
    lines.append("## 4. Primary Results By Horizon")
    for _, r in horizon_df.iterrows():
        k = int(r["HorizonK"])
        ahead_ms = int(k * args.time_window_size)
        lines.append(f"### k={k} ({ahead_ms} ms ahead)")
        lines.append(f"- Evaluated test pairs: {int(r['Pairs'])}")
        lines.append(f"- AUC: {r['AUC_Mean']:.4f} +- {r['AUC_Std']:.4f}")
        lines.append(f"- Average Precision (PR-AUC): {r['PR_AUC_Mean']:.4f} +- {r['PR_AUC_Std']:.4f}")
        lines.append(f"- Precision@K: {r['PrecisionAtK_Mean']:.4f} +- {r['PrecisionAtK_Std']:.4f}")
        lines.append(f"- Recall@K: {r['RecallAtK_Mean']:.4f} +- {r['RecallAtK_Std']:.4f}")
        lines.append(f"- F1 (secondary): {r['F1_Mean']:.4f} +- {r['F1_Std']:.4f}")
        lines.append(
            f"- Inference time per prediction (ms): {r['InferenceMs_Mean']:.2f} +- {r['InferenceMs_Std']:.2f}"
        )
        lines.append("")

    lines.append("## 5. Train/Validation Metadata")
    for row in train_val_rows:
        lines.append(
            f"- k={int(row['HorizonK'])}: train_pairs={int(row['TrainPairs'])}, "
            f"val_pairs={int(row['ValPairs'])}, best_epoch={int(row['BestEpoch'])}, "
            f"training_time_sec={row['TrainingTimeSec']:.2f}"
        )
    lines.append("")

    lines.append("## 6. Saved Artifacts")
    lines.append("- horizon_per_pair_metrics.csv")
    lines.append("- horizon_summary.csv")
    lines.append("- horizon_summary.json")

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

    horizons = sorted(set(int(v.strip()) for v in args.horizons.split(",") if v.strip()))
    if any(k <= 0 for k in horizons):
        raise ValueError("All horizons must be positive integers.")

    all_rows: list[dict[str, float]] = []
    train_val_rows: list[dict[str, float]] = []

    for k in horizons:
        print(f"\n===== Horizon k={k} =====")
        max_train_input = args.train_end_index - k
        if max_train_input < 0:
            raise ValueError(f"k={k} is too large for train_end_index={args.train_end_index}.")

        all_train_inputs = list(range(0, max_train_input + 1))
        if len(all_train_inputs) < 3:
            raise ValueError(f"Not enough training pairs for k={k}.")

        val_count = max(1, int(round(len(all_train_inputs) * args.val_ratio)))
        val_inputs = all_train_inputs[-val_count:]
        train_inputs = all_train_inputs[:-val_count]
        if not train_inputs:
            raise ValueError(f"No train inputs left for k={k}; lower val_ratio.")

        test_last_input = len(graphs) - 1 - k
        if args.test_start_index > test_last_input:
            raise ValueError(
                f"test_start_index={args.test_start_index} too large for k={k} with only {len(graphs)} windows."
            )
        test_inputs = list(range(args.test_start_index, test_last_input + 1))

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

        tr_start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_losses = []
            for idx in train_inputs:
                optimizer.zero_grad()
                loss = pair_loss(
                    model,
                    args.model,
                    graphs[idx],
                    graphs[idx + k],
                    neg_seed=args.seed * 100000 + k * 10000 + epoch * 100 + idx,
                )
                loss.backward()
                optimizer.step()
                tr_losses.append(float(loss.item()))

            model.eval()
            with torch.no_grad():
                val_losses = []
                for idx in val_inputs:
                    v_loss = pair_loss(
                        model,
                        args.model,
                        graphs[idx],
                        graphs[idx + k],
                        neg_seed=args.seed * 100000 + k * 10000 + epoch * 100 + 50000 + idx,
                    )
                    val_losses.append(float(v_loss.item()))

            avg_train = float(np.mean(tr_losses))
            avg_val = float(np.mean(val_losses))
            print(f"k={k} | Epoch {epoch:03d} | TrainLoss={avg_train:.6f} | ValLoss={avg_val:.6f}")

            if avg_val < best_val * (1.0 - args.delta):
                best_val = avg_val
                best_epoch = epoch
                no_improve = 0
                best_state = {name: val.detach().clone() for name, val in model.state_dict().items()}
            else:
                no_improve += 1

            if epoch >= args.min_epochs and no_improve >= args.patience:
                print(f"k={k} early stopping at epoch {epoch} (best={best_epoch}).")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        tr_sec = float(time.time() - tr_start)
        train_val_rows.append(
            {
                "HorizonK": float(k),
                "TrainPairs": float(len(train_inputs)),
                "ValPairs": float(len(val_inputs)),
                "BestEpoch": float(best_epoch),
                "TrainingTimeSec": tr_sec,
            }
        )

        for idx in test_inputs:
            row = evaluate_pair(
                model,
                args.model,
                graphs[idx],
                graphs[idx + k],
                neg_seed=args.seed * 200000 + k * 10000 + idx,
                top_k=args.top_k,
            )
            if row is None:
                print(f"k={k} skipping G_{idx} -> E_{idx + k} because E_{idx + k} is empty.")
                continue

            row["HorizonK"] = float(k)
            row["InputGraph"] = float(idx)
            row["TargetGraph"] = float(idx + k)
            all_rows.append(row)

        print(f"k={k} completed with {sum(1 for r in all_rows if int(r['HorizonK']) == k)} evaluated test pairs.")

    if not all_rows:
        raise RuntimeError("No test rows were evaluated across horizons.")

    per_pair_df = pd.DataFrame(all_rows)
    per_pair_path = results_dir / "horizon_per_pair_metrics.csv"
    per_pair_df.to_csv(per_pair_path, index=False)

    horizon_rows = []
    for k in horizons:
        df_h = per_pair_df[per_pair_df["HorizonK"] == float(k)]
        stats = summarize_horizon(df_h)
        stats["HorizonK"] = int(k)
        stats["AheadMs"] = int(k * args.time_window_size)
        horizon_rows.append(stats)

    horizon_df = pd.DataFrame(horizon_rows).sort_values("HorizonK")
    horizon_csv = results_dir / "horizon_summary.csv"
    horizon_df.to_csv(horizon_csv, index=False)

    summary_json = {
        "config": {
            "model": args.model,
            "dataset_path": str(args.dataset_path),
            "time_window_size": args.time_window_size,
            "horizons": horizons,
            "train_end_index": args.train_end_index,
            "test_start_index": args.test_start_index,
            "epochs": args.epochs,
            "seed": args.seed,
            "top_k": args.top_k,
        },
        "train_val": train_val_rows,
        "horizon_summary": horizon_df.to_dict(orient="records"),
    }
    summary_path = results_dir / "horizon_summary.json"
    summary_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    report_path = results_dir / "horizon_experiment_report.md"
    write_report(report_path, args, horizon_df, train_val_rows)

    print("\n===== Final Multi-Horizon Summary =====")
    for _, r in horizon_df.iterrows():
        print(
            f"k={int(r['HorizonK'])} ({int(r['AheadMs'])}ms) | "
            f"AUC={r['AUC_Mean']:.4f} | PR-AUC={r['PR_AUC_Mean']:.4f} | "
            f"P@K={r['PrecisionAtK_Mean']:.4f} | R@K={r['RecallAtK_Mean']:.4f} | "
            f"F1={r['F1_Mean']:.4f} | InferMs={r['InferenceMs_Mean']:.2f}"
        )

    print(f"Saved: {per_pair_path}")
    print(f"Saved: {horizon_csv}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-horizon temporal forecasting experiment with G_t -> G_{t+k}."
    )
    parser.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    parser.add_argument("--results-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--model", type=str, default="GAT")
    parser.add_argument("--time-window-size", type=int, default=100)
    parser.add_argument("--max-time", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)

    parser.add_argument("--horizons", type=str, default="1,5,10,30")
    parser.add_argument("--top-k", type=int, default=100)

    parser.add_argument("--train-end-index", type=int, default=69)
    parser.add_argument("--test-start-index", type=int, default=70)
    parser.add_argument("--val-ratio", type=float, default=0.2)

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
    arguments = build_arg_parser().parse_args()
    run(arguments)
