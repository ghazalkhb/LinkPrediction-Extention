import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
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


@dataclass
class WindowSpec:
    name: str
    size_ms: int
    overlap: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_windows_with_overlap(df: pd.DataFrame, window_size: int, overlap: float, max_time: int):
    step = int(round(window_size * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("Window step must be positive. Check overlap value.")

    windows = []
    starts = []
    ends = []
    used_indices: list[np.ndarray] = []

    for start in range(0, max_time, step):
        end = start + window_size
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        windows.append(w)
        starts.append(start)
        ends.append(end)
        used_indices.append(w.index.to_numpy(dtype=np.int64, copy=False))

    total_occ = float(sum(len(idx) for idx in used_indices))
    if total_occ == 0:
        dup_pct = 0.0
    else:
        unique_occ = float(len(np.unique(np.concatenate(used_indices)))) if used_indices else 0.0
        dup_pct = max(0.0, (total_occ - unique_occ) / total_occ * 100.0)

    return windows, starts, ends, dup_pct


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


def sample_negative_edges_fast(num_nodes: int, count: int, forbidden_keys: np.ndarray, seed: int, edge_index: torch.Tensor = None) -> torch.Tensor:
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


def train_pair_loss(model, model_name, g_t: Data, g_t1: Data, seed: int) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, g_t)
    pos_ei = g_t1.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
    neg_ei = sample_negative_edges_fast(g_t1.num_nodes, int(pos_ei.size(1)), forbidden, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    return F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy(
        neg_scores, torch.zeros_like(neg_scores)
    )


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k_eval = int(max(1, min(k, len(y_score))))
    top_idx = np.argsort(y_score)[::-1][:k_eval]
    return float(np.sum(y_true[top_idx])) / float(k_eval)


def eval_pair_metrics(model, model_name, g_t: Data, g_t1: Data, neg_ratio: int, top_k: int, seed: int):
    model.eval()
    with torch.no_grad():
        z = model_forward_embeddings(model, model_name, g_t)
        pos_ei = g_t1.edge_index
        if pos_ei.numel() == 0:
            return None

        pos_count = int(pos_ei.size(1))
        neg_count = int(pos_count * neg_ratio)

        pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
        forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
        neg_ei = sample_negative_edges_fast(g_t1.num_nodes, neg_count, forbidden, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
        neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
        y_score = np.concatenate([pos_scores, neg_scores])
        y_hat = (y_score > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_hat, average="binary", zero_division=0
        )

        return {
            "AUC": float(roc_auc_score(y_true, y_score)),
            "F1": float(f1),
            "PR_AUC": float(average_precision_score(y_true, y_score)),
            "PrecisionAtK": float(precision_at_k(y_true, y_score, top_k)),
            "Precision": float(precision),
            "Recall": float(recall),
            "PosEdges": float(pos_count),
        }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    metrics = ["AUC", "F1", "PR_AUC", "PrecisionAtK", "Precision", "Recall"]
    out = {"Pairs": float(len(rows))}
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        out[f"{m}_Mean"] = float(vals.mean())
        out[f"{m}_Std"] = float(vals.std(ddof=0))
    out["AvgPosEdgesPerPair"] = float(np.mean([r["PosEdges"] for r in rows]))
    return out


def run_setting(
    spec: WindowSpec,
    df: pd.DataFrame,
    num_nodes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    windows, starts, ends, dup_pct = create_windows_with_overlap(df, spec.size_ms, spec.overlap, args.max_time)
    graphs = [create_graph(w, num_nodes) for w in windows]

    x = torch.nn.Parameter(torch.randn(num_nodes, args.embedding_dim, device=device))
    for g in graphs:
        g.x = x

    graphs = [g.to(device) for g in graphs]

    train_pairs = [
        i for i in range(len(graphs) - 1) if ends[i] <= args.train_time_end and ends[i + 1] <= args.train_time_end
    ]
    test_pairs = [i for i in range(len(graphs) - 1) if starts[i] >= args.test_time_start and starts[i + 1] >= args.test_time_start]

    if len(train_pairs) < 5 or len(test_pairs) < 5:
        raise ValueError(f"Not enough train/test pairs for setting {spec.name}")

    val_count = max(1, int(round(len(train_pairs) * args.val_ratio)))
    val_pairs = train_pairs[-val_count:]
    tr_pairs = train_pairs[:-val_count]
    if not tr_pairs:
        raise ValueError(f"No training pairs left for setting {spec.name}")

    model = get_model(args.model, args.embedding_dim, num_nodes, device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [x],
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
        for i in tr_pairs:
            optimizer.zero_grad()
            loss = train_pair_loss(model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + i)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            v_losses = [
                float(
                    train_pair_loss(
                        model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + 50000 + i
                    ).item()
                )
                for i in val_pairs
            ]

        avg_val = float(np.mean(v_losses))
        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_sec = time.time() - train_start

    eval_start = time.time()
    rows = []
    for i in test_pairs:
        row = eval_pair_metrics(
            model,
            args.model,
            graphs[i],
            graphs[i + 1],
            neg_ratio=args.eval_neg_ratio,
            top_k=args.top_k,
            seed=args.seed * 200000 + i,
        )
        if row is not None:
            rows.append(row)

    eval_sec = time.time() - eval_start
    s = summarize(rows)

    return {
        "Setting": spec.name,
        "WindowSizeMs": float(spec.size_ms),
        "OverlapPct": float(spec.overlap * 100.0),
        "TrainWindows": float(len(tr_pairs) + len(val_pairs) + 1),
        "TestWindows": float(len(test_pairs) + 1),
        "AvgEdgesPerWindow": float(np.mean([g.edge_index.size(1) for g in graphs])),
        "DuplicatedEdgePct": float(dup_pct),
        "BestEpoch": float(best_epoch),
        "TrainTimeSec": float(train_sec),
        "EvalTimeSec": float(eval_sec),
        **s,
    }


def write_report(report_path: Path, args: argparse.Namespace, summary_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Fair Overlap Comparison Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment compares overlap and non-overlap windowing fairly, including the reviewer-requested "
        "comparison between 50 ms non-overlap and 100 ms with 50% overlap."
    )
    lines.append("")
    lines.append("## 2. Compared Settings")
    lines.append("- 50 ms non-overlap")
    lines.append("- 100 ms non-overlap")
    lines.append("- 100 ms with 50% overlap")
    lines.append("- 200 ms non-overlap")
    lines.append("- 500 ms non-overlap")
    lines.append("")
    lines.append("## 3. Shared Evaluation Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append("- Forecasting direction: G_t -> E_{t+1}")
    lines.append(f"- Train time end: {args.train_time_end}")
    lines.append(f"- Test time start: {args.test_time_start}")
    lines.append(f"- Test negative ratio: 1:{args.eval_neg_ratio}")
    lines.append(f"- Precision@K uses K={args.top_k}")
    lines.append("")
    lines.append("## 4. Results")

    for _, r in summary_df.iterrows():
        lines.append(f"### {r['Setting']}")
        lines.append(f"- Train windows: {int(r['TrainWindows'])}")
        lines.append(f"- Test windows: {int(r['TestWindows'])}")
        lines.append(f"- Average edges per window: {r['AvgEdgesPerWindow']:.1f}")
        lines.append(f"- Duplicated edge percentage from overlap: {r['DuplicatedEdgePct']:.2f}%")
        lines.append(f"- AUC: {r['AUC_Mean']:.4f} +- {r['AUC_Std']:.4f}")
        lines.append(f"- F1: {r['F1_Mean']:.4f} +- {r['F1_Std']:.4f}")
        lines.append(f"- PR-AUC: {r['PR_AUC_Mean']:.4f} +- {r['PR_AUC_Std']:.4f}")
        lines.append(f"- Precision@K: {r['PrecisionAtK_Mean']:.4f} +- {r['PrecisionAtK_Std']:.4f}")
        lines.append(f"- Runtime train/eval (sec): {r['TrainTimeSec']:.2f} / {r['EvalTimeSec']:.2f}")
        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append(
        "This comparison distinguishes improvements from true temporal resolution effects versus overlap-induced "
        "duplication and larger effective sample reuse."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fair overlap experiment across five window settings.")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--model", type=str, default="GAT")
    p.add_argument("--embedding-dim", type=int, default=64)

    p.add_argument("--train-time-end", type=int, default=7000)
    p.add_argument("--test-time-start", type=int, default=7000)
    p.add_argument("--max-time", type=int, default=10000)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--eval-neg-ratio", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
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

    settings = [
        WindowSpec("50ms_non_overlap", 50, 0.0),
        WindowSpec("100ms_non_overlap", 100, 0.0),
        WindowSpec("100ms_overlap_50pct", 100, 0.5),
        WindowSpec("200ms_non_overlap", 200, 0.0),
        WindowSpec("500ms_non_overlap", 500, 0.0),
    ]

    rows = []
    for spec in settings:
        print(f"\n=== Running {spec.name} ===")
        row = run_setting(spec, df, len(all_nodes), args, device)
        rows.append(row)
        print(
            f"{spec.name} | AUC={row['AUC_Mean']:.4f} | F1={row['F1_Mean']:.4f} | "
            f"PR-AUC={row['PR_AUC_Mean']:.4f} | P@K={row['PrecisionAtK_Mean']:.4f}"
        )

    summary_df = pd.DataFrame(rows)

    per_setting_path = results_dir / "overlap_fair_summary.csv"
    summary_df.to_csv(per_setting_path, index=False)

    summary_json = {
        "config": {
            "model": args.model,
            "dataset_path": str(args.dataset_path),
            "train_time_end": args.train_time_end,
            "test_time_start": args.test_time_start,
            "max_time": args.max_time,
            "eval_neg_ratio": args.eval_neg_ratio,
            "top_k": args.top_k,
        },
        "results": summary_df.to_dict(orient="records"),
    }
    summary_json_path = results_dir / "overlap_fair_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    report_path = results_dir / "overlap_fair_experiment_report.md"
    write_report(report_path, args, summary_df)

    print("\n=== Fair Overlap Summary ===")
    print(summary_df[["Setting", "AUC_Mean", "F1_Mean", "PR_AUC_Mean", "PrecisionAtK_Mean", "TrainTimeSec", "EvalTimeSec"]])
    print(f"Saved: {per_setting_path}")
    print(f"Saved: {summary_json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
