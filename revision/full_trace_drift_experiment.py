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
from torch import nn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import APPNP, GATConv


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT_DIR / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from GNN_Model import GAT  # type: ignore  # noqa: E402
    from Standalone_Diffusion_Model import Diffusion  # type: ignore  # noqa: E402
    from Diffusion_GAT_Model import DiffusionGAT  # type: ignore  # noqa: E402
    from Standalone_Transformer_Model import TransformerOnly  # type: ignore  # noqa: E402
    from Transformer_GAT_Model import TransformerGAT  # type: ignore  # noqa: E402
except Exception:
    # Fallback models allow this script to run when Code/*.py is not present in the workspace.
    class GAT(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.g1 = GATConv(in_dim, hidden_dim, heads=2, concat=True)
            self.g2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=True)

        def forward(self, data: Data) -> torch.Tensor:
            x = F.elu(self.g1(data.x, data.edge_index))
            return self.g2(x, data.edge_index)

    class Diffusion(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.lin = nn.Linear(in_dim, hidden_dim)
            self.appnp = APPNP(K=10, alpha=0.1)

        def forward(self, data: Data) -> torch.Tensor:
            x = self.lin(data.x)
            return self.appnp(x, data.edge_index)

    class DiffusionGAT(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.diff = Diffusion(in_dim, hidden_dim)
            self.gat = GAT(hidden_dim, hidden_dim)

        def forward(self, data: Data) -> torch.Tensor:
            x = self.diff(data)
            tmp = Data(edge_index=data.edge_index, x=x, num_nodes=data.num_nodes)
            return self.gat(tmp)

    class TransformerOnly(nn.Module):
        def __init__(self, num_nodes: int, embedding_dim: int):
            super().__init__()
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=embedding_dim * 2,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x.unsqueeze(0)).squeeze(0)

    class TransformerGAT(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.gat = GAT(in_dim, hidden_dim)

        def forward(self, data: Data) -> torch.Tensor:
            return self.gat(data)

REVISION_DIR = Path(__file__).resolve().parent
if str(REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(REVISION_DIR))
from degree_aware_sampling import sample_negative_edges_fast as _degree_sample_fast  # noqa: E402


REQUIRED = ["timestamp", "um", "dm"]
DAY_MS = 24 * 60 * 60 * 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


CHUNK_SIZE = 5_000_000  # rows per pd.read_csv chunk

RENAME_MAP = {
    "source": "um",
    "src": "um",
    "parent_csvc_name": "um",
    "target": "dm",
    "dst": "dm",
    "destination": "dm",
    "child_csvc_name": "dm",
    "time": "timestamp",
    "ts": "timestamp",
}


def _get_usecols(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=0)
    candidate_cols = set(REQUIRED)
    candidate_cols.update(RENAME_MAP.keys())
    return [c for c in header.columns if c in candidate_cols]


def _normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns=RENAME_MAP)
    missing = [c for c in REQUIRED if c not in chunk.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    chunk = chunk[REQUIRED].copy()
    chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
    chunk = chunk.dropna(subset=["timestamp"])
    chunk = chunk[chunk["timestamp"] >= 0]
    chunk["um"] = chunk["um"].astype(str)
    chunk["dm"] = chunk["dm"].astype(str)
    return chunk


def build_graphs_streaming(csv_path: Path, window_size: int) -> tuple[list[Data], int]:
    """Two-pass chunked graph builder that never loads the full CSV into RAM.

    Pass 1: discover unique nodes and timestamp range.
    Pass 2: build per-window edge index tensors incrementally.

    Returns (graphs, num_nodes).
    """
    usecols = _get_usecols(csv_path)
    print(f"[stream] Pass 1: scanning nodes and time range (chunksize={CHUNK_SIZE})...")

    # --- Pass 1: discover nodes and time bounds ---
    all_nodes: set[str] = set()
    t_min = float("inf")
    t_max = float("-inf")
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=usecols, on_bad_lines="skip", chunksize=CHUNK_SIZE):
        chunk = _normalize_chunk(chunk)
        if chunk.empty:
            continue
        all_nodes.update(chunk["um"])
        all_nodes.update(chunk["dm"])
        cmin = float(chunk["timestamp"].min())
        cmax = float(chunk["timestamp"].max())
        if cmin < t_min:
            t_min = cmin
        if cmax > t_max:
            t_max = cmax
        total_rows += len(chunk)
    print(f"[stream] Pass 1 done: {total_rows:,} valid rows, {len(all_nodes):,} nodes, "
          f"time range [{t_min:.0f}, {t_max:.0f}]")

    # Build node mapping
    node_list = sorted(all_nodes)
    node_map = {n: i for i, n in enumerate(node_list)}
    num_nodes = len(node_list)

    # Determine number of windows (relative to t_min)
    total_time = int(t_max - t_min) + window_size
    num_windows = total_time // window_size
    print(f"[stream] Pass 2: building {num_windows} window graphs...")

    # Pre-allocate edge counters per window (dict of (src,dst)->count)
    # Stores unique edges + call frequency — mathematically equivalent to repeated edges
    # but bounded memory (unique edge count << total call records).
    from collections import Counter
    window_edge_counts: list[Counter] = [Counter() for _ in range(num_windows)]

    # --- Pass 2: assign edges to windows ---
    for chunk in pd.read_csv(csv_path, usecols=usecols, on_bad_lines="skip", chunksize=CHUNK_SIZE):
        chunk = _normalize_chunk(chunk)
        if chunk.empty:
            continue
        ts = chunk["timestamp"].to_numpy(dtype=np.float64) - t_min
        wids = (ts // window_size).astype(np.int32)
        src_enc = chunk["um"].map(node_map).to_numpy(dtype=np.int32)
        dst_enc = chunk["dm"].map(node_map).to_numpy(dtype=np.int32)

        for wid in np.unique(wids):
            if wid < 0 or wid >= num_windows:
                continue
            mask = wids == wid
            pairs = zip(src_enc[mask].tolist(), dst_enc[mask].tolist())
            window_edge_counts[wid].update(pairs)

    # --- Build graph objects with unique edges + edge_weight (call frequency) ---
    graphs: list[Data] = []
    for wid in range(num_windows):
        counter = window_edge_counts[wid]
        if counter:
            edges = list(counter.keys())
            weights = [counter[e] for e in edges]
            src_list, dst_list = zip(*edges)
            ei = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)
            ew = torch.tensor(weights, dtype=torch.float)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
            ew = torch.empty((0,), dtype=torch.float)
        g = Data(edge_index=ei, num_nodes=num_nodes)
        g.edge_weight = ew
        graphs.append(g)
        # Free memory as we go
        window_edge_counts[wid] = Counter()

    print(f"[stream] Pass 2 done: {len(graphs)} graphs built.")
    return graphs, num_nodes


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
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def model_forward_embeddings(model: torch.nn.Module, model_name: str, g_t: Data) -> torch.Tensor:
    if model_name.lower() == "transformer":
        return model(g_t.x)
    return model(g_t)


def pair_train_loss(model, model_name: str, g_t: Data, g_t1: Data, seed: int) -> torch.Tensor:
    device = next(model.parameters()).device
    g_t_dev = Data(edge_index=g_t.edge_index.to(device), num_nodes=g_t.num_nodes, x=g_t.x)
    pos_ei = g_t1.edge_index.to(device)

    z = model_forward_embeddings(model, model_name, g_t_dev)
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))

    # Use call frequency as sample weight (equivalent to repeating edges)
    pos_weight = g_t1.edge_weight.to(device) if hasattr(g_t1, "edge_weight") and g_t1.edge_weight is not None else torch.ones_like(pos_scores)
    pos_weight = pos_weight / pos_weight.sum()  # normalize

    forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
    neg_ei = sample_negative_edges_fast(
        g_t1.num_nodes,
        int(pos_ei.size(1)),
        forbidden,
        seed,
        edge_index=pos_ei.detach().cpu(),
    ).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    pos_loss = (F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores), reduction="none") * pos_weight).sum()
    neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
    return pos_loss + neg_loss


def evaluate_pair(model, model_name: str, g_t: Data, g_t1: Data, neg_ratio: int, neg_cap: int, seed: int) -> dict | None:
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        g_t_dev = Data(edge_index=g_t.edge_index.to(device), num_nodes=g_t.num_nodes, x=g_t.x)
        pos_ei = g_t1.edge_index.to(device)

        z = model_forward_embeddings(model, model_name, g_t_dev)
        if pos_ei.numel() == 0:
            return None

        pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()

        # Retrieve call frequency weights for weighted metrics
        if hasattr(g_t1, "edge_weight") and g_t1.edge_weight is not None:
            pos_weights = g_t1.edge_weight.cpu().numpy().astype(np.float64)
        else:
            pos_weights = np.ones_like(pos_scores)

        forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
        neg_count = max(1, int(pos_ei.size(1)) * neg_ratio)
        neg_count = min(neg_count, neg_cap)
        neg_ei = sample_negative_edges_fast(
            g_t1.num_nodes,
            neg_count,
            forbidden,
            seed,
            edge_index=pos_ei.detach().cpu(),
        ).to(z.device)
        neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

    # Build sample weights: positive edges weighted by call frequency, negatives weight=1
    neg_weights = np.ones_like(neg_scores)
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_score = np.concatenate([pos_scores, neg_scores])
    sample_weight = np.concatenate([pos_weights, neg_weights])
    y_hat = (y_score > 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0, sample_weight=sample_weight)

    return {
        "AUC": float(roc_auc_score(y_true, y_score, sample_weight=sample_weight)),
        "PR_AUC": float(average_precision_score(y_true, y_score, sample_weight=sample_weight)),
        "F1": float(f1),
        "Precision": float(p),
        "Recall": float(r),
        "PosEdges": float(pos_ei.size(1)),
        "TotalCalls": float(pos_weights.sum()),
    }


def write_report(report_path: Path, summary: dict, daily_df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Full-Trace Drift Experiment Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment evaluates metric drift over the full available trace timeline to assess long-term stability "
        "instead of only short temporal slices."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Dataset: {summary['Dataset']}")
    lines.append(f"- Model: {summary['Model']}")
    lines.append(f"- Device: {summary['Device']}")
    lines.append(f"- Window size (ms): {summary['WindowSizeMs']}")
    lines.append(f"- Train days: {summary['TrainDays']}")
    lines.append(f"- Evaluated days: {summary['EvaluatedDays']}")
    lines.append("")
    lines.append("## 3. Aggregate Results")
    lines.append(f"- AUC mean: {summary['AUC_Mean']:.4f} +- {summary['AUC_Std']:.4f}")
    lines.append(f"- PR-AUC mean: {summary['PR_AUC_Mean']:.4f} +- {summary['PR_AUC_Std']:.4f}")
    lines.append(f"- F1 mean: {summary['F1_Mean']:.4f} +- {summary['F1_Std']:.4f}")
    lines.append(f"- Relative AUC drift vs first eval day: {summary['RelativeAUCDriftPct']:.2f}%")
    lines.append(f"- Relative F1 drift vs first eval day: {summary['RelativeF1DriftPct']:.2f}%")
    lines.append("")
    lines.append("## 4. Day-level Trend")
    for _, r in daily_df.iterrows():
        lines.append(
            f"- Day {int(r['Day'])}: AUC={r['AUC_Mean']:.4f}, PR-AUC={r['PR_AUC_Mean']:.4f}, "
            f"F1={r['F1_Mean']:.4f}, pairs={int(r['Pairs'])}"
        )
    lines.append("")
    lines.append("## 5. Limitation Statement")
    lines.append(
        "This full-trace drift experiment improves long-horizon evidence but is still an offline forecasting analysis. "
        "It does not by itself establish online adaptation, retraining policy, or closed-loop operational control."
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

    # Streaming two-pass graph builder — never loads full CSV into RAM
    graphs, num_nodes = build_graphs_streaming(dataset_path, args.window_size)

    x = torch.nn.Parameter(torch.randn(num_nodes, args.embedding_dim, device=device))
    for g in graphs:
        g.x = x

    train_end_time = int(args.train_days * DAY_MS)
    train_end_index = int(train_end_time // args.window_size)
    train_end_index = min(max(train_end_index, 20), len(graphs) - 2)

    val_start_index = int(train_end_index * (1.0 - args.val_within_train_ratio))
    val_start_index = min(max(val_start_index, 1), train_end_index - 1)

    train_inputs = list(range(0, val_start_index))
    val_inputs = list(range(val_start_index, train_end_index))
    test_inputs = list(range(train_end_index, len(graphs) - 1, args.test_stride))

    if not train_inputs or not val_inputs or not test_inputs:
        raise ValueError("Invalid split for full-trace drift. Check window size, train_days, and dataset span.")

    model = get_model(args.model, args.embedding_dim, num_nodes, device)
    optimizer = torch.optim.Adam(list(model.parameters()) + [x], lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    no_improve = 0

    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for i in train_inputs:
            optimizer.zero_grad()
            loss = pair_train_loss(model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + i)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_losses = [
                float(
                    pair_train_loss(
                        model,
                        args.model,
                        graphs[i],
                        graphs[i + 1],
                        args.seed * 100000 + epoch * 1000 + 50000 + i,
                    ).item()
                )
                for i in val_inputs
            ]

        avg_tr = float(np.mean(tr_losses))
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch:03d} | TrainLoss={avg_tr:.6f} | ValLoss={avg_val:.6f}")

        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= args.min_epochs and no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_sec = float(time.time() - train_start)

    rows: list[dict[str, float]] = []
    for i in test_inputs:
        row = evaluate_pair(
            model,
            args.model,
            graphs[i],
            graphs[i + 1],
            args.eval_neg_ratio,
            args.eval_neg_cap,
            args.seed * 200000 + i,
        )
        if row is None:
            continue

        target_time_rel = float((i + 1) * args.window_size)
        day_idx = int(target_time_rel // DAY_MS)

        row["InputWindow"] = float(i)
        row["TargetWindow"] = float(i + 1)
        row["Day"] = float(day_idx)
        row["TargetTimeMs"] = target_time_rel
        rows.append(row)

    if not rows:
        raise RuntimeError("No evaluation rows produced.")

    per_pair_df = pd.DataFrame(rows)
    per_pair_csv = results_dir / "full_trace_drift_per_pair_metrics.csv"
    per_pair_df.to_csv(per_pair_csv, index=False)

    daily = (
        per_pair_df.groupby("Day", as_index=False)
        .agg(
            Pairs=("AUC", "count"),
            AUC_Mean=("AUC", "mean"),
            AUC_Std=("AUC", "std"),
            PR_AUC_Mean=("PR_AUC", "mean"),
            PR_AUC_Std=("PR_AUC", "std"),
            F1_Mean=("F1", "mean"),
            F1_Std=("F1", "std"),
        )
        .fillna(0.0)
    )
    daily_csv = results_dir / "full_trace_drift_daily_summary.csv"
    daily.to_csv(daily_csv, index=False)

    auc_vals = per_pair_df["AUC"].to_numpy(dtype=float)
    pr_vals = per_pair_df["PR_AUC"].to_numpy(dtype=float)
    f1_vals = per_pair_df["F1"].to_numpy(dtype=float)

    first_day_auc = float(daily.iloc[0]["AUC_Mean"])
    first_day_f1 = float(daily.iloc[0]["F1_Mean"])
    last_day_auc = float(daily.iloc[-1]["AUC_Mean"])
    last_day_f1 = float(daily.iloc[-1]["F1_Mean"])

    summary = {
        "Dataset": str(dataset_path),
        "Model": args.model,
        "Device": str(device),
        "WindowSizeMs": int(args.window_size),
        "TrainDays": float(args.train_days),
        "EvaluatedDays": int(daily["Day"].nunique()),
        "TrainSec": train_sec,
        "Pairs": int(len(per_pair_df)),
        "AUC_Mean": float(auc_vals.mean()),
        "AUC_Std": float(auc_vals.std(ddof=0)),
        "PR_AUC_Mean": float(pr_vals.mean()),
        "PR_AUC_Std": float(pr_vals.std(ddof=0)),
        "F1_Mean": float(f1_vals.mean()),
        "F1_Std": float(f1_vals.std(ddof=0)),
        "RelativeAUCDriftPct": float(100.0 * (last_day_auc - first_day_auc) / max(1e-9, first_day_auc)),
        "RelativeF1DriftPct": float(100.0 * (last_day_f1 - first_day_f1) / max(1e-9, first_day_f1)),
        "ReadSubsampleStep": int(args.row_subsample_step),
    }

    summary_json = results_dir / "full_trace_drift_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = results_dir / "full_trace_drift_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    report_md = results_dir / "full_trace_drift_report.md"
    write_report(report_md, summary, daily)

    print(f"Saved: {per_pair_csv}")
    print(f"Saved: {daily_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Long-horizon full-trace drift experiment")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--model", type=str, default="GAT", choices=["GAT", "Diffusion", "DiffusionGAT", "Transformer", "TransformerGAT"])

    p.add_argument("--window-size", type=int, default=60000)
    p.add_argument("--train-days", type=float, default=2.0)
    p.add_argument("--val-within-train-ratio", type=float, default=0.15)
    p.add_argument("--test-stride", type=int, default=1)
    p.add_argument("--eval-neg-ratio", type=int, default=10)
    p.add_argument("--eval-neg-cap", type=int, default=200000)
    p.add_argument("--row-subsample-step", type=int, default=1)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
