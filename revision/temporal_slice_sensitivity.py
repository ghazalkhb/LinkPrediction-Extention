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


REQUIRED = ["timestamp", "um", "dm"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_calls(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    rename_map = {
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

    candidate_cols = set(REQUIRED)
    candidate_cols.update(rename_map.keys())
    usecols = [c for c in header.columns if c in candidate_cols]

    df = pd.read_csv(csv_path, on_bad_lines="skip", usecols=usecols)
    df = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    df = df[REQUIRED].copy()
    df = df.dropna()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= 0]
    df["um"] = df["um"].astype(str)
    df["dm"] = df["dm"].astype(str)
    return df.sort_values("timestamp").reset_index(drop=True)


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


def train_pair_loss(model, model_name: str, g_t: Data, g_t1: Data, seed: int) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, g_t)
    pos_ei = g_t1.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
    neg_ei = sample_negative_edges_fast(g_t1.num_nodes, pos_ei.size(1), forbidden, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    return F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy(
        neg_scores, torch.zeros_like(neg_scores)
    )


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k_eval = int(max(1, min(k, len(y_score))))
    top_idx = np.argsort(y_score)[::-1][:k_eval]
    return float(np.sum(y_true[top_idx])) / float(k_eval)


def evaluate_pair(model, model_name: str, g_t: Data, g_t1: Data, neg_ratio: int, top_k: int, seed: int):
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
    out: dict[str, float] = {"Pairs": float(len(rows))}
    metrics = ["AUC", "F1", "PR_AUC", "PrecisionAtK", "Precision", "Recall"]
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        out[f"{m}_Mean"] = float(vals.mean())
        out[f"{m}_Std"] = float(vals.std(ddof=0))
    out["AvgPosEdgesPerPair"] = float(np.mean([r["PosEdges"] for r in rows]))
    return out


def run_slice(slice_name: str, sdf: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> dict | None:
    if sdf.empty:
        return None

    sdf = sdf.copy()
    sdf["timestamp"] = sdf["timestamp"] - float(sdf["timestamp"].min())

    services = pd.concat([sdf["um"], sdf["dm"]]).unique()
    node_map = {node: i for i, node in enumerate(services)}
    sdf["um_encoded"] = sdf["um"].map(node_map)
    sdf["dm_encoded"] = sdf["dm"].map(node_map)

    max_time = int(sdf["timestamp"].max()) + args.window_size
    windows = []
    for start in range(0, max_time, args.window_size):
        w = sdf[(sdf["timestamp"] >= start) & (sdf["timestamp"] < start + args.window_size)]
        windows.append(w)

    graphs = [create_graph(w, len(services)) for w in windows]
    x = torch.nn.Parameter(torch.randn(len(services), args.embedding_dim, device=device))
    for g in graphs:
        g.x = x
    graphs = [g.to(device) for g in graphs]

    num_windows = len(graphs)
    if num_windows < args.min_windows_required:
        return {
            "Slice": slice_name,
            "Status": "insufficient_windows",
            "NumRows": float(len(sdf)),
            "NumServices": float(len(services)),
            "NumWindows": float(num_windows),
        }

    train_end = int(num_windows * args.train_ratio)
    train_end = min(max(train_end, 6), num_windows - 3)
    val_start = int(train_end * (1.0 - args.val_within_train_ratio))
    val_start = min(max(val_start, 1), train_end - 1)

    train_pairs = list(range(0, val_start))
    val_pairs = list(range(val_start, train_end))
    test_pairs = list(range(train_end, num_windows - 1))

    if not train_pairs or not val_pairs or not test_pairs:
        return {
            "Slice": slice_name,
            "Status": "invalid_split",
            "NumRows": float(len(sdf)),
            "NumServices": float(len(services)),
            "NumWindows": float(num_windows),
        }

    model = get_model(args.model, args.embedding_dim, len(services), device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [x],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i in train_pairs:
            optimizer.zero_grad()
            loss = train_pair_loss(model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + i)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_losses = [
                float(
                    train_pair_loss(
                        model,
                        args.model,
                        graphs[i],
                        graphs[i + 1],
                        args.seed * 100000 + epoch * 1000 + 50000 + i,
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

    train_sec = float(time.time() - t_train)

    rows = []
    for i in test_pairs:
        row = evaluate_pair(
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

    if not rows:
        return {
            "Slice": slice_name,
            "Status": "no_eval_rows",
            "NumRows": float(len(sdf)),
            "NumServices": float(len(services)),
            "NumWindows": float(num_windows),
        }

    s = summarize(rows)

    unique_edges = len(pd.util.hash_pandas_object(pd.DataFrame({"um": sdf["um"], "dm": sdf["dm"]}), index=False).unique())
    density = 0.0 if len(services) <= 1 else float(unique_edges / (len(services) * (len(services) - 1)))

    return {
        "Slice": slice_name,
        "Status": "ok",
        "NumRows": float(len(sdf)),
        "NumServices": float(len(services)),
        "UniqueDirectedEdges": float(unique_edges),
        "NumWindows": float(num_windows),
        "MeanEdgesPerWindow": float(np.mean([g.edge_index.size(1) for g in graphs])),
        "MedianEdgesPerWindow": float(np.median([g.edge_index.size(1) for g in graphs])),
        "EdgeDensity": density,
        "TrainPairs": float(len(train_pairs)),
        "ValPairs": float(len(val_pairs)),
        "TestPairs": float(len(test_pairs)),
        "BestEpoch": float(best_epoch),
        "TrainTimeSec": train_sec,
        **s,
    }


def write_report(report_path: Path, args: argparse.Namespace, summary_df: pd.DataFrame, mode_used: str, limitation_note: str) -> None:
    lines = []
    lines.append("# Longer Temporal Slice / Sensitivity Experiment")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment tests temporal-slice stability under the corrected rolling protocol to address reviewer concerns "
        "that short slices may distort conclusions."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Window size: {args.window_size}")
    lines.append(f"- Rolling direction: G_t -> E_(t+1)")
    lines.append(f"- Evaluation negative ratio: 1:{args.eval_neg_ratio}")
    lines.append(f"- Precision@K: K={args.top_k}")
    lines.append(f"- Sensitivity mode used: {mode_used}")
    lines.append("")

    lines.append("## 3. Slice Results")
    for _, r in summary_df.iterrows():
        lines.append(f"### {r['Slice']} ({r['Status']})")
        lines.append(f"- Rows: {int(r['NumRows'])}")
        lines.append(f"- Services: {int(r['NumServices'])}")
        lines.append(f"- Windows: {int(r['NumWindows'])}")
        if r["Status"] == "ok":
            lines.append(f"- Mean/median edges per window: {r['MeanEdgesPerWindow']:.2f} / {r['MedianEdgesPerWindow']:.2f}")
            lines.append(f"- Edge density: {r['EdgeDensity']:.8f}")
            lines.append(f"- AUC: {r['AUC_Mean']:.4f} +- {r['AUC_Std']:.4f}")
            lines.append(f"- F1: {r['F1_Mean']:.4f} +- {r['F1_Std']:.4f}")
            lines.append(f"- PR-AUC: {r['PR_AUC_Mean']:.4f} +- {r['PR_AUC_Std']:.4f}")
            lines.append(f"- Precision@K: {r['PrecisionAtK_Mean']:.4f} +- {r['PrecisionAtK_Std']:.4f}")
            lines.append(f"- Runtime (train sec): {r['TrainTimeSec']:.2f}")
        lines.append("")

    lines.append("## 4. Stability Comment")
    ok = summary_df[summary_df["Status"] == "ok"]
    if len(ok) >= 2:
        auc_span = float(ok["AUC_Mean"].max() - ok["AUC_Mean"].min())
        f1_span = float(ok["F1_Mean"].max() - ok["F1_Mean"].min())
        pr_span = float(ok["PR_AUC_Mean"].max() - ok["PR_AUC_Mean"].min())
        lines.append(
            f"Observed metric spread across evaluated slices: AUC span={auc_span:.4f}, F1 span={f1_span:.4f}, PR-AUC span={pr_span:.4f}."
        )
    else:
        lines.append("Not enough valid slices to assess stability quantitatively.")

    lines.append("")
    lines.append("## 5. Limitation")
    lines.append(limitation_note)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Longer temporal slice sensitivity experiment.")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--model", type=str, default="GAT")
    p.add_argument("--window-size", type=int, default=100)

    p.add_argument("--slice-mode", type=str, default="auto", choices=["auto", "hours", "segments"])
    p.add_argument("--hour-slices", type=str, default="1,2,4,8")
    p.add_argument("--timestamp-unit-ms", type=float, default=1.0)

    p.add_argument("--eval-neg-ratio", type=int, default=10)
    p.add_argument("--top-k", type=int, default=100)

    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-within-train-ratio", type=float, default=0.15)
    p.add_argument("--min-windows-required", type=int, default=20)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = load_calls(Path(args.dataset_path))
    t_min = float(df["timestamp"].min())
    t_max = float(df["timestamp"].max())
    span_raw = t_max - t_min
    span_ms = span_raw * float(args.timestamp_unit_ms)

    hour_list = [int(x.strip()) for x in args.hour_slices.split(",") if x.strip()]
    max_needed_ms = max(hour_list) * 3600.0 * 1000.0

    if args.slice_mode == "hours":
        mode_used = "hours"
    elif args.slice_mode == "segments":
        mode_used = "segments"
    else:
        mode_used = "hours" if span_ms >= max_needed_ms else "segments"

    slices: list[tuple[str, pd.DataFrame]] = []
    limitation_note = ""

    if mode_used == "hours":
        for h in hour_list:
            dur_raw = (h * 3600.0 * 1000.0) / float(args.timestamp_unit_ms)
            sdf = df[(df["timestamp"] >= t_min) & (df["timestamp"] < t_min + dur_raw)].copy()
            slices.append((f"first_{h}h", sdf))
        limitation_note = (
            "Hour-based sensitivity was feasible and executed directly. "
            "If full multi-day coverage is still required, extend this pilot to contiguous later hour blocks."
        )
    else:
        # Early/middle/late equal-duration pilot slices.
        q1 = t_min + span_raw / 3.0
        q2 = t_min + 2.0 * span_raw / 3.0

        early = df[(df["timestamp"] >= t_min) & (df["timestamp"] < q1)].copy()
        middle = df[(df["timestamp"] >= q1) & (df["timestamp"] < q2)].copy()
        late = df[(df["timestamp"] >= q2) & (df["timestamp"] <= t_max)].copy()

        slices = [("early_third", early), ("middle_third", middle), ("late_third", late)]

        limitation_note = (
            "Requested 1/2/4/8-hour slices were not feasible under inferred timestamp span/units for this file, "
            "so a structured early/middle/late pilot was run instead. "
            "This is a sensitivity proxy and should be followed by true longer-hour slices when full-duration traces are available."
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    rows = []
    for sname, sdf in slices:
        print(f"Running slice: {sname}")
        out = run_slice(sname, sdf, args, device)
        if out is not None:
            rows.append(out)

    if not rows:
        raise RuntimeError("No slice outputs were produced.")

    summary_df = pd.DataFrame(rows)

    csv_path = results_dir / "temporal_slice_sensitivity_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    json_path = results_dir / "temporal_slice_sensitivity_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "dataset_path": args.dataset_path,
                    "model": args.model,
                    "window_size": args.window_size,
                    "slice_mode_requested": args.slice_mode,
                    "slice_mode_used": mode_used,
                    "hour_slices": hour_list,
                    "timestamp_unit_ms": args.timestamp_unit_ms,
                    "eval_neg_ratio": args.eval_neg_ratio,
                    "top_k": args.top_k,
                },
                "summary": summary_df.to_dict(orient="records"),
                "limitation": limitation_note,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = results_dir / "temporal_slice_sensitivity_report.md"
    write_report(report_path, args, summary_df, mode_used, limitation_note)

    print("\n=== Temporal Slice Sensitivity Summary ===")
    print(summary_df[["Slice", "Status", "NumRows", "NumWindows"] + [c for c in ["AUC_Mean", "F1_Mean", "PR_AUC_Mean", "PrecisionAtK_Mean"] if c in summary_df.columns]])
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
