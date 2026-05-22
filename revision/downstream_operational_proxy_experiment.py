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
    candidate_cols.add("rt")
    usecols = [c for c in header.columns if c in candidate_cols]

    df = pd.read_csv(csv_path, on_bad_lines="skip", usecols=usecols)
    df = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    keep_cols = ["timestamp", "um", "dm"] + (["rt"] if "rt" in df.columns else [])
    df = df[keep_cols].copy()
    df = df.dropna(subset=["timestamp", "um", "dm"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= 0]
    df["um"] = df["um"].astype(str)
    df["dm"] = df["dm"].astype(str)

    if "rt" in df.columns:
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")

    return df.sort_values("timestamp").reset_index(drop=True)


def create_time_windows(df: pd.DataFrame, window_size: int, max_time: int) -> list[pd.DataFrame]:
    windows = []
    for start in range(0, max_time, window_size):
        windows.append(df[(df["timestamp"] >= start) & (df["timestamp"] < start + window_size)])
    return windows


def create_graph(window_df: pd.DataFrame, num_nodes: int) -> Data:
    if window_df.empty:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        um = window_df["um_encoded"].to_numpy(dtype=np.int64, copy=False)
        dm = window_df["dm_encoded"].to_numpy(dtype=np.int64, copy=False)
        edge_index = torch.from_numpy(np.vstack([um, dm])).to(dtype=torch.long)
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


def edge_index_to_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    if edge_index.numel() == 0:
        return set()
    return set((int(s), int(d)) for s, d in edge_index.t().tolist())


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


def model_forward_embeddings(model: torch.nn.Module, model_name: str, graph_t: Data) -> torch.Tensor:
    if model_name.lower() == "transformer":
        return model(graph_t.x)
    return model(graph_t)


def pair_train_loss(model, model_name: str, g_t: Data, g_t1: Data, seed: int) -> torch.Tensor:
    z = model_forward_embeddings(model, model_name, g_t)
    pos_ei = g_t1.edge_index
    if pos_ei.numel() == 0:
        return torch.tensor(0.0, device=z.device, requires_grad=True)

    pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1))
    forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
    neg_ei = sample_negative_edges_fast(
        g_t1.num_nodes,
        pos_ei.size(1),
        forbidden,
        seed,
        edge_index=pos_ei.detach().cpu(),
    ).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    return F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy(
        neg_scores, torch.zeros_like(neg_scores)
    )


def aggregate(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[f"{key}_Mean"] = float(vals.mean()) if vals.size > 0 else float("nan")
        out[f"{key}_Std"] = float(vals.std(ddof=0)) if vals.size > 0 else float("nan")
    return out


def write_report(
    report_path: Path,
    args: argparse.Namespace,
    summary: dict,
    has_rt: bool,
    train_sec: float,
) -> None:
    lines: list[str] = []
    lines.append("# Downstream Operational Proxy Validation Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment evaluates a downstream proxy for operational benefit by mapping link-prediction outputs "
        "to anomaly-triage effectiveness."
    )
    lines.append("")
    lines.append("## 2. Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Dataset: {args.dataset_path}")
    lines.append(f"- Forecasting: G_t -> E_(t+1)")
    lines.append(f"- Top-K triage budget: {args.top_k}")
    lines.append(f"- Candidate negative ratio: 1:{args.candidate_neg_ratio}")
    lines.append(f"- Window anomaly percentile (edge volume/churn): p{args.anomaly_percentile}")
    lines.append(f"- Alarm threshold percentile over validation risk score: p{args.alarm_percentile}")
    lines.append(f"- Training runtime (sec): {train_sec:.2f}")
    lines.append("")
    lines.append("## 3. Results")
    lines.append(f"- AUC: {summary['AUC_Mean']:.4f} +- {summary['AUC_Std']:.4f}")
    lines.append(f"- PR-AUC: {summary['PR_AUC_Mean']:.4f} +- {summary['PR_AUC_Std']:.4f}")
    lines.append(f"- Triage Precision@K: {summary['TriagePrecisionAtK_Mean']:.4f} +- {summary['TriagePrecisionAtK_Std']:.4f}")
    lines.append(f"- Triage Recall(New Edges)@K: {summary['TriageRecallNewAtK_Mean']:.4f} +- {summary['TriageRecallNewAtK_Std']:.4f}")
    lines.append(f"- Anomaly Alert Precision: {summary['AlertPrecision_Mean']:.4f} +- {summary['AlertPrecision_Std']:.4f}")
    lines.append(f"- Anomaly Alert Recall: {summary['AlertRecall_Mean']:.4f} +- {summary['AlertRecall_Std']:.4f}")
    lines.append(f"- Anomaly Alert F1: {summary['AlertF1_Mean']:.4f} +- {summary['AlertF1_Std']:.4f}")

    if has_rt:
        lines.append(f"- RT-based SLO proxy recall@K: {summary['SLOProxyRecallAtK_Mean']:.4f} +- {summary['SLOProxyRecallAtK_Std']:.4f}")
    else:
        lines.append("- RT-based SLO proxy: unavailable (no rt column in dataset)")

    lines.append("")
    lines.append("## 4. Limitation Statement")
    lines.append(
        "This is a proxy-level operational validation (anomaly triage and optional RT-risk proxy), not a full "
        "closed-loop production evaluation of SLO compliance or mitigation outcomes. Full downstream operational "
        "validation remains future work."
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

    df = load_calls(dataset_path)
    has_rt = "rt" in df.columns

    all_nodes = pd.concat([df["um"], df["dm"]]).unique()
    node_map = {n: i for i, n in enumerate(all_nodes)}
    df["um_encoded"] = df["um"].map(node_map)
    df["dm_encoded"] = df["dm"].map(node_map)

    if args.max_time is None:
        max_time = int(df["timestamp"].max()) + args.time_window_size
    else:
        max_time = int(args.max_time)

    windows = create_time_windows(df, args.time_window_size, max_time)
    graphs = [create_graph(w, len(all_nodes)) for w in windows]

    node_features = torch.nn.Parameter(torch.randn(len(all_nodes), args.embedding_dim, device=device))
    for g in graphs:
        g.x = node_features
    graphs = [g.to(device) for g in graphs]

    if len(graphs) <= args.train_end_index + 1:
        raise ValueError("Not enough windows for requested split. Reduce train_end_index or increase max_time.")

    train_inputs = list(range(0, args.val_start_index))
    val_inputs = list(range(args.val_start_index, args.train_end_index))
    test_inputs = list(range(args.test_start_index, len(graphs) - 1))
    if not train_inputs or not val_inputs or not test_inputs:
        raise ValueError("Invalid train/val/test split.")

    model = get_model(args.model, args.embedding_dim, len(all_nodes), device)
    optimizer = torch.optim.Adam(list(model.parameters()) + [node_features], lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for i in train_inputs:
            optimizer.zero_grad()
            loss = pair_train_loss(model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + i)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

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

        avg_train = float(np.mean(train_losses))
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
            print(f"Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_sec = float(time.time() - t_train)

    train_edge_counts = np.array([int(graphs[i + 1].edge_index.size(1)) for i in train_inputs], dtype=float)
    train_churns = []
    for i in train_inputs:
        e_t = edge_index_to_set(graphs[i].edge_index.detach().cpu())
        e_t1 = edge_index_to_set(graphs[i + 1].edge_index.detach().cpu())
        denom = max(1, len(e_t | e_t1))
        train_churns.append(float(len(e_t ^ e_t1)) / float(denom))

    edge_count_thr = float(np.percentile(train_edge_counts, args.anomaly_percentile))
    churn_thr = float(np.percentile(np.array(train_churns, dtype=float), args.anomaly_percentile))

    rt_thr = None
    if has_rt:
        rt_train = windows[0 : args.train_end_index]
        rt_vals = np.concatenate(
            [w["rt"].dropna().to_numpy(dtype=float) for w in rt_train if "rt" in w.columns and not w.empty]
        )
        if rt_vals.size > 0:
            rt_thr = float(np.percentile(rt_vals, args.slo_rt_percentile))

    # Use validation windows to calibrate alert threshold from risk score distribution.
    val_risks = []
    model.eval()
    with torch.no_grad():
        for i in val_inputs:
            z = model_forward_embeddings(model, args.model, graphs[i])
            s = torch.sigmoid((z[graphs[i + 1].edge_index[0]] * z[graphs[i + 1].edge_index[1]]).sum(dim=1))
            if s.numel() > 0:
                val_risks.append(float(torch.quantile(s, 0.9).item()))
    alarm_thr = float(np.percentile(np.array(val_risks, dtype=float), args.alarm_percentile)) if val_risks else 0.5

    rows: list[dict[str, float]] = []
    for i in test_inputs:
        g_t = graphs[i]
        g_t1 = graphs[i + 1]
        pos_ei = g_t1.edge_index
        if pos_ei.numel() == 0:
            continue

        with torch.no_grad():
            z = model_forward_embeddings(model, args.model, g_t)
            pos_scores_t = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()

            pos_count = int(pos_ei.size(1))
            neg_count = int(pos_count * args.candidate_neg_ratio)
            forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
            neg_ei = sample_negative_edges_fast(
                g_t1.num_nodes,
                neg_count,
                forbidden,
                args.seed * 200000 + i,
                edge_index=pos_ei.detach().cpu(),
            ).to(device)
            neg_scores_t = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

        y_true = np.concatenate([np.ones_like(pos_scores_t), np.zeros_like(neg_scores_t)])
        y_score = np.concatenate([pos_scores_t, neg_scores_t])

        # Build edge tuples aligned with y_score so top-K can be interpreted operationally.
        pos_edges = [(int(a), int(b)) for a, b in pos_ei.t().detach().cpu().tolist()]
        neg_edges = [(int(a), int(b)) for a, b in neg_ei.t().detach().cpu().tolist()]
        all_edges = pos_edges + neg_edges

        top_k = int(min(args.top_k, len(all_edges)))
        top_idx = np.argsort(y_score)[::-1][:top_k]
        top_edges = [all_edges[j] for j in top_idx]

        e_t = edge_index_to_set(g_t.edge_index.detach().cpu())
        e_t1 = set(pos_edges)
        new_edges = e_t1 - e_t

        top_true_flags = [1 if e in e_t1 else 0 for e in top_edges]
        triage_precision_at_k = float(np.mean(top_true_flags)) if top_true_flags else 0.0

        if new_edges:
            captured_new = sum(1 for e in top_edges if e in new_edges)
            triage_recall_new_at_k = float(captured_new / len(new_edges))
        else:
            triage_recall_new_at_k = 0.0

        edge_count = float(len(e_t1))
        churn = float(len(e_t ^ e_t1) / max(1, len(e_t | e_t1)))
        anomaly_true = 1 if (edge_count >= edge_count_thr or churn >= churn_thr) else 0

        risk_score = float(np.mean(y_score[top_idx])) if top_idx.size > 0 else 0.0
        anomaly_pred = 1 if risk_score >= alarm_thr else 0

        slo_recall_at_k = float("nan")
        if has_rt and rt_thr is not None:
            wt1 = windows[i + 1]
            if "rt" in wt1.columns:
                hi = wt1[wt1["rt"] > rt_thr]
                if not hi.empty:
                    hi_edges = set(zip(hi["um_encoded"].astype(int), hi["dm_encoded"].astype(int)))
                    captured_hi = sum(1 for e in top_edges if e in hi_edges)
                    slo_recall_at_k = float(captured_hi / max(1, len(hi_edges)))
                else:
                    slo_recall_at_k = 0.0

        rows.append(
            {
                "InputWindow": float(i),
                "TargetWindow": float(i + 1),
                "AUC": float(roc_auc_score(y_true, y_score)),
                "PR_AUC": float(average_precision_score(y_true, y_score)),
                "TriagePrecisionAtK": triage_precision_at_k,
                "TriageRecallNewAtK": triage_recall_new_at_k,
                "AnomalyTrue": float(anomaly_true),
                "AnomalyPred": float(anomaly_pred),
                "SLOProxyRecallAtK": slo_recall_at_k,
            }
        )

    if not rows:
        raise RuntimeError("No test rows produced.")

    y_true_alert = np.array([int(r["AnomalyTrue"]) for r in rows], dtype=int)
    y_pred_alert = np.array([int(r["AnomalyPred"]) for r in rows], dtype=int)
    alert_p, alert_r, alert_f1, _ = precision_recall_fscore_support(
        y_true_alert, y_pred_alert, average="binary", zero_division=0
    )

    metrics = ["AUC", "PR_AUC", "TriagePrecisionAtK", "TriageRecallNewAtK"]
    if has_rt:
        valid_slo = [r for r in rows if not np.isnan(r["SLOProxyRecallAtK"])]
        if valid_slo:
            metrics.append("SLOProxyRecallAtK")

    summary = {
        "Model": args.model,
        "Dataset": str(dataset_path),
        "Device": str(device),
        "TopK": int(args.top_k),
        "BestEpoch": int(best_epoch),
        "TrainingSec": train_sec,
        "AnomalyEdgeCountThreshold": edge_count_thr,
        "AnomalyChurnThreshold": churn_thr,
        "AlarmThreshold": alarm_thr,
        "HasRT": bool(has_rt),
        "AlertPrecision_Mean": float(alert_p),
        "AlertPrecision_Std": 0.0,
        "AlertRecall_Mean": float(alert_r),
        "AlertRecall_Std": 0.0,
        "AlertF1_Mean": float(alert_f1),
        "AlertF1_Std": 0.0,
    }
    summary.update(aggregate(rows, metrics))

    per_pair_csv = results_dir / "downstream_operational_proxy_per_pair.csv"
    pd.DataFrame(rows).to_csv(per_pair_csv, index=False)

    summary_json = results_dir / "downstream_operational_proxy_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = results_dir / "downstream_operational_proxy_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    report_md = results_dir / "downstream_operational_proxy_report.md"
    write_report(report_md, args, summary, has_rt, train_sec)

    print(f"Saved: {per_pair_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_md}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downstream operational proxy validation (anomaly triage + optional RT proxy)")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--model", type=str, default="GAT", choices=["GAT", "Diffusion", "DiffusionGAT", "Transformer", "TransformerGAT"])

    p.add_argument("--time-window-size", type=int, default=100)
    p.add_argument("--max-time", type=int, default=None)

    p.add_argument("--train-end-index", type=int, default=70)
    p.add_argument("--val-start-index", type=int, default=60)
    p.add_argument("--test-start-index", type=int, default=70)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--delta", type=float, default=0.001)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-5)

    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--candidate-neg-ratio", type=int, default=50)
    p.add_argument("--anomaly-percentile", type=float, default=90.0)
    p.add_argument("--alarm-percentile", type=float, default=90.0)
    p.add_argument("--slo-rt-percentile", type=float, default=95.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-cpu", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
