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
from sklearn.metrics import average_precision_score, roc_auc_score
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
TRANSFORMER_MODELS = {"transformer", "transformergat"}
ATTENTION_BYTES_PER_NODE_SQ = 32  # float32 * 4 heads * 2 (Q·K + softmax)


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
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"] - float(df["timestamp"].min())
    return df


def estimate_attention_bytes(num_nodes: int, num_layers: int = 2) -> int:
    return num_nodes * num_nodes * ATTENTION_BYTES_PER_NODE_SQ * num_layers


def maybe_downscale_for_transformer(df: pd.DataFrame, max_time: int, max_nodes: int, mem_limit_gb: float) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    if max_time > 0:
        out = out[out["timestamp"] < max_time].copy()

    if max_nodes > 0:
        node_counts = pd.concat([out["um"], out["dm"]]).value_counts()
        keep_nodes = set(node_counts.head(max_nodes).index.tolist())
        out = out[out["um"].isin(keep_nodes) & out["dm"].isin(keep_nodes)].copy()

    node_count = len(pd.concat([out["um"], out["dm"]]).unique())
    est_gb = estimate_attention_bytes(node_count) / (1024 ** 3)
    info = {
        "NodeCountAfterCap": float(node_count),
        "TransformerAttentionEstGB": float(est_gb),
        "MemLimitGB": float(mem_limit_gb),
    }

    if est_gb > mem_limit_gb and node_count > 1:
        safe_nodes = int(((mem_limit_gb * (1024 ** 3)) / (ATTENTION_BYTES_PER_NODE_SQ * 2)) ** 0.5)
        node_counts = pd.concat([out["um"], out["dm"]]).value_counts()
        keep_nodes = set(node_counts.head(max(2, safe_nodes)).index.tolist())
        out = out[out["um"].isin(keep_nodes) & out["dm"].isin(keep_nodes)].copy()
        node_count = len(pd.concat([out["um"], out["dm"]]).unique())
        est_gb = estimate_attention_bytes(node_count) / (1024 ** 3)
        info["NodeCountAfterCap"] = float(node_count)
        info["TransformerAttentionEstGB"] = float(est_gb)
        info["AutoDownscaled"] = 1.0
    else:
        info["AutoDownscaled"] = 0.0

    return out, info


def encode_nodes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    services = pd.concat([out["um"], out["dm"]]).unique()
    node_map = {node: i for i, node in enumerate(services)}
    out["um_encoded"] = out["um"].map(node_map)
    out["dm_encoded"] = out["dm"].map(node_map)
    return out, len(services)


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


def build_non_overlap_windows(df: pd.DataFrame, window_size: int) -> tuple[list[Data], list[int], list[int]]:
    max_time = int(df["timestamp"].max()) + window_size
    num_nodes = int(max(df["um_encoded"].max(), df["dm_encoded"].max()) + 1)

    windows = []
    starts = []
    ends = []
    for start in range(0, max_time, window_size):
        end = start + window_size
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        windows.append(create_graph(w, num_nodes))
        starts.append(start)
        ends.append(end)
    return windows, starts, ends


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
        raise ValueError(f"Unsupported model '{model_name}'.")
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


def evaluate_with_timing(model, model_name: str, graphs: list[Data], test_pairs: list[int], neg_ratio: int, seed: int) -> dict:
    model.eval()
    auc_vals = []
    pr_vals = []

    total_inference_sec = 0.0
    t_eval = time.perf_counter()

    with torch.no_grad():
        for i in test_pairs:
            g_t = graphs[i]
            g_t1 = graphs[i + 1]

            t_inf0 = time.perf_counter()
            z = model_forward_embeddings(model, model_name, g_t)
            total_inference_sec += time.perf_counter() - t_inf0

            pos_ei = g_t1.edge_index
            if pos_ei.numel() == 0:
                continue

            pos_count = int(pos_ei.size(1))
            neg_count = int(pos_count * neg_ratio)

            pos_scores = torch.sigmoid((z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=1)).cpu().numpy()
            forbidden = edge_tensor_to_keys(pos_ei, g_t1.num_nodes)
            neg_ei = sample_negative_edges_fast(g_t1.num_nodes, neg_count, forbidden, seed * 200000 + i, edge_index=pos_ei.detach().cpu()).to(z.device)
            neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1)).cpu().numpy()

            y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
            y_score = np.concatenate([pos_scores, neg_scores])

            auc_vals.append(float(roc_auc_score(y_true, y_score)))
            pr_vals.append(float(average_precision_score(y_true, y_score)))

    eval_sec = time.perf_counter() - t_eval

    return {
        "Pairs": float(len(auc_vals)),
        "AUC_Mean": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "PR_AUC_Mean": float(np.mean(pr_vals)) if pr_vals else float("nan"),
        "ModelInferenceSec": float(total_inference_sec),
        "EndToEndEvalSec": float(eval_sec),
    }


def run_model_runtime(model_name: str, graphs: list[Data], pair_indices: dict[str, list[int]], args: argparse.Namespace,
                      device: torch.device, seed: int) -> tuple[dict, list[dict]]:
    num_nodes = graphs[0].num_nodes

    x = torch.nn.Parameter(torch.randn(num_nodes, args.embedding_dim, device=device))
    model_graphs = []
    for g in graphs:
        gc = g.clone()
        gc.x = x
        model_graphs.append(gc.to(device))

    model = get_model(model_name, args.embedding_dim, num_nodes, device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [x],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    epoch_rows = []
    t_train0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t_epoch0 = time.perf_counter()
        model.train()

        for idx in pair_indices["train"]:
            optimizer.zero_grad()
            loss = train_pair_loss(model, model_name, model_graphs[idx], model_graphs[idx + 1], seed * 100000 + epoch * 1000 + idx)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [
                float(
                    train_pair_loss(
                        model,
                        model_name,
                        model_graphs[idx],
                        model_graphs[idx + 1],
                        seed * 100000 + epoch * 1000 + 50000 + idx,
                    ).item()
                )
                for idx in pair_indices["val"]
            ]

        avg_val = float(np.mean(val_losses))
        if avg_val < best_val * (1.0 - args.delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        epoch_sec = time.perf_counter() - t_epoch0
        epoch_rows.append({
            "Model": model_name,
            "Epoch": float(epoch),
            "EpochTrainSec": float(epoch_sec),
            "ValLoss": float(avg_val),
        })

        if epoch >= args.min_epochs and no_improve >= args.patience:
            break

    total_train_sec = time.perf_counter() - t_train0

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_stats = evaluate_with_timing(
        model,
        model_name,
        model_graphs,
        pair_indices["test"],
        neg_ratio=args.eval_neg_ratio,
        seed=seed,
    )

    epoch_secs = np.array([r["EpochTrainSec"] for r in epoch_rows], dtype=float)
    eval_overhead = max(0.0, eval_stats["EndToEndEvalSec"] - eval_stats["ModelInferenceSec"])

    summary = {
        "Model": model_name,
        "BestEpoch": float(best_epoch),
        "EpochsRun": float(len(epoch_rows)),
        "TrainingTimePerEpochMeanSec": float(np.mean(epoch_secs)) if len(epoch_secs) > 0 else float("nan"),
        "TrainingTimePerEpochStdSec": float(np.std(epoch_secs, ddof=0)) if len(epoch_secs) > 0 else float("nan"),
        "TotalTrainingSec": float(total_train_sec),
        **eval_stats,
        "EvalOverheadSec": float(eval_overhead),
    }
    return summary, epoch_rows


def build_report(summary_df: pd.DataFrame, common: dict, report_path: Path) -> None:
    lines = []
    lines.append("# Runtime Measurement Breakdown")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("Refine runtime reporting under the corrected rolling protocol by separating data, graph, training, and evaluation components.")
    lines.append("")
    lines.append("## 2. Shared Runtime Components")
    lines.append(f"- Data loading time: {common['DataLoadingSec']:.2f} sec")
    lines.append(f"- Graph construction time: {common['GraphConstructionSec']:.2f} sec")
    lines.append(f"- Node count after cap: {int(common['NodeCountAfterCap'])}")
    lines.append(f"- Transformer attention estimate: {common['TransformerAttentionEstGB']:.2f} GB")
    lines.append("")
    lines.append("## 3. Per-Model Runtime Breakdown")

    for _, r in summary_df.iterrows():
        lines.append(f"### {r['Model']}")
        lines.append(f"- Training time per epoch (mean +- std): {r['TrainingTimePerEpochMeanSec']:.3f} +- {r['TrainingTimePerEpochStdSec']:.3f} sec")
        lines.append(f"- Total training time: {r['TotalTrainingSec']:.2f} sec")
        lines.append(f"- Model inference time (test loop forward only): {r['ModelInferenceSec']:.2f} sec")
        lines.append(f"- End-to-end evaluation time: {r['EndToEndEvalSec']:.2f} sec")
        lines.append(f"- Evaluation overhead (sampling + scoring + metrics): {r['EvalOverheadSec']:.2f} sec")
        lines.append(f"- AUC / PR-AUC: {r['AUC_Mean']:.4f} / {r['PR_AUC_Mean']:.4f}")
        lines.append("")

    lines.append("## 4. Bottleneck Interpretation")
    for _, r in summary_df.iterrows():
        neural = float(r["TotalTrainingSec"] + r["ModelInferenceSec"])
        pipeline = float(common["GraphConstructionSec"] + r["EvalOverheadSec"])
        if pipeline > neural:
            verdict = "Pipeline-dominant"
        else:
            verdict = "Model-dominant"
        lines.append(
            f"- {r['Model']}: pipeline={pipeline:.2f}s vs neural={neural:.2f}s -> {verdict}"
        )

    lines.append("")
    lines.append("## 5. Note")
    lines.append("This is a precision runtime redo only; it does not require rerunning every historical dataset/window combination.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Runtime breakdown under corrected G_t -> E_(t+1) protocol.")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--train-end-index", type=int, default=69)
    p.add_argument("--val-start-index", type=int, default=60)
    p.add_argument("--test-start-index", type=int, default=70)

    p.add_argument("--models", type=str, default="GAT,Diffusion,DiffusionGAT,Transformer,TransformerGAT")
    p.add_argument("--eval-neg-ratio", type=int, default=10)

    p.add_argument("--max-time", type=int, default=10000)
    p.add_argument("--max-nodes", type=int, default=2000)
    p.add_argument("--mem-limit-gb", type=float, default=4.0)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--force-cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    t_load0 = time.perf_counter()
    df = load_calls(Path(args.dataset_path))
    df, cap_info = maybe_downscale_for_transformer(
        df,
        max_time=args.max_time,
        max_nodes=args.max_nodes,
        mem_limit_gb=args.mem_limit_gb,
    )
    data_loading_sec = time.perf_counter() - t_load0

    t_graph0 = time.perf_counter()
    df, _ = encode_nodes(df)
    graphs, starts, ends = build_non_overlap_windows(df, args.window_size)
    graph_construction_sec = time.perf_counter() - t_graph0

    if len(graphs) <= args.train_end_index + 1 or args.test_start_index >= len(graphs) - 1:
        raise ValueError("Not enough windows for requested train/val/test split after capping.")

    pair_indices = {
        "train": [i for i in range(0, args.train_end_index) if i < args.val_start_index],
        "val": [i for i in range(0, args.train_end_index) if i >= args.val_start_index],
        "test": list(range(args.test_start_index, len(graphs) - 1)),
    }

    summary_rows = []
    epoch_rows = []

    for model_name in models:
        print(f"Running model: {model_name}")
        s_row, e_rows = run_model_runtime(model_name, graphs, pair_indices, args, device, args.seed)
        summary_rows.append(s_row)
        epoch_rows.extend(e_rows)

    summary_df = pd.DataFrame(summary_rows)
    epoch_df = pd.DataFrame(epoch_rows)

    summary_df.insert(1, "DataLoadingSec", float(data_loading_sec))
    summary_df.insert(2, "GraphConstructionSec", float(graph_construction_sec))
    for key in ["NodeCountAfterCap", "TransformerAttentionEstGB", "MemLimitGB", "AutoDownscaled"]:
        summary_df[key] = float(cap_info[key])

    summary_path = results_dir / "runtime_breakdown_summary.csv"
    epoch_path = results_dir / "runtime_breakdown_epoch_times.csv"
    json_path = results_dir / "runtime_breakdown_summary.json"
    report_path = results_dir / "runtime_breakdown_report.md"

    summary_df.to_csv(summary_path, index=False)
    epoch_df.to_csv(epoch_path, index=False)

    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "dataset_path": args.dataset_path,
                    "models": models,
                    "window_size": args.window_size,
                    "train_end_index": args.train_end_index,
                    "val_start_index": args.val_start_index,
                    "test_start_index": args.test_start_index,
                    "eval_neg_ratio": args.eval_neg_ratio,
                    "max_time": args.max_time,
                    "max_nodes": args.max_nodes,
                    "mem_limit_gb": args.mem_limit_gb,
                    "seed": args.seed,
                },
                "shared_timing": {
                    "data_loading_sec": float(data_loading_sec),
                    "graph_construction_sec": float(graph_construction_sec),
                    **cap_info,
                },
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    build_report(
        summary_df,
        {
            "DataLoadingSec": float(data_loading_sec),
            "GraphConstructionSec": float(graph_construction_sec),
            **cap_info,
        },
        report_path,
    )

    print("Saved:", summary_path)
    print("Saved:", epoch_path)
    print("Saved:", json_path)
    print("Saved:", report_path)
    print(summary_df[["Model", "DataLoadingSec", "GraphConstructionSec", "TrainingTimePerEpochMeanSec", "TotalTrainingSec", "ModelInferenceSec", "EndToEndEvalSec"]])


if __name__ == "__main__":
    main()
