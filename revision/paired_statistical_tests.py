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
from scipy import stats
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
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
        raise ValueError(f"Missing required columns after normalization for {csv_path}: {missing}")

    df = df[REQUIRED].copy()
    df = df.dropna()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df[df["timestamp"] >= 0]
    df["um"] = df["um"].astype(str)
    df["dm"] = df["dm"].astype(str)
    return df.sort_values("timestamp").reset_index(drop=True)


def rebase_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = out["timestamp"] - float(out["timestamp"].min())
    return out


def encode_nodes(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    services = pd.concat([out["um"], out["dm"]]).unique()
    node_map = {node: idx for idx, node in enumerate(services)}
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
    neg_ei = sample_negative_edges_fast(g_t1.num_nodes, int(pos_ei.size(1)), forbidden, seed, edge_index=pos_ei.detach().cpu()).to(z.device)
    neg_scores = torch.sigmoid((z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=1))

    return F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy(
        neg_scores, torch.zeros_like(neg_scores)
    )


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k_eval = int(max(1, min(k, len(y_score))))
    top_idx = np.argsort(y_score)[::-1][:k_eval]
    return float(np.sum(y_true[top_idx])) / float(k_eval)


def evaluate_pair(model, model_name: str, g_t: Data, g_t1: Data, neg_ratio: int, top_k: int, seed: int) -> dict | None:
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
            "Accuracy": float(accuracy_score(y_true, y_hat)),
            "Precision": float(precision),
            "Recall": float(recall),
            "PR_AUC": float(average_precision_score(y_true, y_score)),
            "PrecisionAt100": float(precision_at_k(y_true, y_score, top_k)),
            "PosEdges": float(pos_count),
            "NegEdges": float(neg_count),
        }


def build_non_overlap_windows(df: pd.DataFrame, window_size: int, max_time: int) -> tuple[list[Data], list[int], list[int]]:
    windows = []
    starts = []
    ends = []
    num_nodes = int(max(df["um_encoded"].max(), df["dm_encoded"].max()) + 1)
    for start in range(0, max_time, window_size):
        end = start + window_size
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        windows.append(create_graph(w, num_nodes))
        starts.append(start)
        ends.append(end)
    return windows, starts, ends


def build_overlap_windows(df: pd.DataFrame, window_size: int, overlap: float, max_time: int) -> tuple[list[Data], list[int], list[int]]:
    step = int(round(window_size * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("Window step must be positive.")

    windows = []
    starts = []
    ends = []
    num_nodes = int(max(df["um_encoded"].max(), df["dm_encoded"].max()) + 1)
    for start in range(0, max_time, step):
        end = start + window_size
        w = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        windows.append(create_graph(w, num_nodes))
        starts.append(start)
        ends.append(end)
    return windows, starts, ends


def attach_node_features(
    graphs: list[Data], num_nodes: int, embedding_dim: int, device: torch.device
) -> tuple[list[Data], torch.nn.Parameter]:
    x = torch.nn.Parameter(torch.randn(num_nodes, embedding_dim, device=device))
    for graph in graphs:
        graph.x = x
    return [graph.to(device) for graph in graphs], x


def train_and_eval(
    graphs: list[Data],
    starts: list[int],
    ends: list[int],
    pair_indices: dict[str, list[int]],
    model_name: str,
    seed: int,
    embedding_dim: int,
    lr: float,
    weight_decay: float,
    epochs: int,
    min_epochs: int,
    patience: int,
    delta: float,
    eval_neg_ratio: int,
    top_k: int,
    device: torch.device,
    meta: dict[str, str],
) -> tuple[list[dict], dict[str, float]]:
    num_nodes = graphs[0].num_nodes if graphs else 0
    graphs, node_features = attach_node_features(graphs, num_nodes, embedding_dim, device)

    model = get_model(model_name, embedding_dim, num_nodes, device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [node_features],
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in pair_indices["train"]:
            optimizer.zero_grad()
            loss = train_pair_loss(model, model_name, graphs[idx], graphs[idx + 1], seed * 100000 + epoch * 1000 + idx)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_losses = [
                float(
                    train_pair_loss(
                        model,
                        model_name,
                        graphs[idx],
                        graphs[idx + 1],
                        seed * 100000 + epoch * 1000 + 50000 + idx,
                    ).item()
                )
                for idx in pair_indices["val"]
            ]

        avg_val = float(np.mean(val_losses))
        if avg_val < best_val * (1.0 - delta):
            best_val = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch >= min_epochs and no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_time = float(time.time() - t0)
    rows = []
    eval_start = time.time()
    for idx in pair_indices["test"]:
        row = evaluate_pair(
            model,
            model_name,
            graphs[idx],
            graphs[idx + 1],
            neg_ratio=eval_neg_ratio,
            top_k=top_k,
            seed=seed * 200000 + idx,
        )
        if row is None:
            continue
        rows.append(
            {
                **meta,
                "Seed": seed,
                "InputGraph": idx,
                "TargetGraph": idx + 1,
                "InputStart": starts[idx],
                "InputEnd": ends[idx],
                "TargetStart": starts[idx + 1],
                "TargetEnd": ends[idx + 1],
                **row,
            }
        )

    return rows, {
        "BestEpoch": float(best_epoch),
        "TrainTimeSec": train_time,
        "EvalTimeSec": float(time.time() - eval_start),
    }


def paired_effect_size(diff: np.ndarray) -> float:
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd == 0.0:
        return float("inf") if float(np.mean(diff)) != 0.0 else 0.0
    return float(np.mean(diff) / sd)


def run_paired_test(left: np.ndarray, right: np.ndarray) -> dict[str, float | str]:
    diff = right - left
    t_res = stats.ttest_rel(right, left, nan_policy="omit")

    nonzero_diff = diff[np.abs(diff) > 0]
    if nonzero_diff.size == 0:
        w_stat = 0.0
        w_p = 1.0
    else:
        w_res = stats.wilcoxon(nonzero_diff, zero_method="wilcox", alternative="two-sided", correction=False)
        w_stat = float(w_res.statistic)
        w_p = float(w_res.pvalue)

    return {
        "N": int(diff.size),
        "MeanLeft": float(np.mean(left)),
        "MeanRight": float(np.mean(right)),
        "MeanDiff": float(np.mean(diff)),
        "MedianDiff": float(np.median(diff)),
        "PairedTStat": float(t_res.statistic),
        "PairedTPValue": float(t_res.pvalue),
        "WilcoxonStat": w_stat,
        "WilcoxonPValue": w_p,
        "CohenDz": paired_effect_size(diff),
    }


def compute_paired_summary(
    rows_df: pd.DataFrame,
    comparison_name: str,
    left_label: str,
    right_label: str,
    scenario: str,
    unit_cols: list[str],
    metrics: list[str],
    left_filter: dict[str, str],
    right_filter: dict[str, str],
) -> list[dict]:
    left_df = rows_df.copy()
    right_df = rows_df.copy()
    for key, value in left_filter.items():
        left_df = left_df[left_df[key] == value]
    for key, value in right_filter.items():
        right_df = right_df[right_df[key] == value]

    cols = unit_cols + metrics
    left_df = left_df[cols].rename(columns={m: f"{m}_left" for m in metrics})
    right_df = right_df[cols].rename(columns={m: f"{m}_right" for m in metrics})
    merged = left_df.merge(right_df, on=unit_cols, how="inner")

    out = []
    for metric in metrics:
        left = merged[f"{metric}_left"].to_numpy(dtype=float)
        right = merged[f"{metric}_right"].to_numpy(dtype=float)
        stats_row = run_paired_test(left, right)
        out.append(
            {
                "Scenario": scenario,
                "Comparison": comparison_name,
                "Left": left_label,
                "Right": right_label,
                "Metric": metric,
                "UnitsMatched": int(len(merged)),
                **stats_row,
            }
        )
    return out


def estimate_attention_bytes(num_nodes: int, num_layers: int = 2) -> int:
    return num_nodes * num_nodes * ATTENTION_BYTES_PER_NODE_SQ * num_layers


def preflight_check_and_downscale(df: pd.DataFrame, max_time: int, max_nodes: int,
                                   mem_limit_bytes: int, models: list[str]) -> pd.DataFrame:
    if max_time > 0:
        df = df[df["timestamp"] < max_time].copy()

    if max_nodes > 0:
        node_counts = pd.concat([df["um"], df["dm"]]).value_counts()
        keep_nodes = set(node_counts.head(max_nodes).index.tolist())
        df = df[df["um"].isin(keep_nodes) & df["dm"].isin(keep_nodes)].copy()

    has_transformer = any(m.lower() in TRANSFORMER_MODELS for m in models)
    if has_transformer:
        current_nodes = len(pd.concat([df["um"], df["dm"]]).unique())
        est_bytes = estimate_attention_bytes(current_nodes)
        if est_bytes > mem_limit_bytes:
            safe_nodes = int((mem_limit_bytes / (ATTENTION_BYTES_PER_NODE_SQ * 2)) ** 0.5)
            print(f"[preflight] Transformer attention would need ~{est_bytes / 1e9:.1f} GB "
                  f"for {current_nodes} nodes; auto-downscaling to {safe_nodes} nodes "
                  f"(limit {mem_limit_bytes / 1e9:.1f} GB).")
            node_counts = pd.concat([df["um"], df["dm"]]).value_counts()
            keep_nodes = set(node_counts.head(safe_nodes).index.tolist())
            df = df[df["um"].isin(keep_nodes) & df["dm"].isin(keep_nodes)].copy()
            current_nodes = len(pd.concat([df["um"], df["dm"]]).unique())
            print(f"[preflight] After downscale: {current_nodes} nodes, "
                  f"~{estimate_attention_bytes(current_nodes) / 1e9:.2f} GB estimated.")
        else:
            print(f"[preflight] Transformer memory OK: {current_nodes} nodes, "
                  f"~{est_bytes / 1e9:.2f} GB estimated.")
    return df


def run_model_comparisons(args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = rebase_timestamp(load_calls(Path(args.alibaba2022_path)))
    models = [m.strip() for m in args.model_comparison_models.split(",") if m.strip()]
    df = preflight_check_and_downscale(
        df,
        max_time=args.model_max_time,
        max_nodes=args.model_max_nodes,
        mem_limit_bytes=args.mem_limit_gb * (1024 ** 3),
        models=models,
    )

    df, num_nodes = encode_nodes(df)
    max_time = int(df["timestamp"].max()) + args.window_size
    graphs, starts, ends = build_non_overlap_windows(df, args.window_size, max_time)

    if len(graphs) <= args.train_end_index + 1 or args.test_start_index >= len(graphs) - 1:
        raise ValueError("Alibaba 2022 does not have enough windows for the requested rolling split.")

    train_pairs = [idx for idx in range(0, args.train_end_index) if idx < args.val_start_index]
    val_pairs = [idx for idx in range(0, args.train_end_index) if idx >= args.val_start_index]
    test_pairs = list(range(args.test_start_index, len(graphs) - 1))
    pair_indices = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    all_rows = []
    meta_rows = []
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for seed in seeds:
        set_seed(seed)
        for model_name in models:
            rows, meta = train_and_eval(
                graphs=[g.clone() for g in graphs],
                starts=starts,
                ends=ends,
                pair_indices=pair_indices,
                model_name=model_name,
                seed=seed,
                embedding_dim=args.embedding_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                delta=args.delta,
                eval_neg_ratio=args.model_eval_neg_ratio,
                top_k=args.top_k,
                device=device,
                meta={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": model_name},
            )
            all_rows.extend(rows)
            meta_rows.append({"Scenario": "model", "Dataset": "Alibaba2022", "Condition": model_name, "Seed": seed, **meta})

    rows_df = pd.DataFrame(all_rows)
    summary_rows = []
    summary_rows.extend(
        compute_paired_summary(
            rows_df,
            comparison_name="GAT_vs_DiffusionGAT",
            left_label="GAT",
            right_label="DiffusionGAT",
            scenario="model",
            unit_cols=["Dataset", "Seed", "TargetGraph"],
            metrics=["AUC", "F1", "Accuracy"],
            left_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "GAT"},
            right_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "DiffusionGAT"},
        )
    )
    summary_rows.extend(
        compute_paired_summary(
            rows_df,
            comparison_name="Diffusion_vs_DiffusionGAT",
            left_label="Diffusion",
            right_label="DiffusionGAT",
            scenario="model",
            unit_cols=["Dataset", "Seed", "TargetGraph"],
            metrics=["AUC", "F1", "Accuracy"],
            left_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "Diffusion"},
            right_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "DiffusionGAT"},
        )
    )
    summary_rows.extend(
        compute_paired_summary(
            rows_df,
            comparison_name="Transformer_vs_TransformerGAT",
            left_label="Transformer",
            right_label="TransformerGAT",
            scenario="model",
            unit_cols=["Dataset", "Seed", "TargetGraph"],
            metrics=["AUC", "F1", "Accuracy"],
            left_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "Transformer"},
            right_filter={"Scenario": "model", "Dataset": "Alibaba2022", "Condition": "TransformerGAT"},
        )
    )
    return rows_df, pd.DataFrame(summary_rows)


def run_overlap_comparison(args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = rebase_timestamp(load_calls(Path(args.alibaba2022_path)))
    df, _ = encode_nodes(df)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_rows = []
    meta_rows = []

    settings = [
        ("100ms_non_overlap", 0.0),
        ("100ms_overlap_50pct", 0.5),
    ]

    for seed in seeds:
        set_seed(seed)
        for name, overlap in settings:
            graphs, starts, ends = build_overlap_windows(df, args.window_size, overlap, args.overlap_max_time)
            train_pairs = [
                idx
                for idx in range(len(graphs) - 1)
                if ends[idx] <= args.overlap_train_time_end and ends[idx + 1] <= args.overlap_train_time_end
            ]
            test_pairs = [
                idx
                for idx in range(len(graphs) - 1)
                if starts[idx] >= args.overlap_test_time_start and starts[idx + 1] >= args.overlap_test_time_start
            ]
            val_count = max(1, int(round(len(train_pairs) * args.overlap_val_ratio)))
            pair_indices = {
                "train": train_pairs[:-val_count],
                "val": train_pairs[-val_count:],
                "test": test_pairs,
            }

            rows, meta = train_and_eval(
                graphs=graphs,
                starts=starts,
                ends=ends,
                pair_indices=pair_indices,
                model_name=args.overlap_model,
                seed=seed,
                embedding_dim=args.embedding_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                delta=args.delta,
                eval_neg_ratio=args.overlap_eval_neg_ratio,
                top_k=args.top_k,
                device=device,
                meta={"Scenario": "overlap", "Dataset": "Alibaba2022", "Condition": name},
            )
            all_rows.extend(rows)
            meta_rows.append({"Scenario": "overlap", "Dataset": "Alibaba2022", "Condition": name, "Seed": seed, **meta})

    rows_df = pd.DataFrame(all_rows)
    summary_rows = compute_paired_summary(
        rows_df,
        comparison_name="100ms_non_overlap_vs_100ms_overlap_50pct",
        left_label="100ms_non_overlap",
        right_label="100ms_overlap_50pct",
        scenario="overlap",
        unit_cols=["Dataset", "Seed", "TargetStart"],
        metrics=["AUC", "F1", "PR_AUC", "PrecisionAt100"],
        left_filter={"Scenario": "overlap", "Dataset": "Alibaba2022", "Condition": "100ms_non_overlap"},
        right_filter={"Scenario": "overlap", "Dataset": "Alibaba2022", "Condition": "100ms_overlap_50pct"},
    )
    return rows_df, pd.DataFrame(summary_rows)


def run_dataset_note(args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    datasets = [
        ("Alibaba2022", Path(args.alibaba2022_path)),
        ("Huawei2021", Path(args.huawei_path)),
    ]

    all_rows = []
    meta_rows = []
    for dataset_name, dataset_path in datasets:
        df = rebase_timestamp(load_calls(dataset_path))
        df, _ = encode_nodes(df)
        max_time = int(df["timestamp"].max()) + args.window_size if args.dataset_max_time <= 0 else args.dataset_max_time
        graphs, starts, ends = build_non_overlap_windows(df, args.window_size, max_time)
        if len(graphs) <= args.train_end_index + 1 or args.test_start_index >= len(graphs) - 1:
            raise ValueError(f"{dataset_name} does not have enough windows for the requested rolling split.")

        pair_indices = {
            "train": [idx for idx in range(0, args.train_end_index) if idx < args.val_start_index],
            "val": [idx for idx in range(0, args.train_end_index) if idx >= args.val_start_index],
            "test": list(range(args.test_start_index, len(graphs) - 1)),
        }
        for seed in seeds:
            set_seed(seed)
            rows, meta = train_and_eval(
                graphs=[g.clone() for g in graphs],
                starts=starts,
                ends=ends,
                pair_indices=pair_indices,
                model_name=args.dataset_note_model,
                seed=seed,
                embedding_dim=args.embedding_dim,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                min_epochs=args.min_epochs,
                patience=args.patience,
                delta=args.delta,
                eval_neg_ratio=args.dataset_eval_neg_ratio,
                top_k=args.top_k,
                device=device,
                meta={"Scenario": "dataset", "Dataset": dataset_name, "Condition": args.dataset_note_model},
            )
            all_rows.extend(rows)
            meta_rows.append({"Scenario": "dataset", "Dataset": dataset_name, "Condition": args.dataset_note_model, "Seed": seed, **meta})

    rows_df = pd.DataFrame(all_rows)
    note_df = pd.DataFrame(
        [
            {
                "Scenario": "dataset",
                "Comparison": "Alibaba2022_vs_Huawei2021",
                "Status": "not_paired_tested",
                "Reason": (
                    "Dataset identity is part of the experimental unit, so Alibaba-vs-Huawei observations are not the "
                    "same seed-window-dataset units required for paired inference. Descriptive summaries are reported instead."
                ),
            }
        ]
    )
    return rows_df, note_df


def write_report(report_path: Path, args: argparse.Namespace, tests_df: pd.DataFrame, dataset_note_df: pd.DataFrame, raw_df: pd.DataFrame, stage_status: dict[str, str] | None = None) -> None:
    lines = []
    lines.append("# Matched Statistical Tests Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment reruns corrected evaluations and computes paired statistical tests only on matched experimental units."
    )
    lines.append("")
    lines.append("## 2. Matched Unit Definition")
    lines.append("- Primary unit for paired tests: seed + dataset + test target window")
    lines.append("- Model comparisons use the same Alibaba 2022 rolling windows under the corrected G_t -> E_(t+1) protocol")
    lines.append("- Overlap comparison matches on the same target window start times; overlap rows without a non-overlap target match are excluded")
    lines.append("- Alibaba vs Huawei is not forced into a paired test because dataset identity changes the unit itself")
    lines.append("")
    lines.append("## 3. Configuration")
    lines.append(f"- Seeds: {args.seeds}")
    lines.append(f"- Rolling window size: {args.window_size}")
    lines.append(f"- Rolling split: train_end_index={args.train_end_index}, val_start_index={args.val_start_index}, test_start_index={args.test_start_index}")
    lines.append(f"- Model-comparison negative ratio: 1:{args.model_eval_neg_ratio}")
    lines.append(f"- Model-comparison max_time cap: {args.model_max_time}")
    lines.append(f"- Model-comparison max_nodes cap: {args.model_max_nodes}")
    lines.append(f"- Overlap negative ratio: 1:{args.overlap_eval_neg_ratio}")
    lines.append(f"- Dataset descriptive negative ratio: 1:{args.dataset_eval_neg_ratio}")
    lines.append("")

    lines.append("## 4. Paired Test Summary")
    if tests_df.empty or "Comparison" not in tests_df.columns:
        lines.append("No test results available (all stages may have failed).")
    else:
        for comparison in tests_df["Comparison"].drop_duplicates().tolist():
            lines.append(f"### {comparison}")
            sub = tests_df[tests_df["Comparison"] == comparison]
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['Metric']}: n={int(row['UnitsMatched'])}, mean diff ({row['Right']} - {row['Left']})={row['MeanDiff']:.6f}, "
                    f"paired t p={row['PairedTPValue']:.4g}, Wilcoxon p={row['WilcoxonPValue']:.4g}, Cohen's d_z={row['CohenDz']:.4f}"
                )
            lines.append("")

    lines.append("## 5. Dataset Comparison Note")
    for _, row in dataset_note_df.iterrows():
        lines.append(f"- {row['Comparison']}: {row['Reason']}")

    lines.append("")
    lines.append("## 6. Descriptive Dataset Means")
    dataset_rows = raw_df[raw_df["Scenario"] == "dataset"]
    if len(dataset_rows) > 0:
        for dataset_name in dataset_rows["Dataset"].drop_duplicates().tolist():
            sub = dataset_rows[dataset_rows["Dataset"] == dataset_name]
            lines.append(
                f"- {dataset_name}: AUC={sub['AUC'].mean():.4f}, F1={sub['F1'].mean():.4f}, Accuracy={sub['Accuracy'].mean():.4f} "
                f"over {len(sub)} matched seed-window evaluations"
            )
    else:
        lines.append("- No dataset descriptive rows were generated.")

    lines.append("")
    lines.append("## 7. Outputs Verified")
    output_files = [
        report_path.parent / "paired_statistical_tests_raw_units.csv",
        report_path.parent / "paired_statistical_tests_summary.csv",
        report_path.parent / "paired_statistical_tests_summary.json",
        report_path,
    ]
    for f in output_files:
        exists = f.exists() or f == report_path  # report is being written now
        lines.append(f"- `{f.name}`: {'present' if exists else 'MISSING'}")
    if stage_status:
        lines.append("")
        lines.append("### Stage Execution Status")
        for stage, status in stage_status.items():
            icon = 'PASS' if status == 'ok' else 'FAIL'
            lines.append(f"- {stage}: [{icon}] {status}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Redo paired statistical tests using matched experimental units.")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--comparisons", type=str, default="model,overlap,dataset")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")

    p.add_argument("--alibaba2022-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--huawei-path", type=str, default="Data/Huawei/status_1min_20210411.csv")

    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--train-end-index", type=int, default=69)
    p.add_argument("--val-start-index", type=int, default=60)
    p.add_argument("--test-start-index", type=int, default=70)

    p.add_argument("--model-comparison-models", type=str, default="GAT,Diffusion,DiffusionGAT,Transformer,TransformerGAT")
    p.add_argument("--model-eval-neg-ratio", type=int, default=1)
    p.add_argument("--model-max-time", type=int, default=10000)
    p.add_argument("--model-max-nodes", type=int, default=2000)
    p.add_argument("--mem-limit-gb", type=float, default=4.0)

    p.add_argument("--overlap-model", type=str, default="DiffusionGAT")
    p.add_argument("--overlap-train-time-end", type=int, default=7000)
    p.add_argument("--overlap-test-time-start", type=int, default=7000)
    p.add_argument("--overlap-max-time", type=int, default=10000)
    p.add_argument("--overlap-val-ratio", type=float, default=0.2)
    p.add_argument("--overlap-eval-neg-ratio", type=int, default=10)

    p.add_argument("--dataset-note-model", type=str, default="GAT")
    p.add_argument("--dataset-eval-neg-ratio", type=int, default=10)
    p.add_argument("--dataset-max-time", type=int, default=-1)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--delta", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--force-cpu", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {name.strip().lower() for name in args.comparisons.split(",") if name.strip()}
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    raw_parts = []
    test_parts = []
    dataset_note_parts = []
    stage_status: dict[str, str] = {}

    if "model" in comparisons:
        try:
            print("\n=== Stage: model comparisons ===")
            model_raw, model_tests = run_model_comparisons(args, device)
            raw_parts.append(model_raw)
            test_parts.append(model_tests)
            stage_status["model"] = "ok"
        except Exception as exc:
            print(f"[stage:model] FAILED: {exc}")
            stage_status["model"] = f"failed: {exc}"

    if "overlap" in comparisons:
        try:
            print("\n=== Stage: overlap comparison ===")
            overlap_raw, overlap_tests = run_overlap_comparison(args, device)
            raw_parts.append(overlap_raw)
            test_parts.append(overlap_tests)
            stage_status["overlap"] = "ok"
        except Exception as exc:
            print(f"[stage:overlap] FAILED: {exc}")
            stage_status["overlap"] = f"failed: {exc}"

    if "dataset" in comparisons:
        try:
            print("\n=== Stage: dataset descriptive ===")
            dataset_raw, dataset_note = run_dataset_note(args, device)
            raw_parts.append(dataset_raw)
            dataset_note_parts.append(dataset_note)
            stage_status["dataset"] = "ok"
        except Exception as exc:
            print(f"[stage:dataset] FAILED: {exc}")
            stage_status["dataset"] = f"failed: {exc}"

    raw_df = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    tests_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()
    dataset_note_df = pd.concat(dataset_note_parts, ignore_index=True) if dataset_note_parts else pd.DataFrame()

    raw_path = results_dir / "paired_statistical_tests_raw_units.csv"
    tests_path = results_dir / "paired_statistical_tests_summary.csv"
    json_path = results_dir / "paired_statistical_tests_summary.json"
    report_path = results_dir / "paired_statistical_tests_report.md"

    raw_df.to_csv(raw_path, index=False)
    tests_df.to_csv(tests_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "comparisons": sorted(comparisons),
                    "seeds": args.seeds,
                    "window_size": args.window_size,
                    "train_end_index": args.train_end_index,
                    "val_start_index": args.val_start_index,
                    "test_start_index": args.test_start_index,
                    "model_eval_neg_ratio": args.model_eval_neg_ratio,
                    "model_max_time": args.model_max_time,
                    "model_max_nodes": args.model_max_nodes,
                    "mem_limit_gb": args.mem_limit_gb,
                    "overlap_eval_neg_ratio": args.overlap_eval_neg_ratio,
                    "dataset_eval_neg_ratio": args.dataset_eval_neg_ratio,
                },
                "stage_status": stage_status,
                "tests": tests_df.to_dict(orient="records"),
                "dataset_note": dataset_note_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(report_path, args, tests_df, dataset_note_df, raw_df, stage_status)

    print("\n=== Stage Status ===")
    for stage, status in stage_status.items():
        print(f"  {stage}: {status}")
    print("Saved:", raw_path)
    print("Saved:", tests_path)
    print("Saved:", json_path)
    print("Saved:", report_path)
    if not tests_df.empty:
        print(tests_df[["Comparison", "Metric", "UnitsMatched", "MeanDiff", "PairedTPValue", "WilcoxonPValue", "CohenDz"]])
    if not dataset_note_df.empty:
        print(dataset_note_df)


if __name__ == "__main__":
    main()