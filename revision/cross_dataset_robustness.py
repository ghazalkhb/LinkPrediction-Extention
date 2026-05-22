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


REQUIRED = ["timestamp", "um", "dm"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_calls(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    cols = set(header.columns.tolist())

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
    return df.reset_index(drop=True)


def rebase_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t0 = float(d["timestamp"].min())
    d["timestamp"] = d["timestamp"] - t0
    return d


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


def pair_train_loss(model, model_name: str, g_t: Data, g_t1: Data, seed: int) -> torch.Tensor:
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


def evaluate_pair_ratio(model, model_name: str, g_t: Data, g_t1: Data, neg_ratio: int, seed: int):
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
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "Accuracy": float(accuracy_score(y_true, y_hat)),
            "PosEdges": float(pos_count),
            "NegEdges": float(neg_count),
        }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {"Pairs": float(len(rows))}
    metrics = ["AUC", "Precision", "Recall", "F1", "Accuracy"]
    for m in metrics:
        vals = np.array([r[m] for r in rows], dtype=float)
        out[f"{m}_Mean"] = float(vals.mean())
        out[f"{m}_Std"] = float(vals.std(ddof=0))
    out["AvgPosEdges"] = float(np.mean([r["PosEdges"] for r in rows]))
    out["AvgNegEdges"] = float(np.mean([r["NegEdges"] for r in rows]))
    return out


def run_dataset(dataset_name: str, csv_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    df = load_calls(csv_path)
    df = rebase_timestamp(df)

    all_nodes = pd.concat([df["um"], df["dm"]]).unique()
    node_map = {node: i for i, node in enumerate(all_nodes)}
    df["um_encoded"] = df["um"].map(node_map)
    df["dm_encoded"] = df["dm"].map(node_map)

    if args.max_time <= 0:
        max_time = int(df["timestamp"].max()) + args.window_size
    else:
        max_time = args.max_time
    windows = create_time_windows(df, args.window_size, max_time)
    graphs = [create_graph(w, len(all_nodes)) for w in windows]

    x = torch.nn.Parameter(torch.randn(len(all_nodes), args.embedding_dim, device=device))
    for g in graphs:
        g.x = x
    graphs = [g.to(device) for g in graphs]

    if len(graphs) <= args.train_end_index + 1:
        raise ValueError(f"{dataset_name}: not enough windows for train_end_index={args.train_end_index}")

    all_train = list(range(0, args.train_end_index))
    val_pairs = [i for i in all_train if i >= args.val_start_index]
    train_pairs = [i for i in all_train if i < args.val_start_index]
    if not train_pairs or not val_pairs:
        raise ValueError(f"{dataset_name}: invalid train/val split")

    model = get_model(args.model, args.embedding_dim, len(all_nodes), device)
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
        tr_losses = []
        for i in train_pairs:
            optimizer.zero_grad()
            loss = pair_train_loss(model, args.model, graphs[i], graphs[i + 1], args.seed * 100000 + epoch * 1000 + i)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            v_losses = [
                float(
                    pair_train_loss(
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

    test_last = len(graphs) - 2
    test_pairs_all = list(range(args.test_start_index, test_last + 1))

    ratios = [int(x.strip()) for x in args.eval_ratios.split(",") if x.strip()]
    per_ratio_summary = []

    for ratio in ratios:
        rows_ratio = []
        for i in test_pairs_all:
            row = evaluate_pair_ratio(
                model,
                args.model,
                graphs[i],
                graphs[i + 1],
                neg_ratio=ratio,
                seed=args.seed * 200000 + ratio * 10000 + i,
            )
            if row is None:
                continue
            row["Dataset"] = dataset_name
            row["NegRatio"] = float(ratio)
            row["InputGraph"] = float(i)
            row["TargetGraph"] = float(i + 1)
            rows_ratio.append(row)
            if ratio >= args.high_ratio_threshold and args.high_ratio_max_pairs > 0:
                if len(rows_ratio) >= args.high_ratio_max_pairs:
                    break

        if not rows_ratio:
            continue

        s = summarize(rows_ratio)
        s.update(
            {
                "Dataset": dataset_name,
                "NegRatio": float(ratio),
                "NumWindows": float(len(graphs)),
                "TrainPairs": float(len(train_pairs)),
                "ValPairs": float(len(val_pairs)),
                "TestPairsUsed": float(len(rows_ratio)),
                "BestEpoch": float(best_epoch),
                "TrainTimeSec": train_sec,
            }
        )
        per_ratio_summary.append(s)

    meta = {
        "Dataset": dataset_name,
        "NumRows": int(len(df)),
        "NumServices": int(len(all_nodes)),
        "NumWindows": int(len(graphs)),
        "TrainPairs": int(len(train_pairs)),
        "ValPairs": int(len(val_pairs)),
        "BestEpoch": int(best_epoch),
        "TrainTimeSec": train_sec,
    }

    return per_ratio_summary, meta


def write_report(report_path: Path, args: argparse.Namespace, summary_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Cross-Dataset Robustness Under Corrected Protocol")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This experiment applies the same corrected rolling G_t -> G_{t+1} protocol across Alibaba 2022, Alibaba 2021, "
        "and Huawei 2021 to test whether weak Huawei performance persists after protocol corrections."
    )
    lines.append("")
    lines.append("## 2. Shared Protocol")
    lines.append(f"- Model: {args.model}")
    lines.append(f"- Window size: {args.window_size}")
    lines.append(f"- Rolling direction: G_t -> E_(t+1)")
    lines.append(f"- Train/val split by window index: train up to G_{args.train_end_index-1}, val starts at G_{args.val_start_index}")
    lines.append(f"- Test starts at G_{args.test_start_index}")
    lines.append(f"- Negative evaluation ratios: {args.eval_ratios}")
    lines.append("")

    lines.append("## 3. Dataset Metadata")
    for _, r in meta_df.iterrows():
        lines.append(f"### {r['Dataset']}")
        lines.append(f"- Rows after preprocessing: {int(r['NumRows'])}")
        lines.append(f"- Services: {int(r['NumServices'])}")
        lines.append(f"- Windows: {int(r['NumWindows'])}")
        lines.append(f"- Train/Val pairs: {int(r['TrainPairs'])}/{int(r['ValPairs'])}")
        lines.append(f"- Best epoch: {int(r['BestEpoch'])}")
        lines.append(f"- Training time (sec): {r['TrainTimeSec']:.2f}")
        lines.append("")

    lines.append("## 4. Performance by Dataset and Ratio")
    datasets = ["Alibaba 2022", "Alibaba 2021", "Huawei 2021"]
    for d in datasets:
        ddf = summary_df[summary_df["Dataset"] == d].sort_values("NegRatio")
        if ddf.empty:
            continue
        lines.append(f"### {d}")
        for _, r in ddf.iterrows():
            ratio = int(r["NegRatio"])
            lines.append(f"- 1:{ratio} | AUC={r['AUC_Mean']:.4f}, Precision={r['Precision_Mean']:.4f}, Recall={r['Recall_Mean']:.4f}, F1={r['F1_Mean']:.4f}, Accuracy={r['Accuracy_Mean']:.4f}, TestPairs={int(r['TestPairsUsed'])}")
        lines.append("")

    lines.append("## 5. Huawei Persistence Check")
    try:
        h = summary_df[(summary_df["Dataset"] == "Huawei 2021") & (summary_df["NegRatio"] == 10.0)].iloc[0]
        a22 = summary_df[(summary_df["Dataset"] == "Alibaba 2022") & (summary_df["NegRatio"] == 10.0)].iloc[0]
        a21 = summary_df[(summary_df["Dataset"] == "Alibaba 2021") & (summary_df["NegRatio"] == 10.0)].iloc[0]
        lines.append(
            "Huawei still underperforms under the corrected protocol if its AUC/F1 remain materially lower than both Alibaba datasets at the same ratio."
        )
        lines.append(
            f"At 1:10, AUC: Huawei={h['AUC_Mean']:.4f}, Alibaba2022={a22['AUC_Mean']:.4f}, Alibaba2021={a21['AUC_Mean']:.4f}."
        )
        lines.append(
            f"At 1:10, F1: Huawei={h['F1_Mean']:.4f}, Alibaba2022={a22['F1_Mean']:.4f}, Alibaba2021={a21['F1_Mean']:.4f}."
        )
    except Exception:
        lines.append("Insufficient rows to compute a strict 1:10 Huawei persistence comparison.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cross-dataset robustness with corrected rolling protocol.")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--model", type=str, default="GAT")
    p.add_argument("--window-size", type=int, default=100)

    p.add_argument("--alibaba2022-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--alibaba2021-path", type=str, default="Data/MSCallGraph_0.csv")
    p.add_argument("--huawei2021-path", type=str, default="Data/Huawei/status_1min_20210411.csv")

    p.add_argument("--max-time", type=int, default=-1, help="Use <=0 to auto-cover full timestamp range per dataset")
    p.add_argument("--train-end-index", type=int, default=69)
    p.add_argument("--val-start-index", type=int, default=60)
    p.add_argument("--test-start-index", type=int, default=70)

    p.add_argument("--eval-ratios", type=str, default="1,5,10,50")
    p.add_argument("--high-ratio-threshold", type=int, default=50)
    p.add_argument("--high-ratio-max-pairs", type=int, default=300)

    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min-epochs", type=int, default=20)
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

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Alibaba 2022", Path(args.alibaba2022_path)),
        ("Alibaba 2021", Path(args.alibaba2021_path)),
        ("Huawei 2021", Path(args.huawei2021_path)),
    ]

    all_summary = []
    all_meta = []

    for dname, dpath in datasets:
        print(f"\n=== Running dataset: {dname} ===")
        summary_rows, meta = run_dataset(dname, dpath, args, device)
        all_summary.extend(summary_rows)
        all_meta.append(meta)

    if not all_summary:
        raise RuntimeError("No summary rows were produced.")

    summary_df = pd.DataFrame(all_summary)
    meta_df = pd.DataFrame(all_meta)

    summary_csv = results_dir / "cross_dataset_robustness_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_json = results_dir / "cross_dataset_robustness_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "config": {
                    "model": args.model,
                    "window_size": args.window_size,
                    "max_time": args.max_time,
                    "train_end_index": args.train_end_index,
                    "val_start_index": args.val_start_index,
                    "test_start_index": args.test_start_index,
                    "eval_ratios": args.eval_ratios,
                },
                "meta": meta_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = results_dir / "cross_dataset_robustness_report.md"
    write_report(report_path, args, summary_df, meta_df)

    print("\n=== Cross-Dataset Robustness Summary ===")
    print(summary_df[["Dataset", "NegRatio", "AUC_Mean", "F1_Mean", "Precision_Mean", "Recall_Mean", "Accuracy_Mean", "TestPairsUsed"]])
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
