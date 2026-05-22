import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = ["timestamp", "um", "dm"]


def count_raw_rows(csv_path: Path) -> int:
    # Fast newline counting without loading the full file into memory.
    with open(csv_path, "rb") as f:
        count = 0
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            count += block.count(b"\n")
    # Subtract header line when file is non-empty.
    return max(0, count - 1)


def normalize_chunk_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    return chunk.rename(
        columns={
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
    )


def process_dataset(
    name: str,
    csv_path: Path,
    window_size: float,
    chunksize: int,
) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    t0 = time.time()
    raw_rows = count_raw_rows(csv_path)

    rows_after_badline = 0
    rows_after_dropna = 0

    services: set[str] = set()
    edge_hashes: set[int] = set()

    bucket_counts: dict[int, int] = defaultdict(int)
    min_bucket = None
    max_bucket = None

    for chunk in pd.read_csv(csv_path, on_bad_lines="skip", chunksize=chunksize):
        rows_after_badline += len(chunk)

        chunk = normalize_chunk_columns(chunk)
        if any(col not in chunk.columns for col in REQUIRED):
            missing = [c for c in REQUIRED if c not in chunk.columns]
            raise ValueError(f"Missing required columns after normalization for {name}: {missing}")

        work = chunk[REQUIRED].copy()
        work = work.dropna()

        # Normalize timestamp robustness across datasets.
        work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
        work = work.dropna(subset=["timestamp"])
        work = work[work["timestamp"] >= 0]

        if work.empty:
            continue

        rows_after_dropna += len(work)

        um = work["um"].astype(str)
        dm = work["dm"].astype(str)

        services.update(um.unique().tolist())
        services.update(dm.unique().tolist())

        # Use 64-bit hashed pairs for memory-efficient unique directed edge counting.
        edge_series = pd.util.hash_pandas_object(pd.DataFrame({"um": um, "dm": dm}), index=False)
        edge_hashes.update(edge_series.astype("uint64").tolist())

        buckets = np.floor(work["timestamp"].to_numpy(dtype=float) / float(window_size)).astype(np.int64)
        ub, counts = np.unique(buckets, return_counts=True)
        for b, c in zip(ub.tolist(), counts.tolist()):
            bucket_counts[b] += c

        if ub.size > 0:
            bmin = int(ub.min())
            bmax = int(ub.max())
            min_bucket = bmin if min_bucket is None else min(min_bucket, bmin)
            max_bucket = bmax if max_bucket is None else max(max_bucket, bmax)

    if min_bucket is None or max_bucket is None:
        num_windows = 0
        mean_edges_per_window = 0.0
        median_edges_per_window = 0.0
    else:
        num_windows = int(max_bucket - min_bucket + 1)
        edge_counts = np.zeros(num_windows, dtype=np.int64)
        for b, c in bucket_counts.items():
            edge_counts[b - min_bucket] = c
        mean_edges_per_window = float(edge_counts.mean())
        median_edges_per_window = float(np.median(edge_counts))

    unique_services = int(len(services))
    unique_directed_edges = int(len(edge_hashes))

    if unique_services <= 1:
        edge_density = 0.0
    else:
        edge_density = float(unique_directed_edges / (unique_services * (unique_services - 1)))

    removed_pct = 0.0 if raw_rows == 0 else float((raw_rows - rows_after_dropna) / raw_rows * 100.0)

    return {
        "Dataset": name,
        "Path": str(csv_path),
        "WindowSize": float(window_size),
        "RawRows": int(raw_rows),
        "RowsAfterBadLineRemoval": int(rows_after_badline),
        "RowsAfterDropna": int(rows_after_dropna),
        "RemovedPct": float(removed_pct),
        "UniqueServices": unique_services,
        "UniqueDirectedEdges": unique_directed_edges,
        "NumWindows": int(num_windows),
        "MeanEdgesPerWindow": float(mean_edges_per_window),
        "MedianEdgesPerWindow": float(median_edges_per_window),
        "EdgeDensity": float(edge_density),
        "RuntimeSec": float(time.time() - t0),
    }


def write_report(report_path: Path, stats_df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Preprocessing Statistics Per Dataset")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append(
        "This report quantifies preprocessing effects per dataset, including bad-line removal and dropna filtering, "
        "to address reviewer concerns about dataset noise and cleaning impact."
    )
    lines.append("")
    lines.append("## 2. Metrics Reported")
    lines.append("- raw rows")
    lines.append("- rows after bad-line removal")
    lines.append("- rows after dropna()")
    lines.append("- percentage removed")
    lines.append("- unique services")
    lines.append("- unique directed edges")
    lines.append("- number of windows")
    lines.append("- mean/median edges per window")
    lines.append("- edge density")
    lines.append("")
    lines.append("## 3. Dataset Statistics")

    for _, r in stats_df.iterrows():
        lines.append(f"### {r['Dataset']}")
        lines.append(f"- Source file: {r['Path']}")
        lines.append(f"- Window size used: {r['WindowSize']}")
        lines.append(f"- Raw rows: {int(r['RawRows'])}")
        lines.append(f"- Rows after bad-line removal: {int(r['RowsAfterBadLineRemoval'])}")
        lines.append(f"- Rows after dropna(): {int(r['RowsAfterDropna'])}")
        lines.append(f"- Percentage removed: {r['RemovedPct']:.2f}%")
        lines.append(f"- Unique services: {int(r['UniqueServices'])}")
        lines.append(f"- Unique directed edges: {int(r['UniqueDirectedEdges'])}")
        lines.append(f"- Number of windows: {int(r['NumWindows'])}")
        lines.append(f"- Mean edges per window: {r['MeanEdgesPerWindow']:.2f}")
        lines.append(f"- Median edges per window: {r['MedianEdgesPerWindow']:.2f}")
        lines.append(f"- Edge density: {r['EdgeDensity']:.8f}")
        lines.append(f"- Runtime (sec): {r['RuntimeSec']:.2f}")
        lines.append("")

    lines.append("## 4. Notes")
    lines.append(
        "dropna() is applied after schema normalization to required modeling fields (timestamp, um, dm), "
        "matching the paper's unified preprocessing pattern."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute preprocessing statistics per dataset.")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--chunksize", type=int, default=500000)

    p.add_argument("--alibaba2021-path", type=str, default="Data/MSCallGraph_0.csv")
    p.add_argument("--alibaba2022-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--huawei2021-path", type=str, default="Data/Huawei/status_1min_20210411.csv")

    p.add_argument("--alibaba2021-window", type=float, default=100.0)
    p.add_argument("--alibaba2022-window", type=float, default=100.0)
    p.add_argument("--huawei2021-window", type=float, default=100.0)
    return p


def main() -> None:
    args = build_parser().parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("Alibaba 2021", Path(args.alibaba2021_path), args.alibaba2021_window),
        ("Alibaba 2022", Path(args.alibaba2022_path), args.alibaba2022_window),
        ("Huawei 2021", Path(args.huawei2021_path), args.huawei2021_window),
    ]

    rows = []
    for name, path, window in targets:
        print(f"Processing {name} ...")
        rows.append(process_dataset(name, path, window, args.chunksize))

    stats_df = pd.DataFrame(rows)

    csv_path = results_dir / "preprocessing_dataset_stats.csv"
    stats_df.to_csv(csv_path, index=False)

    json_path = results_dir / "preprocessing_dataset_stats.json"
    json_path.write_text(
        json.dumps(
            {
                "datasets": stats_df.to_dict(orient="records"),
                "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report_path = results_dir / "preprocessing_dataset_stats_report.md"
    write_report(report_path, stats_df)

    print("\n=== Preprocessing Statistics Summary ===")
    print(
        stats_df[
            [
                "Dataset",
                "RawRows",
                "RowsAfterBadLineRemoval",
                "RowsAfterDropna",
                "RemovedPct",
                "UniqueServices",
                "UniqueDirectedEdges",
                "NumWindows",
                "MeanEdgesPerWindow",
                "MedianEdgesPerWindow",
                "EdgeDensity",
            ]
        ]
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
