import argparse
import csv
import io
import json
import re
import tarfile
import time
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2022MicroservicesTraces"
CALLGRAPH_REMOTE_PREFIX = "CallGraph/CallGraph"
REQUIRED_COLS = ["timestamp", "um", "dm"]


def parse_day_hour(token: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)d(\d+)", token.strip())
    if m is None:
        raise ValueError(f"Invalid date token '{token}'. Expected format like 0d0, 1d12.")
    day = int(m.group(1))
    hour = int(m.group(2))
    if hour < 0 or hour > 23:
        raise ValueError(f"Hour must be in [0, 23], got {hour}.")
    return day, hour


def date_to_minutes(day: int, hour: int) -> int:
    return day * 24 * 60 + hour * 60


def callgraph_indices(start_token: str, end_token: str) -> list[int]:
    s_day, s_hour = parse_day_hour(start_token)
    e_day, e_hour = parse_day_hour(end_token)
    start_m = date_to_minutes(s_day, s_hour)
    end_m = date_to_minutes(e_day, e_hour)
    if end_m <= start_m:
        raise ValueError("end_date must be after start_date.")

    ratio_minutes = 3
    start_idx = start_m // ratio_minutes
    end_idx = end_m // ratio_minutes - 1
    return list(range(start_idx, end_idx + 1))


def download_file(url: str, out_path: Path, timeout_sec: int, retries: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            with urlopen(url, timeout=timeout_sec) as resp, open(out_path, "wb") as out:
                out.write(resp.read())
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Failed downloading {url} after {retries} attempts: {exc}") from exc
            wait_sec = min(30, attempt * 3)
            print(f"Download failed ({exc}), retrying in {wait_sec}s: {url}")
            time.sleep(wait_sec)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
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
    out = df.rename(columns=rename_map)
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return out


def append_normalized_csv(
    input_csv: Path,
    output_csv: Path,
    chunksize: int,
    include_rt: bool,
) -> int:
    header_written = output_csv.exists() and output_csv.stat().st_size > 0
    rows = 0

    for chunk in pd.read_csv(input_csv, on_bad_lines="skip", chunksize=chunksize):
        chunk = normalize_cols(chunk)
        cols = ["timestamp", "um", "dm"]
        if include_rt and "rt" in chunk.columns:
            cols.append("rt")

        slim = chunk[cols].copy()
        slim = slim.dropna(subset=["timestamp", "um", "dm"])
        slim["timestamp"] = pd.to_numeric(slim["timestamp"], errors="coerce")
        slim = slim.dropna(subset=["timestamp"])
        slim = slim[slim["timestamp"] >= 0]

        if slim.empty:
            continue

        slim["um"] = slim["um"].astype(str)
        slim["dm"] = slim["dm"].astype(str)
        slim.to_csv(output_csv, mode="a", index=False, header=not header_written)
        header_written = True
        rows += int(len(slim))

    return rows


def append_from_archive(
    archive_path: Path,
    output_csv: Path,
    chunksize: int,
    include_rt: bool,
) -> int:
    rows = 0
    with tarfile.open(archive_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".csv"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            text_stream = io.TextIOWrapper(fobj, encoding="utf-8", errors="replace", newline="")
            for chunk in pd.read_csv(text_stream, on_bad_lines="skip", chunksize=chunksize):
                chunk = normalize_cols(chunk)
                cols = ["timestamp", "um", "dm"]
                if include_rt and "rt" in chunk.columns:
                    cols.append("rt")

                slim = chunk[cols].copy()
                slim = slim.dropna(subset=["timestamp", "um", "dm"])
                slim["timestamp"] = pd.to_numeric(slim["timestamp"], errors="coerce")
                slim = slim.dropna(subset=["timestamp"])
                slim = slim[slim["timestamp"] >= 0]
                if slim.empty:
                    continue

                slim["um"] = slim["um"].astype(str)
                slim["dm"] = slim["dm"].astype(str)
                header_written = output_csv.exists() and output_csv.stat().st_size > 0
                slim.to_csv(output_csv, mode="a", index=False, header=not header_written)
                rows += int(len(slim))
            text_stream.detach()
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Alibaba 2022 microservice call graph data and build CallGraph_0.csv")
    p.add_argument("--start-date", type=str, default="0d0", help="Inclusive start in day-hour format, e.g., 0d0")
    p.add_argument("--end-date", type=str, default="0d1", help="Exclusive end in day-hour format, e.g., 0d1")
    p.add_argument("--output-dir", type=str, default="Data/Alibaba 2022")
    p.add_argument("--output-csv", type=str, default="CallGraph_0.csv")
    p.add_argument("--archives-dir", type=str, default="", help="Optional directory to store temporary archives")
    p.add_argument("--start-idx", type=int, default=None, help="Optional explicit start archive index (inclusive)")
    p.add_argument("--end-idx", type=int, default=None, help="Optional explicit end archive index (inclusive)")
    p.add_argument("--resume", action="store_true", help="Append to existing output CSV and continue from progress file")
    p.add_argument("--progress-file", type=str, default="", help="Path to progress json file")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--include-rt", action="store_true", help="Keep rt column if present")
    p.add_argument("--timeout-sec", type=int, default=120)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--chunksize", type=int, default=500000)
    return p


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archives_dir = Path(args.archives_dir) if args.archives_dir else (output_dir / "raw_archives")
    archives_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / args.output_csv
    progress_file = Path(args.progress_file) if args.progress_file else (output_dir / "download_progress.json")

    if out_csv.exists() and not args.resume:
        out_csv.unlink()

    indices = callgraph_indices(args.start_date, args.end_date)
    if not indices:
        raise RuntimeError("No call graph indices computed from the date range.")

    if args.start_idx is not None:
        indices = [i for i in indices if i >= args.start_idx]
    if args.end_idx is not None:
        indices = [i for i in indices if i <= args.end_idx]

    if args.resume and progress_file.exists():
        try:
            prog = json.loads(progress_file.read_text(encoding="utf-8"))
            last_idx = int(prog.get("last_completed_idx", -1))
            indices = [i for i in indices if i > last_idx]
        except Exception:
            pass

    print(f"Downloading {len(indices)} CallGraph archive(s) for range [{args.start_date}, {args.end_date})", flush=True)

    total_rows = 0

    for idx in indices:
        remote = f"{CALLGRAPH_REMOTE_PREFIX}_{idx}.tar.gz"
        url = f"{BASE_URL}/{remote}"
        archive_path = archives_dir / f"CallGraph_{idx}.tar.gz"

        print(f"Downloading idx={idx} ...", flush=True)
        download_file(url, archive_path, timeout_sec=args.timeout_sec, retries=args.retries)

        total_rows += append_from_archive(
            archive_path,
            out_csv,
            chunksize=args.chunksize,
            include_rt=args.include_rt,
        )

        if not args.keep_archives:
            archive_path.unlink(missing_ok=True)

        progress_file.write_text(
            json.dumps(
                {
                    "last_completed_idx": int(idx),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_csv": str(out_csv),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if total_rows == 0:
        raise RuntimeError("No rows were written. Check requested date range and upstream availability.")

    print("Done", flush=True)
    print(f"Output CSV: {out_csv}", flush=True)
    print(f"Rows written: {total_rows}", flush=True)


if __name__ == "__main__":
    main()
