"""
Master runner: execute all revision experiments sequentially with degree-aware sampling.
Writes progress to revision/run_progress.log so the user can check status.
Supports GPU-by-default execution and Narval-friendly CLI overrides.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

REVISION_DIR = Path(__file__).resolve().parent
ROOT_DIR = REVISION_DIR.parent  # workspace root where Data/ lives
RESULTS_DIR = REVISION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROGRESS_LOG = REVISION_DIR / "run_progress.log"
DEFAULT_TIMEOUT_SECONDS = 7200  # 2 hours for typical experiments

# Per-script timeout overrides for known long-running experiments.
SCRIPT_TIMEOUTS = {
    "training_testing_gap_experiment.py": 21600,  # 6 hours
    "paired_statistical_tests.py": 21600,         # 6 hours
    "full_trace_drift_experiment.py": 43200,      # 12 hours
    "downstream_operational_proxy_experiment.py": 21600,
}

def build_experiments(dataset_path: str, results_dir: str, full_trace_window_ms: int) -> list[tuple[str, str, list[str]]]:
    # Each entry: (script_name, description, extra_args_list)
    # Scripts are run with cwd=ROOT_DIR so relative Data/ paths resolve correctly.
    return [
        ("preprocessing_dataset_statistics.py", "Dataset statistics (no training)", ["--results-dir", results_dir]),
        ("rolling_gt_to_gt1_experiment.py", "Rolling G_t -> G_{t+1}",
         ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("training_testing_gap_experiment.py", "Training-testing gap experiment", ["--results-dir", results_dir]),
        ("recurring_activity_experiment.py", "Recurring activity experiment", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("horizon_gt_to_gtk_experiment.py", "Horizon G_t -> G_{t+k}", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("overlap_fair_experiment.py", "Overlap-fair window evaluation", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("imbalanced_negative_evaluation.py", "Imbalanced negative ratios", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("temporal_slice_sensitivity.py", "Temporal slice sensitivity", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("cross_dataset_robustness.py", "Cross-dataset robustness", ["--results-dir", results_dir]),
        ("ranking_evaluation.py", "Ranking metrics evaluation", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("downstream_operational_proxy_experiment.py", "Downstream operational proxy validation", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        (
            "full_trace_drift_experiment.py",
            "Full-trace long-horizon drift experiment",
            ["--dataset-path", dataset_path, "--results-dir", results_dir, "--window-size", str(full_trace_window_ms)],
        ),
        ("runtime_measurement_breakdown.py", "Runtime measurement breakdown", ["--dataset-path", dataset_path, "--results-dir", results_dir]),
        ("paired_statistical_tests.py", "Paired statistical tests", ["--results-dir", results_dir]),
    ]


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run all revision experiments sequentially.")
    p.add_argument("--dataset-path", type=str, default="Data/Alibaba 2022/CallGraph_0.csv")
    p.add_argument("--results-dir", type=str, default="revision/results")
    p.add_argument("--full-trace-window-ms", type=int, default=60000)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--only", type=str, default="", help="Comma-separated script names to run.")
    return p


def main():
    args = build_parser().parse_args()
    experiments = build_experiments(args.dataset_path, args.results_dir, args.full_trace_window_ms)
    if args.only.strip():
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        experiments = [exp for exp in experiments if exp[0] in allow]

    if not experiments:
        raise ValueError("No experiments selected. Check --only filter.")

    log("=" * 70)
    log("STARTING ALL REVISION EXPERIMENTS (degree-aware sampling)")
    log(f"dataset_path={args.dataset_path}")
    log(f"results_dir={args.results_dir}")
    log(f"device_mode={'CPU-forced' if args.force_cpu else 'Auto(CUDA if available)'}")
    log("=" * 70)

    total = len(experiments)
    passed = 0
    failed = []

    for idx, (script, description, extra_args) in enumerate(experiments, 1):
        script_path = REVISION_DIR / script
        log(f"[{idx}/{total}] RUNNING: {description} ({script})")
        start = time.time()

        cmd = [sys.executable, str(script_path)] + extra_args
        if args.force_cpu:
            cmd.append("--force-cpu")
        timeout_seconds = SCRIPT_TIMEOUTS.get(script, DEFAULT_TIMEOUT_SECONDS)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            elapsed = time.time() - start

            if result.returncode == 0:
                log(f"[{idx}/{total}] PASSED: {description} ({elapsed:.1f}s)")
                passed += 1
            else:
                log(f"[{idx}/{total}] FAILED (rc={result.returncode}): {description} ({elapsed:.1f}s)")
                log(f"  STDERR (last 500 chars): {result.stderr[-500:]}")
                failed.append(script)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            log(f"[{idx}/{total}] TIMEOUT: {description} ({elapsed:.1f}s)")
            failed.append(script)
        except Exception as e:
            elapsed = time.time() - start
            log(f"[{idx}/{total}] ERROR: {description} ({elapsed:.1f}s) - {e}")
            failed.append(script)

    log("=" * 70)
    log(f"FINISHED: {passed}/{total} passed, {len(failed)} failed")
    if failed:
        log(f"Failed scripts: {failed}")

    # Always refresh consolidated report from any available outputs.
    report_cmd = [
        sys.executable,
        str(REVISION_DIR / "build_full_revision_report.py"),
        "--results-dir",
        args.results_dir,
        "--output",
        str(REVISION_DIR / "FULL_EXPERIMENT_REPORT.md"),
    ]
    report_result = subprocess.run(report_cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    if report_result.returncode == 0:
        log("Consolidated report updated: revision/FULL_EXPERIMENT_REPORT.md")
    else:
        log(
            "Consolidated report generation failed. "
            f"stderr(last500)={report_result.stderr[-500:]}"
        )
    log("=" * 70)


if __name__ == "__main__":
    main()
