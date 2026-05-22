import argparse
import time
from pathlib import Path


SECTION_FILES = [
    "rolling_experiment_report.md",
    "gap_experiment_report.md",
    "recurring_experiment_report.md",
    "horizon_experiment_report.md",
    "overlap_fair_experiment_report.md",
    "imbalanced_experiment_report.md",
    "temporal_slice_sensitivity_report.md",
    "cross_dataset_robustness_report.md",
    "ranking_experiment_report.md",
    "downstream_operational_proxy_report.md",
    "full_trace_drift_report.md",
    "runtime_breakdown_report.md",
    "paired_statistical_tests_report.md",
    "preprocessing_dataset_stats_report.md",
]


def build_report(results_dir: Path, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Full Report: Revision Experiments")
    lines.append("")
    lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This file consolidates all per-experiment markdown reports generated in revision/results.")
    lines.append("")

    missing = []
    included = 0

    for name in SECTION_FILES:
        path = results_dir / name
        if not path.exists():
            missing.append(name)
            continue

        content = path.read_text(encoding="utf-8").strip()
        lines.append("---")
        lines.append("")
        lines.append(content)
        lines.append("")
        included += 1

    lines.append("---")
    lines.append("")
    lines.append("## Coverage Summary")
    lines.append(f"- Included reports: {included}")
    if missing:
        lines.append(f"- Missing reports: {len(missing)}")
        for name in missing:
            lines.append(f"  - {name}")
    else:
        lines.append("- Missing reports: 0")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build one consolidated markdown report from revision/results")
    parser.add_argument("--results-dir", type=str, default="revision/results")
    parser.add_argument("--output", type=str, default="revision/FULL_EXPERIMENT_REPORT.md")
    args = parser.parse_args()

    build_report(Path(args.results_dir), Path(args.output))
    print(f"Saved: {args.output}")
