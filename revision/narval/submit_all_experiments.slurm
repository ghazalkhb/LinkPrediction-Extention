#!/bin/bash
#SBATCH --job-name=lp-revision-all
#SBATCH --account=def-YOUR_PI
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=revision/narval/logs/%x-%j.out
#SBATCH --error=revision/narval/logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p revision/narval/logs

module purge
module load python/3.10
module load cuda/12.2

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Optional: one-hour Alibaba 2022 sample; expand end date for larger runs.
python revision/download_alibaba2022_microservices.py \
  --start-date 0d0 \
  --end-date 0d1 \
  --output-dir "Data/Alibaba 2022" \
  --output-csv CallGraph_0.csv \
  --include-rt

python revision/run_all_experiments.py \
  --dataset-path "Data/Alibaba 2022/CallGraph_0.csv" \
  --results-dir revision/results \
  --full-trace-window-ms 60000

python revision/build_full_revision_report.py \
  --results-dir revision/results \
  --output revision/FULL_EXPERIMENT_REPORT.md
