#!/bin/bash
#SBATCH --job-name=lp-revision-one
#SBATCH --account=def-YOUR_PI
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=revision/narval/logs/%x-%j.out
#SBATCH --error=revision/narval/logs/%x-%j.err

# Usage:
# sbatch --export=ALL,EXPERIMENT=downstream_operational_proxy_experiment.py,DATASET=Data/Alibaba\ 2022/CallGraph_0.csv revision/narval/submit_single_experiment.slurm

set -euo pipefail

: "${EXPERIMENT:?Set EXPERIMENT script name, e.g., rolling_gt_to_gt1_experiment.py}"
DATASET="${DATASET:-Data/Alibaba 2022/CallGraph_0.csv}"
RESULTS_DIR="${RESULTS_DIR:-revision/results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

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
if ! python -m pip install -r requirements.txt; then
  echo "Pinned requirements install failed on Narval wheelhouse; using compatible core deps."
  python -m pip install numpy pandas scikit-learn scipy torch torch-geometric tqdm networkx
fi

python "revision/${EXPERIMENT}" --dataset-path "$DATASET" --results-dir "$RESULTS_DIR" ${EXTRA_ARGS}
