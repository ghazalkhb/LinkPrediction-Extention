# Narval GPU Submission

## 1. Run Entire Revision Suite

```bash
sbatch revision/narval/submit_all_experiments.slurm
```

Before running, update account in `submit_all_experiments.slurm`:
- `#SBATCH --account=def-YOUR_PI`

## 2. Run One Experiment

```bash
sbatch --export=ALL,EXPERIMENT=downstream_operational_proxy_experiment.py,DATASET=Data/Alibaba\ 2022/CallGraph_0.csv revision/narval/submit_single_experiment.slurm
```

Optional additional args:

```bash
sbatch --export=ALL,EXPERIMENT=full_trace_drift_experiment.py,DATASET=Data/Alibaba\ 2022/CallGraph_0.csv,EXTRA_ARGS="--window-size 60000 --train-days 2" revision/narval/submit_single_experiment.slurm
```

## 3. Notes

- All revision scripts auto-select GPU via CUDA when available.
- Use `--force-cpu` on script commands only for debugging.
- Full 13-day data is large; start with a short date range for smoke checks, then expand.
