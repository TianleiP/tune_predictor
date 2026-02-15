# Minimal GPU Hyperparameter Tuner (XGBoost Ranker)

This is a **self-contained** brute-force hyperparameter tuner for cross-sectional stock ranking using **XGBoost**.

It optimizes:
- **Primary**: `spread_tstat` (t-stat of daily Top‑N spread)
- **Secondary**: `ndcg@K` (mean daily NDCG at K)

It only needs a **panel dataset** (Parquet) with at least:
- `date` (YYYY-MM-DD or parseable datetime)
- `ticker`
- target column like `future_return_5d`

## 1) Install

On the remote machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r minimal_gpu_tuner/requirements.txt
```

## 2) Run (GPU)

Put your dataset parquet somewhere (example path shown below), then:

```bash
python minimal_gpu_tuner/tune_ranker_min.py \
  --dataset /path/to/dataset_champion_rank_5d_longmom_meta_nosize.parquet \
  --target-col future_return_5d \
  --grid minimal_gpu_tuner/grid_gpu.yaml \
  --train-start 2020-01-01 \
  --valid-start 2024-01-01 \
  --purge-days 10 \
  --topn 20 \
  --ndcg-k 20 \
  --early-stopping-rounds 50 \
  --max-trials 300
```

Output is written to `./artifacts/logs/tune_ranker_<timestamp>.csv` by default.

## 3) Run (CPU)

```bash
python minimal_gpu_tuner/tune_ranker_min.py \
  --dataset /path/to/dataset.parquet \
  --target-col future_return_5d \
  --grid minimal_gpu_tuner/grid_cpu.yaml \
  --train-start 2020-01-01 \
  --valid-start 2024-01-01 \
  --purge-days 10 \
  --early-stopping-rounds 50 \
  --max-trials 50
```

## Notes
- If your machine sleeps, training stops. On servers, use `tmux` / `screen` / `nohup`.
- GPU requires a CUDA-capable NVIDIA GPU and an XGBoost build with CUDA enabled.


