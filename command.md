python minimal_gpu_tuner/tune_ranker_blend_sharpe.py \
  --dataset data/processed/dataset_champion_rank_5d_longmom_meta_nosize.parquet \
  --target-col future_return_5d \
  --grid minimal_gpu_tuner/grid_gpu.yaml \
  --seed 42 \
  --train-start 2020-01-01 --valid-start 2024-01-01 \
  --purge-days 10 --bins 5 \
  --topn 5 --hold-days 5 --cost-bps 5 \
  --early-stopping-rounds 50 \
  --blend-weights 0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.20 \
  --save-best-model artifacts/models/xgb_rank_best_blend_top5.json


  salloc --time=8:00:00 --gpus-per-node=1 --mem=32G --cpus-per-task=4 --account=aip-nanditav