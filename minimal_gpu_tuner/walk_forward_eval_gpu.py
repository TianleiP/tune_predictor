from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import sys

# Make `minimal_gpu_tuner.*` imports work no matter the current working directory.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # tune_predictor/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from minimal_gpu_tuner.tune_ranker_min import (  # noqa: E402
    _apply_feature_policy,
    _dataset_fingerprint,
    _env_fingerprint,
    _evaluate_spread,
    _make_groups,
    _pick_features,
    _purge_embargo,
    _to_relevance_bins,
)


def _read_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"yaml not found: {p}")
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"yaml must be a mapping: {p}")
    return obj


def _load_xgb_from_meta(meta_path: str) -> tuple[dict, int]:
    """
    Load xgboost.train params + num_boost_round from a saved model meta yaml.
    Expected keys: xgb_params, num_boost_round.
    """
    m = _read_yaml(meta_path)
    p = dict(m.get("xgb_params") or {})
    n = int(m.get("num_boost_round") or 600)
    # Normalize aliases
    if "learning_rate" in p and "eta" not in p:
        p["eta"] = float(p.pop("learning_rate"))
    if "reg_lambda" in p and "lambda" not in p:
        p["lambda"] = float(p.pop("reg_lambda"))
    return p, n


def _month_add(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return (dt + pd.DateOffset(months=months)).normalize()


def _time_block_slices(
    *,
    df: pd.DataFrame,
    train_years: int,
    test_months: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Return list of blocks: (train_start, train_end, test_start, test_end) where train_end == test_start.
    """
    blocks: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    test_start = start
    while test_start < end:
        test_end = _month_add(test_start, test_months)
        if test_end > end + pd.Timedelta(days=1):
            test_end = end + pd.Timedelta(days=1)
        train_end = test_start
        train_start = (test_start - pd.DateOffset(years=int(train_years))).normalize()
        blocks.append((train_start, train_end, test_start, test_end))
        test_start = test_end
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward evaluation on GPU (rolling retrain + block-by-block IC/spread/turnover).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target-col", default="future_return_5d")
    ap.add_argument("--include-ohlcv-levels", action="store_true")

    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="", help="Optional last date (YYYY-MM-DD). Default: dataset max.")
    ap.add_argument("--topn", type=int, default=20)

    ap.add_argument("--purge-days", type=int, default=10)
    ap.add_argument("--embargo-days", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stopping-rounds", type=int, default=0, help="0 disables early stopping.")

    ap.add_argument(
        "--xgb-meta",
        default="",
        help="Optional: path to a saved model .meta.yaml containing xgb_params + num_boost_round (overrides CLI params).",
    )
    ap.add_argument("--num-boost-round", type=int, default=600)

    # Light default params if no meta is provided
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lambda", dest="reg_lambda", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--out", default="", help="Output CSV path (default: artifacts/logs/walk_forward_gpu_<ts>.csv)")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    df = pd.read_parquet(ds_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    tgt = str(args.target_col)
    if tgt not in df.columns:
        raise RuntimeError(f"dataset missing target col: {tgt}")
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    df = df.dropna(subset=[tgt])

    feats = _pick_features(df)
    feats = _apply_feature_policy(feats, include_ohlcv_levels=bool(args.include_ohlcv_levels))

    # Params
    if str(args.xgb_meta).strip():
        params, num_round = _load_xgb_from_meta(str(args.xgb_meta))
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda")
        params.setdefault("objective", "rank:pairwise")
        params.setdefault("eval_metric", "ndcg@20")
        params.setdefault("seed", int(args.seed))
        params.setdefault("nthread", -1)
    else:
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@20",
            "tree_method": "hist",
            "device": "cuda",
            "seed": int(args.seed),
            "nthread": -1,
            "max_depth": int(args.max_depth),
            "min_child_weight": float(args.min_child_weight),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "eta": float(args.eta),
            "lambda": float(args.reg_lambda),
            "gamma": float(args.gamma),
        }
        num_round = int(args.num_boost_round)

    start = pd.to_datetime(str(args.start)).normalize()
    end = pd.to_datetime(str(args.end)).normalize() if str(args.end).strip() else pd.to_datetime(df["date"].max()).normalize()

    blocks = _time_block_slices(
        df=df,
        train_years=int(args.train_years),
        test_months=int(args.test_months),
        start=start,
        end=end,
    )
    if not blocks:
        raise RuntimeError("no blocks produced; check --start/--end")

    rows: List[Dict[str, Any]] = []
    for bi, (train_start, train_end, test_start, test_end) in enumerate(blocks):
        tr = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
        te = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()
        if tr.empty or te.empty:
            continue

        # Purge/embargo boundary safety
        tr, te = _purge_embargo(tr, te, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

        # Rank labels + groups
        bins = int(args.bins)
        y_tr = _to_relevance_bins(tr, date_col="date", target_col=tgt, bins=bins)
        y_te = _to_relevance_bins(te, date_col="date", target_col=tgt, bins=bins)
        g_tr = _make_groups(tr["date"])
        g_te = _make_groups(te["date"])

        X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
        X_te = te[feats].replace([np.inf, -np.inf], np.nan)

        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dte = xgb.DMatrix(X_te, label=y_te)
        dtr.set_group(g_tr)
        dte.set_group(g_te)

        esr = int(args.early_stopping_rounds)
        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=int(num_round),
            evals=[(dte, "test")] if esr > 0 else [],
            early_stopping_rounds=esr if esr > 0 else None,
            verbose_eval=False,
        )

        best_it = int(getattr(booster, "best_iteration", -1))
        if best_it >= 0:
            score = booster.predict(dte, iteration_range=(0, best_it + 1))
        else:
            score = booster.predict(dte)

        panel = te[["date", "ticker", tgt]].copy()
        panel["score"] = score

        daily, summary = _evaluate_spread(
            panel,
            date_col="date",
            ticker_col="ticker",
            score_col="score",
            target_col=tgt,
            top_n=int(args.topn),
        )

        rows.append(
            {
                "block": int(bi),
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": (train_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": (test_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "days": int(summary["days"]),
                "ic_mean": float(summary["ic_mean"]),
                "spread_mean": float(summary["spread_mean"]),
                "spread_tstat": float(summary["spread_tstat"]),
                "turnover_mean": float(summary["turnover_mean"]),
                "best_iteration": int(best_it),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no blocks produced results (check dates / dataset coverage)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if str(args.out).strip() else (Path("artifacts") / "logs" / f"walk_forward_gpu_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    meta = {
        "created_at": ts,
        "env": _env_fingerprint(),
        "args": vars(args),
        "dataset": {"path": str(ds_path), "fingerprint": _dataset_fingerprint(df, target_col=tgt)},
        "n_features": int(len(feats)),
        "xgb_params": params,
        "num_boost_round": int(num_round),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"meta={meta_path}")
    print(f"output={out_path}")
    print(out[["block", "test_start", "test_end", "ic_mean", "spread_mean", "spread_tstat", "turnover_mean", "best_iteration"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


