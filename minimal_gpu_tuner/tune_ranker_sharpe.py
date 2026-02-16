from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

from minimal_gpu_tuner.backtest_min import build_ret_matrix, overlap_topn_backtest_from_scores
from minimal_gpu_tuner.tune_ranker_min import (
    _apply_feature_policy,
    _dataset_fingerprint,
    _env_fingerprint,
    _make_groups,
    _pick_features,
    _purge_embargo,
    _time_split,
    _to_relevance_bins,
)


def _read_grid(path: str) -> Dict[str, List[Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"grid not found: {p}")
    g = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(g, dict):
        raise ValueError("grid must be a mapping param -> list(values)")
    out: Dict[str, List[Any]] = {}
    for k, v in g.items():
        if isinstance(v, list):
            out[str(k)] = v
        else:
            out[str(k)] = [v]
    return out


def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {keys[i]: combo[i] for i in range(len(keys))}


def _score_matrix_for_valid(
    valid: pd.DataFrame,
    *,
    dates: List[pd.Timestamp],
    tickers: List[str],
    score_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> np.ndarray:
    s = valid[[date_col, ticker_col, score_col]].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s[ticker_col] = s[ticker_col].astype(str)
    # pivot to match ret matrix axes
    mat = (
        s.pivot(index=date_col, columns=ticker_col, values=score_col)
        .reindex(index=dates, columns=tickers)
        .astype("float32")
        .to_numpy(copy=True)
    )
    return mat


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune XGB ranker hyperparams by portfolio Sharpe (overlap Top-N backtest).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-ohlcv-levels", action="store_true")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="")
    ap.add_argument("--purge-days", type=int, default=10)
    ap.add_argument("--embargo-days", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--price-col", default="adj_close")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--out", default="", help="Output CSV path (default: artifacts/logs/tune_sharpe_<ts>.csv)")
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

    tr, va = _time_split(df, train_start=args.train_start, valid_start=args.valid_start, valid_end=(args.valid_end or None))
    tr, va = _purge_embargo(tr, va, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

    feats = _pick_features(df)
    feats = _apply_feature_policy(feats, include_ohlcv_levels=bool(args.include_ohlcv_levels))

    # Train labels for rank objective
    bins = int(args.bins)
    y_tr = _to_relevance_bins(tr, date_col="date", target_col=tgt, bins=bins)
    y_va = _to_relevance_bins(va, date_col="date", target_col=tgt, bins=bins)
    g_tr = _make_groups(tr["date"])
    g_va = _make_groups(va["date"])

    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    X_va = va[feats].replace([np.inf, -np.inf], np.nan)
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    dtr.set_group(g_tr)
    dva.set_group(g_va)

    # Precompute returns matrix on validation panel (same dates/tickers axes for all trials)
    if str(args.price_col) not in df.columns:
        raise RuntimeError(f"dataset missing price_col: {args.price_col}")
    ret_mat, dates, tickers, _ = build_ret_matrix(va, price_col=str(args.price_col))

    grid = _read_grid(args.grid)
    combos = list(_iter_grid(grid))
    if int(args.max_trials) > 0:
        combos = combos[: int(args.max_trials)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (Path("artifacts") / "logs" / f"tune_sharpe_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    meta = {
        "created_at": ts,
        "env": _env_fingerprint(),
        "args": vars(args),
        "dataset": {"path": str(ds_path), "fingerprint": _dataset_fingerprint(df, target_col=tgt)},
        "n_features": int(len(feats)),
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"meta={meta_path}")
    print(f"dataset={ds_path} train={len(tr)} valid={len(va)} feats={len(feats)} target={tgt} price_col={args.price_col}")
    print(f"trials={len(combos)} grid_keys={list(grid.keys())}")

    results: List[Dict[str, Any]] = []
    best = None

    for i, ov in enumerate(combos, start=1):
        t0 = time.time()
        r: Dict[str, Any] = {"trial": int(i), "overrides": json.dumps(ov, sort_keys=True)}
        try:
            params = dict(ov)
            num_round = int(params.pop("num_boost_round", params.pop("n_estimators", 400)))
            # alias
            if "learning_rate" in params and "eta" not in params:
                params["eta"] = float(params.pop("learning_rate"))
            if "reg_lambda" in params and "lambda" not in params:
                params["lambda"] = float(params.pop("reg_lambda"))

            params.setdefault("seed", int(args.seed))
            params.setdefault("nthread", -1)

            booster = xgb.train(
                params=params,
                dtrain=dtr,
                num_boost_round=num_round,
                evals=[(dva, "valid")],
                early_stopping_rounds=int(args.early_stopping_rounds) if int(args.early_stopping_rounds) > 0 else None,
                verbose_eval=False,
            )
            if booster.best_iteration is not None:
                score = booster.predict(dva, iteration_range=(0, int(booster.best_iteration) + 1))
                r["best_iteration"] = int(booster.best_iteration)
            else:
                score = booster.predict(dva)
                r["best_iteration"] = -1

            va_scored = va[["date", "ticker"]].copy()
            va_scored["score"] = score
            score_mat = _score_matrix_for_valid(va_scored, dates=dates, tickers=tickers, score_col="score")

            _, summ = overlap_topn_backtest_from_scores(
                score_mat=score_mat,
                ret_mat=ret_mat,
                dates=dates,
                top_n=int(args.topn),
                hold_days=int(args.hold_days),
                cost_bps=float(args.cost_bps),
            )

            r.update(
                {
                    "sharpe": float(summ.sharpe),
                    "cum_return": float(summ.cum_return),
                    "ann_return": float(summ.ann_return),
                    "ann_vol": float(summ.ann_vol),
                    "max_drawdown": float(summ.max_drawdown),
                    "avg_turnover": float(summ.avg_turnover),
                }
            )
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
        r["fit_seconds"] = float(time.time() - t0)
        results.append(r)

        if "error" not in r and not np.isnan(r.get("sharpe", np.nan)):
            key = (float(r["sharpe"]), float(r.get("cum_return", -np.inf)))
            if best is None or key > best[0]:
                best = (key, r)

        if i % 10 == 0 or i == 1 or i == len(combos):
            if best:
                print(f"[{i}/{len(combos)}] best_sharpe={best[1]['sharpe']:.3f} best_cum={best[1]['cum_return']:.3f}")
            else:
                print(f"[{i}/{len(combos)}] no_success_yet")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"output={out_path}")
    if best:
        print("best_overrides=", best[1]["overrides"])
        print(f"best_sharpe={best[1]['sharpe']:.3f} best_cum_return={best[1]['cum_return']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


