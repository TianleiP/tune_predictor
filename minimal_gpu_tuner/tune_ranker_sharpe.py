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

import sys

# Make `minimal_gpu_tuner.*` imports work no matter the current working directory.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent  # tune_predictor/
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from minimal_gpu_tuner.overlap_topn_official import run_overlap_topn_backtest
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, obj: dict) -> None:
    _atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False))


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
    ap.add_argument("--slippage-k", type=float, default=0.0)
    ap.add_argument("--slippage-cap-bps", type=float, default=50.0)
    ap.add_argument("--liq-min-logdv", type=float, default=None)
    ap.add_argument("--price-col", default="adj_close")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--out", default="", help="Output CSV path (default: artifacts/logs/tune_sharpe_<ts>.csv)")
    ap.add_argument(
        "--save-best-model",
        default="",
        help="If set, save the best model to this path (overwritten whenever a new best is found).",
    )
    ap.add_argument(
        "--stop-at-sharpe",
        type=float,
        default=None,
        help="If set, stop the search early once best_sharpe >= this value (still writes CSV/meta).",
    )
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

    # Price/scored panels for official backtest
    price_col = str(args.price_col)
    if price_col not in df.columns:
        raise RuntimeError(f"dataset missing price_col: {price_col}")
    price_cols = [c for c in ["date", "ticker", price_col, "amihud_illiq", "log_dollar_volume"] if c in va.columns]
    price_panel = va[price_cols].copy()

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
    best_saved = False
    best_model_path = Path(str(args.save_best_model)) if str(args.save_best_model).strip() else None

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
            scored_cols = [c for c in ["date", "ticker", "score", "amihud_illiq", "log_dollar_volume"] if c in va.columns]
            scored_panel = va[scored_cols].copy()
            scored_panel["score"] = score

            _, summ = run_overlap_topn_backtest(
                scored_panel=scored_panel,
                price_panel=price_panel,
                top_n=int(args.topn),
                hold_days=int(args.hold_days),
                cost_bps=float(args.cost_bps),
                slippage_k=float(args.slippage_k),
                slippage_cap_bps=float(args.slippage_cap_bps),
                liquidity_min_log_dollar_volume=args.liq_min_logdv,
                price_col=price_col,
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
                # Save immediately when a new best is found (eliminates retrain mismatch).
                if best_model_path is not None:
                    best_model_path.parent.mkdir(parents=True, exist_ok=True)
                    booster.save_model(str(best_model_path))
                    meta_path = best_model_path.with_suffix(".meta.yaml")
                    meta = {
                        "created_at": ts,
                        "task": "rank",
                        "source": "minimal_gpu_tuner.tune_ranker_sharpe",
                        "dataset_path": str(ds_path),
                        "target_col": tgt,
                        "feature_cols": list(feats),
                        "selection_metric": {"primary": "sharpe", "secondary": "cum_return"},
                        "best_so_far": {
                            "trial": int(r.get("trial", -1)),
                            "sharpe": float(r.get("sharpe")),
                            "cum_return": float(r.get("cum_return")),
                            "max_drawdown": float(r.get("max_drawdown")),
                            "avg_turnover": float(r.get("avg_turnover")),
                        },
                        "xgb_params": dict(params),
                        "num_boost_round": int(num_round),
                        "best_iteration": int(r.get("best_iteration", -1)),
                    }
                    _write_yaml(meta_path, meta)
                    best_saved = True

                # Optional early stop once we hit the target Sharpe
                if args.stop_at_sharpe is not None and float(r["sharpe"]) >= float(args.stop_at_sharpe):
                    print(f"[early_stop] reached sharpe={float(r['sharpe']):.3f} >= {float(args.stop_at_sharpe):.3f}")
                    break

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
    if best_model_path is not None:
        if best_saved:
            print(f"saved_best_model={best_model_path}")
            print(f"saved_best_meta={best_model_path.with_suffix('.meta.yaml')}")
        else:
            print("saved_best_model=<none> (no successful trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


