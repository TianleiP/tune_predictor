from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

from minimal_gpu_tuner.overlap_topn_official import run_overlap_topn_backtest  # noqa: E402
from minimal_gpu_tuner.tune_ranker_min import (  # noqa: E402
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
        out[str(k)] = v if isinstance(v, list) else [v]
    return out


def _iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield {keys[i]: combo[i] for i in range(len(keys))}


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Match `scripts/blend_strategy_compare.py` behavior:
    z = (x - mean_by_date) / std_by_date with ddof=0, and std==0 -> NaN.
    """
    x = pd.to_numeric(df[col], errors="coerce")
    g = df.groupby("date", sort=True)[col]
    mu = g.transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    sd = g.transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0))
    sd = sd.replace(0, np.nan)
    return (x - mu) / sd


def _parse_weights(s: str) -> List[float]:
    ws = []
    for part in str(s).split(","):
        t = part.strip()
        if not t:
            continue
        ws.append(float(t))
    if not ws:
        raise ValueError("--blend-weights produced empty list")
    return ws


def _select_best(
    rows: List[Dict[str, Any]], *, primary: str, secondary: str
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for r in rows:
        if "error" in r:
            continue
        if primary not in r or secondary not in r:
            continue
        if np.isnan(float(r[primary])):
            continue
        key = (float(r[primary]), float(r.get(secondary, -np.inf)))
        if best is None:
            best = r
        else:
            key_best = (float(best[primary]), float(best.get(secondary, -np.inf)))
            if key > key_best:
                best = r
    return best


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tune XGB ranker hyperparams by BLEND portfolio Sharpe (w*z(mom_12_1) + (1-w)*z(model_score))."
    )
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

    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-k", type=float, default=0.0)
    ap.add_argument("--slippage-cap-bps", type=float, default=50.0)
    ap.add_argument("--liq-min-logdv", type=float, default=None)
    ap.add_argument("--price-col", default="adj_close")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)

    ap.add_argument(
        "--blend-weights",
        default="0.05,0.08,0.10,0.12,0.15,0.20",
        help="Comma-separated w values for momentum (we pick the best per trial).",
    )
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--out", default="", help="Output CSV (default: artifacts/logs/tune_blend_sharpe_<ts>.csv)")
    ap.add_argument(
        "--save-best-model",
        default="",
        help="If set, overwrite this path whenever a new best BLEND Sharpe is found.",
    )
    ap.add_argument(
        "--stop-at-sharpe",
        type=float,
        default=None,
        help="If set, stop early once best_blend_sharpe >= this value.",
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

    if "mom_12_1" not in df.columns:
        raise RuntimeError("dataset missing mom_12_1 (required for blending objective)")

    tr, va = _time_split(df, train_start=args.train_start, valid_start=args.valid_start, valid_end=(args.valid_end or None))
    tr, va = _purge_embargo(tr, va, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

    feats = _pick_features(df)
    feats = _apply_feature_policy(feats, include_ohlcv_levels=bool(args.include_ohlcv_levels))

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

    price_col = str(args.price_col)
    if price_col not in df.columns:
        raise RuntimeError(f"dataset missing price_col: {price_col}")
    price_cols = [c for c in ["date", "ticker", price_col, "amihud_illiq", "log_dollar_volume"] if c in va.columns]
    price_panel = va[price_cols].copy()

    scored_base_cols = [c for c in ["date", "ticker", "amihud_illiq", "log_dollar_volume", "mom_12_1"] if c in va.columns]
    scored_base = va[scored_base_cols].copy()

    weights = _parse_weights(args.blend_weights)

    grid = _read_grid(args.grid)
    combos = list(_iter_grid(grid))
    if int(args.max_trials) > 0:
        combos = combos[: int(args.max_trials)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if str(args.out).strip() else (Path("artifacts") / "logs" / f"tune_blend_sharpe_{ts}.csv")
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
    print(f"dataset={ds_path} train={len(tr)} valid={len(va)} feats={len(feats)} target={tgt}")
    print(f"topn={int(args.topn)} hold_days={int(args.hold_days)} cost_bps={float(args.cost_bps)} weights={weights}")
    print(f"trials={len(combos)} grid_keys={list(grid.keys())}")

    results: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_model_path = Path(str(args.save_best_model)) if str(args.save_best_model).strip() else None
    best_saved = False

    for i, ov in enumerate(combos, start=1):
        t0 = time.time()
        r: Dict[str, Any] = {
            "trial": int(i),
            "overrides": json.dumps(ov, sort_keys=True),
        }
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
            best_it = int(getattr(booster, "best_iteration", -1))
            if best_it >= 0:
                model_score = booster.predict(dva, iteration_range=(0, best_it + 1))
            else:
                model_score = booster.predict(dva)

            # Build per-row blend scores using per-day z-scores (consistent with official scripts)
            base = scored_base.copy()
            base["model_score"] = model_score
            base["mom_z"] = _zscore_by_date(base, "mom_12_1")
            base["model_z"] = _zscore_by_date(base, "model_score")

            best_w: Optional[float] = None
            best_summ: Optional[Any] = None
            best_key: Optional[Tuple[float, float]] = None

            for w in weights:
                scored_panel = base[["date", "ticker"] + [c for c in ["amihud_illiq", "log_dollar_volume"] if c in base.columns]].copy()
                scored_panel["score"] = float(w) * base["mom_z"] + (1.0 - float(w)) * base["model_z"]
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
                key = (float(summ.sharpe), float(summ.cum_return))
                if best_key is None or key > best_key:
                    best_key = key
                    best_w = float(w)
                    best_summ = summ

            if best_summ is None or best_w is None:
                raise RuntimeError("No successful blend evaluation for this trial")

            r.update(
                {
                    "best_w": float(best_w),
                    "blend_sharpe": float(best_summ.sharpe),
                    "blend_cum_return": float(best_summ.cum_return),
                    "blend_ann_return": float(best_summ.ann_return),
                    "blend_ann_vol": float(best_summ.ann_vol),
                    "blend_max_drawdown": float(best_summ.max_drawdown),
                    "blend_avg_turnover": float(best_summ.avg_turnover),
                    "best_iteration": int(best_it),
                    "num_boost_round": int(num_round),
                }
            )

            # Update global best and save exact Booster immediately (no retrain mismatch).
            key_trial = (float(r["blend_sharpe"]), float(r.get("blend_cum_return", -np.inf)))
            if best is None:
                best = r
            else:
                key_best = (float(best["blend_sharpe"]), float(best.get("blend_cum_return", -np.inf)))
                if key_trial > key_best:
                    best = r

            if best_model_path is not None and best is r:
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                booster.save_model(str(best_model_path))
                meta_yaml_path = best_model_path.with_suffix(".meta.yaml")
                meta_out = {
                    "created_at": ts,
                    "task": "rank",
                    "source": "minimal_gpu_tuner.tune_ranker_blend_sharpe",
                    "dataset_path": str(ds_path),
                    "target_col": tgt,
                    "feature_cols": list(feats),
                    "selection_metric": {"primary": "blend_sharpe", "secondary": "blend_cum_return"},
                    "best_so_far": {
                        "trial": int(r.get("trial", -1)),
                        "blend_sharpe": float(r.get("blend_sharpe")),
                        "blend_cum_return": float(r.get("blend_cum_return")),
                        "blend_max_drawdown": float(r.get("blend_max_drawdown")),
                        "blend_avg_turnover": float(r.get("blend_avg_turnover")),
                        "best_w": float(r.get("best_w")),
                    },
                    "blend": {"weights": list(map(float, weights)), "best_w": float(r.get("best_w"))},
                    "backtest": {
                        "topn": int(args.topn),
                        "hold_days": int(args.hold_days),
                        "cost_bps": float(args.cost_bps),
                        "slippage_k": float(args.slippage_k),
                        "slippage_cap_bps": float(args.slippage_cap_bps),
                        "liq_min_logdv": (float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
                        "price_col": price_col,
                    },
                    "env": _env_fingerprint(),
                    "dataset_fingerprint": _dataset_fingerprint(df, target_col=tgt),
                    "xgb_params": dict(params),
                    "num_boost_round": int(num_round),
                    "best_iteration": int(best_it),
                }
                _write_yaml(meta_yaml_path, meta_out)
                best_saved = True

                if args.stop_at_sharpe is not None and float(r["blend_sharpe"]) >= float(args.stop_at_sharpe):
                    print(f"[early_stop] reached blend_sharpe={float(r['blend_sharpe']):.3f} >= {float(args.stop_at_sharpe):.3f}")
                    results.append(r)
                    break

        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"

        r["fit_seconds"] = float(time.time() - t0)
        results.append(r)

        if i % 10 == 0 or i == 1 or i == len(combos):
            if best is not None:
                print(
                    f"[{i}/{len(combos)}] best_blend_sharpe={float(best['blend_sharpe']):.3f} "
                    f"best_cum={float(best.get('blend_cum_return', float('nan'))):.3f} best_w={float(best.get('best_w', float('nan'))):.2f}"
                )
            else:
                print(f"[{i}/{len(combos)}] no_success_yet")

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"output={out_path}")
    if best is not None:
        print("best_overrides=", best["overrides"])
        print(
            f"best_blend_sharpe={float(best['blend_sharpe']):.3f} best_cum_return={float(best.get('blend_cum_return', float('nan'))):.3f} best_w={float(best.get('best_w', float('nan'))):.2f}"
        )
    if best_model_path is not None:
        if best_saved:
            print(f"saved_best_model={best_model_path}")
            print(f"saved_best_meta={best_model_path.with_suffix('.meta.yaml')}")
        else:
            print("saved_best_model=<none> (no successful trials)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


