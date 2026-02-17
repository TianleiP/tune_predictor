from __future__ import annotations

import argparse
import json
from dataclasses import asdict
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

from minimal_gpu_tuner.overlap_topn_official import (  # noqa: E402
    run_overlap_equal_weight_backtest,
    run_overlap_topn_backtest,
)
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


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    # Match `scripts/blend_strategy_compare.py` behavior (ddof=0, std==0 -> NaN)
    x = pd.to_numeric(df[col], errors="coerce")
    g = df.groupby("date", sort=True)[col]
    mu = g.transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    sd = g.transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0))
    sd = sd.replace(0, np.nan)
    return (x - mu) / sd


def _parse_list_floats(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        t = part.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("empty float list")
    return out


def _month_add(dt: pd.Timestamp, months: int) -> pd.Timestamp:
    return (dt + pd.DateOffset(months=months)).normalize()


def _split_train_internal_eval(
    tr: pd.DataFrame, *, eval_months: int, fallback_frac: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split training set into fit/eval by time:
    - eval = last `eval_months` months before train end
    - fallback to last `fallback_frac` of unique dates if month split degenerates
    """
    if tr.empty:
        raise RuntimeError("train is empty")
    end = pd.to_datetime(tr["date"].max()).normalize()
    eval_start = _month_add(end, -int(eval_months))
    tr_fit = tr[tr["date"] < eval_start].copy()
    tr_eval = tr[tr["date"] >= eval_start].copy()
    if not tr_fit.empty and not tr_eval.empty:
        return tr_fit, tr_eval, eval_start

    uniq = sorted(tr["date"].dropna().unique())
    if len(uniq) < 50:
        raise RuntimeError("not enough training dates to create internal eval split")
    cut = int(len(uniq) * (1.0 - float(fallback_frac)))
    cut_dt = pd.to_datetime(uniq[cut]).normalize()
    tr_fit = tr[tr["date"] < cut_dt].copy()
    tr_eval = tr[tr["date"] >= cut_dt].copy()
    if tr_fit.empty or tr_eval.empty:
        raise RuntimeError("failed internal eval split")
    return tr_fit, tr_eval, cut_dt


def _make_price_panel(df: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    cols = [c for c in ["date", "ticker", price_col, "amihud_illiq", "log_dollar_volume"] if c in df.columns]
    if price_col not in cols:
        raise RuntimeError(f"missing price_col={price_col} in df")
    return df[cols].copy()


def _make_scored_panel(df: pd.DataFrame, *, score_col: str) -> pd.DataFrame:
    cols = [c for c in ["date", "ticker", score_col, "amihud_illiq", "log_dollar_volume"] if c in df.columns]
    out = df[cols].copy()
    out.rename(columns={score_col: "score"}, inplace=True)
    return out


def _bt_summary(
    name: str,
    *,
    scored_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    price_col: str,
    topn: int,
    hold_days: int,
    cost_bps: float,
    slippage_k: float,
    slippage_cap_bps: float,
    liq_min_logdv: Optional[float],
) -> Dict[str, Any]:
    _, summ = run_overlap_topn_backtest(
        scored_panel=scored_panel,
        price_panel=price_panel,
        top_n=int(topn),
        hold_days=int(hold_days),
        cost_bps=float(cost_bps),
        slippage_k=float(slippage_k),
        slippage_cap_bps=float(slippage_cap_bps),
        liquidity_min_log_dollar_volume=liq_min_logdv,
        price_col=price_col,
    )
    d = asdict(summ)
    d["name"] = name
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Anchored GPU test: train ranker on GPU (device=cuda), tune blend weight on pre-holdout, then compare vs momentum on holdout."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target-col", default="future_return_5d")
    ap.add_argument("--include-ohlcv-levels", action="store_true")

    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--holdout-start", default="2024-01-01")
    ap.add_argument("--holdout-end", default="", help="Optional holdout end date (YYYY-MM-DD)")
    ap.add_argument("--purge-days", type=int, default=10)
    ap.add_argument("--embargo-days", type=int, default=0)

    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--eval-months", type=int, default=6, help="Internal early-stopping eval window size inside training")
    ap.add_argument("--weight-tune-months", type=int, default=6, help="Tune blend weight on last N months of training")

    ap.add_argument("--topn", type=int, default=5)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-k", type=float, default=0.0)
    ap.add_argument("--slippage-cap-bps", type=float, default=50.0)
    ap.add_argument("--liq-min-logdv", type=float, default=None)
    ap.add_argument("--price-col", default="adj_close")

    ap.add_argument("--blend-weights", default="0,0.05,0.08,0.09,0.10,0.11,0.12,0.15,0.20,0.25,0.30,1.0")

    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    # Model params (can override from CLI if you want)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lambda", dest="reg_lambda", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=600)

    ap.add_argument("--out", default="", help="Output CSV path (default: artifacts/logs/anchored_gpu_<ts>.csv)")
    ap.add_argument("--save-model", default="", help="If set, save the trained Booster to this .json path")
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
        raise RuntimeError("dataset missing mom_12_1 (required)")

    price_col = str(args.price_col)
    if price_col not in df.columns:
        raise RuntimeError(f"dataset missing price_col: {price_col}")

    # Anchored split: train is pre-holdout, holdout is [holdout_start, holdout_end]
    tr, ho = _time_split(
        df,
        train_start=str(args.train_start),
        valid_start=str(args.holdout_start),
        valid_end=(str(args.holdout_end).strip() or None),
    )
    tr, ho = _purge_embargo(tr, ho, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

    feats = _pick_features(df)
    feats = _apply_feature_policy(feats, include_ohlcv_levels=bool(args.include_ohlcv_levels))

    # Internal eval split inside training (for early stopping)
    tr_fit, tr_eval, eval_start_dt = _split_train_internal_eval(tr, eval_months=int(args.eval_months))

    # Rank labels
    bins = int(args.bins)
    y_fit = _to_relevance_bins(tr_fit, date_col="date", target_col=tgt, bins=bins)
    y_eval = _to_relevance_bins(tr_eval, date_col="date", target_col=tgt, bins=bins)
    g_fit = _make_groups(tr_fit["date"])
    g_eval = _make_groups(tr_eval["date"])

    X_fit = tr_fit[feats].replace([np.inf, -np.inf], np.nan)
    X_eval = tr_eval[feats].replace([np.inf, -np.inf], np.nan)
    dfit = xgb.DMatrix(X_fit, label=y_fit)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    dfit.set_group(g_fit)
    deval.set_group(g_eval)

    params: Dict[str, Any] = {
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

    booster = xgb.train(
        params=params,
        dtrain=dfit,
        num_boost_round=num_round,
        evals=[(deval, "train_eval")],
        early_stopping_rounds=int(args.early_stopping_rounds) if int(args.early_stopping_rounds) > 0 else None,
        verbose_eval=False,
    )

    best_it = int(getattr(booster, "best_iteration", -1))
    best_rounds = (best_it + 1) if best_it >= 0 else int(num_round)
    if best_rounds <= 0:
        best_rounds = int(num_round)

    # Retrain on FULL anchored train window using best_rounds (more realistic for deployment)
    y_tr = _to_relevance_bins(tr, date_col="date", target_col=tgt, bins=bins)
    g_tr = _make_groups(tr["date"])
    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dtr.set_group(g_tr)
    booster_full = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(best_rounds),
        evals=[],
        verbose_eval=False,
    )
    it_range = None  # full model already trained to fixed rounds

    # For tuning w: predict on last `weight_tune_months` of training (subset of tr_eval)
    wt_months = int(args.weight_tune_months)
    tune_start = _month_add(pd.to_datetime(str(args.holdout_start)).normalize(), -wt_months)
    tune = tr[tr["date"] >= tune_start].copy()
    if tune.empty:
        tune = tr_eval.copy()

    X_tune = tune[feats].replace([np.inf, -np.inf], np.nan)
    dtune = xgb.DMatrix(X_tune)
    tune["model_score"] = booster_full.predict(dtune)
    tune["mom_z"] = _zscore_by_date(tune, "mom_12_1")
    tune["model_z"] = _zscore_by_date(tune, "model_score")

    weights = _parse_list_floats(args.blend_weights)
    tune_price_panel = _make_price_panel(tune, price_col=price_col)
    tune_rows: List[Dict[str, Any]] = []
    best_w: float = weights[0]
    best_key: Optional[Tuple[float, float]] = None
    for w in weights:
        tune["blend_score"] = float(w) * tune["mom_z"] + (1.0 - float(w)) * tune["model_z"]
        d = _bt_summary(
            f"tune_blend_w{float(w):.2f}",
            scored_panel=_make_scored_panel(tune, score_col="blend_score"),
            price_panel=tune_price_panel,
            price_col=price_col,
            topn=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liq_min_logdv=(float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
        )
        d["w"] = float(w)
        tune_rows.append(d)
        key = (float(d["sharpe"]), float(d["cum_return"]))
        if best_key is None or key > best_key:
            best_key = key
            best_w = float(w)

    # Holdout predictions
    X_ho = ho[feats].replace([np.inf, -np.inf], np.nan)
    dho = xgb.DMatrix(X_ho)
    ho = ho.copy()
    ho["model_score"] = booster_full.predict(dho)
    ho["mom_z"] = _zscore_by_date(ho, "mom_12_1")
    ho["model_z"] = _zscore_by_date(ho, "model_score")
    ho["blend_score"] = float(best_w) * ho["mom_z"] + (1.0 - float(best_w)) * ho["model_z"]

    price_panel = _make_price_panel(ho, price_col=price_col)

    results: List[Dict[str, Any]] = []
    results.append(
        _bt_summary(
            f"holdout_blend_w{best_w:.2f}",
            scored_panel=_make_scored_panel(ho, score_col="blend_score"),
            price_panel=price_panel,
            price_col=price_col,
            topn=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liq_min_logdv=(float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
        )
    )
    results.append(
        _bt_summary(
            "holdout_model_only",
            scored_panel=_make_scored_panel(ho, score_col="model_score"),
            price_panel=price_panel,
            price_col=price_col,
            topn=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liq_min_logdv=(float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
        )
    )
    results.append(
        _bt_summary(
            "holdout_mom_12_1",
            scored_panel=_make_scored_panel(ho, score_col="mom_12_1"),
            price_panel=price_panel,
            price_col=price_col,
            topn=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liq_min_logdv=(float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
        )
    )
    _, eq = run_overlap_equal_weight_backtest(
        price_panel=price_panel,
        price_col=price_col,
        hold_days=int(args.hold_days),
        cost_bps=float(args.cost_bps),
        slippage_k=float(args.slippage_k),
        slippage_cap_bps=float(args.slippage_cap_bps),
        liquidity_min_log_dollar_volume=(float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
    )
    eqd = asdict(eq)
    eqd["name"] = "holdout_equal_weight"
    results.append(eqd)

    # Outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if str(args.out).strip() else (Path("artifacts") / "logs" / f"anchored_gpu_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tune_path = out_path.with_name(out_path.stem + "_weight_tune.csv")
    meta_path = out_path.with_suffix(".meta.json")

    out_df = pd.DataFrame(results)
    out_df["best_w"] = float(best_w)
    out_df["eval_start_dt"] = str(pd.to_datetime(eval_start_dt).strftime("%Y-%m-%d"))
    out_df["weight_tune_start"] = str(pd.to_datetime(tune_start).strftime("%Y-%m-%d"))
    out_df["dataset"] = str(ds_path)
    out_df["target_col"] = tgt
    out_df["price_col"] = price_col
    out_df["n_features"] = int(len(feats))
    out_df.to_csv(out_path, index=False)
    pd.DataFrame(tune_rows).to_csv(tune_path, index=False)

    meta = {
        "created_at": ts,
        "env": _env_fingerprint(),
        "args": vars(args),
        "dataset": {"path": str(ds_path), "fingerprint": _dataset_fingerprint(df, target_col=tgt)},
        "splits": {
            "train_start": str(args.train_start),
            "holdout_start": str(args.holdout_start),
            "holdout_end": str(args.holdout_end).strip() or None,
            "purge_days": int(args.purge_days),
            "embargo_days": int(args.embargo_days),
        },
        "features": {"count": int(len(feats)), "include_ohlcv_levels": bool(args.include_ohlcv_levels)},
        "label": {"target_col": tgt, "bins": int(bins)},
        "xgb_params": params,
        "num_boost_round": int(num_round),
        "best_iteration": int(best_it),
        "best_rounds_used": int(best_rounds),
        "blend": {"weights": weights, "best_w": float(best_w), "weight_tune_start": str(pd.to_datetime(tune_start).strftime("%Y-%m-%d"))},
        "paths": {"out_csv": str(out_path), "weight_tune_csv": str(tune_path)},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    # Save exact booster (eliminates CPU/GPU retrain mismatch)
    if str(args.save_model).strip():
        model_path = Path(str(args.save_model))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster_full.save_model(str(model_path))
        meta_yaml = model_path.with_suffix(".meta.yaml")
        _write_yaml(
            meta_yaml,
            {
                "created_at": ts,
                "task": "rank",
                "source": "minimal_gpu_tuner.anchored_holdout_gpu",
                "dataset_path": str(ds_path),
                "target_col": tgt,
                "feature_cols": list(feats),
                "xgb_params": dict(params),
                "num_boost_round": int(num_round),
                "best_iteration": int(best_it),
                "best_rounds_used": int(best_rounds),
                "anchored": {
                    "train_start": str(args.train_start),
                    "holdout_start": str(args.holdout_start),
                    "holdout_end": (str(args.holdout_end).strip() or None),
                    "purge_days": int(args.purge_days),
                    "embargo_days": int(args.embargo_days),
                    "internal_eval_start": str(pd.to_datetime(eval_start_dt).strftime("%Y-%m-%d")),
                },
                "blend": {"best_w": float(best_w), "weights": weights},
                "env": _env_fingerprint(),
                "dataset_fingerprint": _dataset_fingerprint(df, target_col=tgt),
            },
        )

    print(f"meta={meta_path}")
    print(f"output={out_path}")
    print(f"weight_tune={tune_path}")
    print(f"n_train={len(tr)} n_holdout={len(ho)} n_features={len(feats)}")
    print(f"best_iteration={best_it} best_rounds_used={best_rounds} internal_eval_start={pd.to_datetime(eval_start_dt).strftime('%Y-%m-%d')}")
    print(f"best_w={best_w:.4f}")
    print(out_df[["name", "sharpe", "cum_return", "max_drawdown", "avg_turnover"]].sort_values("sharpe", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


