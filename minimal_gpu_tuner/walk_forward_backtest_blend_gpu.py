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

from minimal_gpu_tuner.overlap_topn_official import run_overlap_topn_backtest  # noqa: E402
from minimal_gpu_tuner.tune_ranker_min import (  # noqa: E402
    _apply_feature_policy,
    _dataset_fingerprint,
    _env_fingerprint,
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
    train_years: int,
    test_months: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
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


def _parse_weights(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(","):
        t = part.strip()
        if not t:
            continue
        out.append(float(t))
    if not out:
        raise ValueError("empty --blend-weights")
    return out


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    g = df.groupby("date", sort=True)[col]
    mu = g.transform(lambda s: pd.to_numeric(s, errors="coerce").mean())
    sd = g.transform(lambda s: pd.to_numeric(s, errors="coerce").std(ddof=0))
    sd = sd.replace(0, np.nan)
    return (x - mu) / sd


def _curve_rescale(curve: pd.DataFrame, *, start_equity: float) -> pd.DataFrame:
    c = curve.copy()
    # curve["equity"] is relative to 1.0; rescale to start_equity
    c["equity"] = start_equity * pd.to_numeric(c["equity"], errors="coerce")
    return c


def _overall_from_daily_returns(daily_returns: pd.Series) -> dict:
    r = pd.to_numeric(daily_returns, errors="coerce").fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    days = int(len(r))
    cum_return = float(equity.iloc[-1] - 1.0) if days else float("nan")
    ann_factor = 252.0
    ann_return = float((equity.iloc[-1]) ** (ann_factor / days) - 1.0) if days > 0 else float("nan")
    ann_vol = float(r.std(ddof=0) * np.sqrt(ann_factor)) if days > 1 else float("nan")
    sharpe = float((r.mean() * ann_factor) / (r.std(ddof=0) * np.sqrt(ann_factor))) if r.std(ddof=0) != 0 else float("nan")
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min()) if days else float("nan")
    return {
        "days": days,
        "cum_return": cum_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GPU walk-forward portfolio backtest (Top-N overlap) for fixed blend weights vs mom_12_1 baseline."
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target-col", default="future_return_5d")
    ap.add_argument("--include-ohlcv-levels", action="store_true")

    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="", help="Optional last date (YYYY-MM-DD). Default: dataset max.")

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

    ap.add_argument("--blend-weights", default="0.09,0.10", help="Comma-separated fixed w values to evaluate on each block.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stopping-rounds", type=int, default=0, help="0 disables early stopping.")
    ap.add_argument("--num-boost-round", type=int, default=600)
    ap.add_argument(
        "--xgb-meta",
        default="",
        help="Optional: path to a saved model .meta.yaml containing xgb_params + num_boost_round (overrides CLI params).",
    )

    # Default params if no meta is provided
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--lambda", dest="reg_lambda", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=1.0)

    ap.add_argument("--out", default="", help="Output per-block CSV (default: artifacts/logs/wf_bt_gpu_<ts>.csv)")
    ap.add_argument("--out-overall", default="", help="Optional overall summary CSV (single row per strategy).")
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
        raise RuntimeError("dataset missing mom_12_1 (required for blend baseline)")

    price_col = str(args.price_col)
    if price_col not in df.columns:
        raise RuntimeError(f"dataset missing price_col: {price_col}")

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
        train_years=int(args.train_years),
        test_months=int(args.test_months),
        start=start,
        end=end,
    )
    if not blocks:
        raise RuntimeError("no blocks produced; check --start/--end")

    weights = _parse_weights(args.blend_weights)

    # Collect per-block summaries and stitched curves per strategy
    stitched_daily_rets: Dict[str, List[float]] = {}
    stitched_dates: Dict[str, List[str]] = {}
    per_block_rows: List[Dict[str, Any]] = []

    for bi, (train_start, train_end, test_start, test_end) in enumerate(blocks):
        tr = df[(df["date"] >= train_start) & (df["date"] < train_end)].copy()
        te = df[(df["date"] >= test_start) & (df["date"] < test_end)].copy()
        if tr.empty or te.empty:
            continue

        tr, te = _purge_embargo(tr, te, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

        # Train ranker
        bins = int(args.bins)
        y_tr = _to_relevance_bins(tr, date_col="date", target_col=tgt, bins=bins)
        g_tr = _make_groups(tr["date"])
        X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dtr.set_group(g_tr)

        # Optional early stopping uses a small internal slice from the END of training block
        esr = int(args.early_stopping_rounds)
        best_rounds_used = int(num_round)
        if esr > 0:
            # Split last ~20% of unique dates as eval
            uniq = sorted(tr["date"].dropna().unique())
            cut = int(len(uniq) * 0.8)
            cut_dt = pd.to_datetime(uniq[cut]).normalize() if cut < len(uniq) else pd.to_datetime(uniq[-1]).normalize()
            tr_fit = tr[tr["date"] < cut_dt].copy()
            tr_eval = tr[tr["date"] >= cut_dt].copy()
            if not tr_fit.empty and not tr_eval.empty:
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
                tmp = xgb.train(
                    params=params,
                    dtrain=dfit,
                    num_boost_round=int(num_round),
                    evals=[(deval, "train_eval")],
                    early_stopping_rounds=esr,
                    verbose_eval=False,
                )
                best_it = int(getattr(tmp, "best_iteration", -1))
                if best_it >= 0:
                    best_rounds_used = int(best_it + 1)

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=int(best_rounds_used),
            evals=[],
            verbose_eval=False,
        )

        # Predict model score on test block
        X_te = te[feats].replace([np.inf, -np.inf], np.nan)
        dte = xgb.DMatrix(X_te)
        te = te.copy()
        te["model_score"] = booster.predict(dte)
        te["mom_z"] = _zscore_by_date(te, "mom_12_1")
        te["model_z"] = _zscore_by_date(te, "model_score")

        price_cols = [c for c in ["date", "ticker", price_col, "amihud_illiq", "log_dollar_volume"] if c in te.columns]
        price_panel = te[price_cols].copy()

        # Momentum baseline (no training)
        scored_mom = te[[c for c in ["date", "ticker", "mom_12_1", "amihud_illiq", "log_dollar_volume"] if c in te.columns]].copy()
        scored_mom.rename(columns={"mom_12_1": "score"}, inplace=True)
        curve_m, summ_m = run_overlap_topn_backtest(
            scored_panel=scored_mom,
            price_panel=price_panel,
            top_n=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liquidity_min_log_dollar_volume=args.liq_min_logdv,
            price_col=price_col,
        )
        m_name = "mom_12_1"
        stitched_daily_rets.setdefault(m_name, []).extend(pd.to_numeric(curve_m["daily_return"], errors="coerce").fillna(0.0).tolist())
        stitched_dates.setdefault(m_name, []).extend(curve_m["date"].astype(str).tolist())
        per_block_rows.append(
            {
                "block": int(bi),
                "strategy": m_name,
                "w": np.nan,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": (train_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": (test_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "best_rounds_used": int(best_rounds_used),
                **asdict(summ_m),
            }
        )

        # Model-only (zscore doesn't change rank; but keep it explicit)
        scored_model = te[[c for c in ["date", "ticker", "model_score", "amihud_illiq", "log_dollar_volume"] if c in te.columns]].copy()
        scored_model.rename(columns={"model_score": "score"}, inplace=True)
        curve_md, summ_md = run_overlap_topn_backtest(
            scored_panel=scored_model,
            price_panel=price_panel,
            top_n=int(args.topn),
            hold_days=int(args.hold_days),
            cost_bps=float(args.cost_bps),
            slippage_k=float(args.slippage_k),
            slippage_cap_bps=float(args.slippage_cap_bps),
            liquidity_min_log_dollar_volume=args.liq_min_logdv,
            price_col=price_col,
        )
        md_name = "model_only"
        stitched_daily_rets.setdefault(md_name, []).extend(pd.to_numeric(curve_md["daily_return"], errors="coerce").fillna(0.0).tolist())
        stitched_dates.setdefault(md_name, []).extend(curve_md["date"].astype(str).tolist())
        per_block_rows.append(
            {
                "block": int(bi),
                "strategy": md_name,
                "w": np.nan,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": (train_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": (test_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "best_rounds_used": int(best_rounds_used),
                **asdict(summ_md),
            }
        )

        # Blend fixed weights
        for w in weights:
            te["blend_score"] = float(w) * te["mom_z"] + (1.0 - float(w)) * te["model_z"]
            scored = te[[c for c in ["date", "ticker", "blend_score", "amihud_illiq", "log_dollar_volume"] if c in te.columns]].copy()
            scored.rename(columns={"blend_score": "score"}, inplace=True)
            curve_b, summ_b = run_overlap_topn_backtest(
                scored_panel=scored,
                price_panel=price_panel,
                top_n=int(args.topn),
                hold_days=int(args.hold_days),
                cost_bps=float(args.cost_bps),
                slippage_k=float(args.slippage_k),
                slippage_cap_bps=float(args.slippage_cap_bps),
                liquidity_min_log_dollar_volume=args.liq_min_logdv,
                price_col=price_col,
            )
            b_name = f"blend_w{float(w):.2f}"
            stitched_daily_rets.setdefault(b_name, []).extend(pd.to_numeric(curve_b["daily_return"], errors="coerce").fillna(0.0).tolist())
            stitched_dates.setdefault(b_name, []).extend(curve_b["date"].astype(str).tolist())
            per_block_rows.append(
                {
                    "block": int(bi),
                    "strategy": b_name,
                    "w": float(w),
                    "train_start": train_start.strftime("%Y-%m-%d"),
                    "train_end": (train_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "test_start": test_start.strftime("%Y-%m-%d"),
                    "test_end": (test_end - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "best_rounds_used": int(best_rounds_used),
                    **asdict(summ_b),
                }
            )

    if not per_block_rows:
        raise RuntimeError("no backtest blocks produced results")

    out_block = pd.DataFrame(per_block_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if str(args.out).strip() else (Path("artifacts") / "logs" / f"wf_bt_gpu_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_block.to_csv(out_path, index=False)

    # Overall stitched metrics
    overall_rows: List[Dict[str, Any]] = []
    for name, rets in stitched_daily_rets.items():
        s = _overall_from_daily_returns(pd.Series(rets, dtype="float64"))
        overall_rows.append({"strategy": name, **s})
    out_overall = pd.DataFrame(overall_rows).sort_values("sharpe", ascending=False)

    out_overall_path = Path(args.out_overall) if str(args.out_overall).strip() else out_path.with_name(out_path.stem + "_overall.csv")
    out_overall.to_csv(out_overall_path, index=False)

    meta = {
        "created_at": ts,
        "env": _env_fingerprint(),
        "args": vars(args),
        "dataset": {"path": str(ds_path), "fingerprint": _dataset_fingerprint(df, target_col=tgt)},
        "n_features": int(len(feats)),
        "xgb_params": params,
        "num_boost_round": int(num_round),
        "blend_weights": weights,
        "outputs": {"per_block_csv": str(out_path), "overall_csv": str(out_overall_path)},
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"meta={meta_path}")
    print(f"output_blocks={out_path}")
    print(f"output_overall={out_overall_path}")
    print(out_overall.round(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


