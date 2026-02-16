from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
    _purge_embargo,
    _time_split,
    _to_relevance_bins,
)


def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"yaml not found: {p}")
    obj = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"yaml must be a mapping: {p}")
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_yaml(path: Path, obj: dict) -> None:
    _atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False))


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    def _one(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").astype(float)
        sd = float(x.std(ddof=0))
        if sd == 0.0 or np.isnan(sd):
            return pd.Series(np.nan, index=s.index)
        return (x - float(x.mean())) / sd

    return df.groupby("date", sort=True)[col].transform(_one)


def _default_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    cols = list(map(str, feature_cols))
    groups: Dict[str, List[str]] = {
        "returns": [c for c in cols if c.startswith("ret_")],
        "moving_avg_gap": [c for c in cols if c.startswith("ma_") and c.endswith("_gap")],
        "momentum_12_1": [c for c in cols if c == "mom_12_1"],
        "volatility": [c for c in cols if c.startswith("vol_")],
        "range_gap": [c for c in cols if c in {"range_pct", "gap_open", "overnight_ret", "intraday_ret"}],
        "volume_activity": [c for c in cols if (c == "vol_1d_chg" or c.startswith("vol_ratio_"))],
        "liquidity": [c for c in cols if c in {"dollar_volume", "log_volume", "log_dollar_volume", "amihud_illiq"}],
        "reversal": [c for c in cols if c.startswith("rev_")],
        "sector_industry_codes": [c for c in cols if c in {"sector_code", "industry_code"}],
        "sector_relative": [c for c in cols if (c.startswith("sector_ret_") or c.startswith("rel_to_sector_ret_"))],
    }
    return {k: v for k, v in groups.items() if v}


def _train_booster_and_score(
    *,
    tr: pd.DataFrame,
    va: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    bins: int,
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[xgb.Booster, np.ndarray, int]:
    X_tr = tr[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_va = va[feature_cols].replace([np.inf, -np.inf], np.nan)

    y_tr = _to_relevance_bins(tr, date_col="date", target_col=target_col, bins=bins)
    y_va = _to_relevance_bins(va, date_col="date", target_col=target_col, bins=bins)
    g_tr = _make_groups(tr["date"])
    g_va = _make_groups(va["date"])

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    dtr.set_group(g_tr)
    dva.set_group(g_va)

    booster = xgb.train(
        params=dict(params),
        dtrain=dtr,
        num_boost_round=int(num_boost_round),
        evals=[(dva, "valid")],
        early_stopping_rounds=int(early_stopping_rounds) if int(early_stopping_rounds) > 0 else None,
        verbose_eval=False,
    )
    best_it = booster.best_iteration
    if best_it is not None:
        pred = booster.predict(dva, iteration_range=(0, int(best_it) + 1))
        return booster, pred, int(best_it)
    pred = booster.predict(dva)
    return booster, pred, -1


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU feature ablation for ranker using official overlap backtest (inside tune_predictor).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--base-model-meta", required=True, help="Meta YAML from tune_ranker_sharpe.py (has feature_cols + xgb_params + num_boost_round).")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="")
    ap.add_argument("--purge-days", type=int, default=10)
    ap.add_argument("--embargo-days", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--include-ohlcv-levels", action="store_true")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--hold-days", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--slippage-k", type=float, default=0.0)
    ap.add_argument("--slippage-cap-bps", type=float, default=50.0)
    ap.add_argument("--liq-min-logdv", type=float, default=None)
    ap.add_argument("--price-col", default="adj_close")
    ap.add_argument("--blend-weight", type=float, default=0.09, help="Evaluate blend at this w for mom_12_1. Set <0 to skip.")
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument(
        "--save-model-dir",
        default="",
        help="If set, save each run's trained Booster + meta.yaml under this directory (so you can validate locally without retraining).",
    )
    ap.add_argument(
        "--save-prefix",
        default="xgb_rank_ablate",
        help="Filename prefix used when saving models (only when --save-model-dir is set).",
    )
    ap.add_argument("--out", default="", help="Output CSV (default: artifacts/logs/feature_ablation_<ts>.csv)")
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

    meta = _read_yaml(args.base_model_meta)
    base_cols = list(meta.get("feature_cols") or [])
    if not base_cols:
        raise RuntimeError(f"base meta missing feature_cols: {args.base_model_meta}")

    # Pull fixed hyperparams from the tuned model meta (preferred).
    params = dict(meta.get("xgb_params") or {})
    if not params:
        raise RuntimeError(f"base meta missing xgb_params: {args.base_model_meta}")
    num_boost_round = int(meta.get("num_boost_round") or 600)

    # Allow overriding split/bins via CLI (useful to match main repo)
    tr, va = _time_split(df, train_start=args.train_start, valid_start=args.valid_start, valid_end=(args.valid_end or None))
    tr, va = _purge_embargo(tr, va, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

    # Ensure feature policy matches main repo (no raw levels by default)
    base_cols = _apply_feature_policy(base_cols, include_ohlcv_levels=bool(args.include_ohlcv_levels))
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"dataset missing {len(missing)} base features (first 10): {missing[:10]}")

    # Price panel for official backtest (use valid only)
    price_col = str(args.price_col)
    if price_col not in va.columns:
        raise RuntimeError(f"dataset missing price_col: {price_col}")
    price_cols = [c for c in ["date", "ticker", price_col, "amihud_illiq", "log_dollar_volume"] if c in va.columns]
    price_panel = va[price_cols].copy()

    groups = _default_feature_groups(base_cols)
    group_names = list(groups.keys())
    runs = 1 + len(group_names)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if str(args.out).strip() else (Path("artifacts") / "logs" / f"feature_ablation_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dir = Path(str(args.save_model_dir)) if str(args.save_model_dir).strip() else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"dataset={ds_path} train={len(tr)} valid={len(va)} feats={len(base_cols)} target={tgt}")
    print(f"groups={group_names} runs={runs} device={params.get('device', 'cpu')} tree_method={params.get('tree_method')}")
    print(f"out={out_path}")

    rows: List[Dict[str, Any]] = []

    def _eval_one(name: str, feat_cols: List[str], dropped: List[str], i: int) -> None:
        t0 = time.time()
        dropped_s = "|".join(dropped) if dropped else "-"
        print(f"[{i}/{runs}] start {name} n_features={len(feat_cols)} dropped={dropped_s}", flush=True)

        booster, score, best_it = _train_booster_and_score(
            tr=tr,
            va=va,
            feature_cols=feat_cols,
            target_col=tgt,
            bins=int(args.bins),
            params=params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=int(args.early_stopping_rounds),
        )

        scored_cols = [c for c in ["date", "ticker", "amihud_illiq", "log_dollar_volume"] if c in va.columns]
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

        r: Dict[str, Any] = {
            "name": name,
            "n_features": int(len(feat_cols)),
            "dropped_groups": dropped_s,
            "model_sharpe": float(summ.sharpe),
            "model_cum_return": float(summ.cum_return),
            "model_max_drawdown": float(summ.max_drawdown),
            "model_avg_turnover": float(summ.avg_turnover),
            "best_iteration": int(best_it),
        }

        w = float(args.blend_weight)
        if w >= 0.0:
            if "mom_12_1" not in va.columns:
                raise RuntimeError("Dataset missing mom_12_1; cannot compute blend.")
            base = va[["date", "ticker", "mom_12_1"] + [c for c in ["amihud_illiq", "log_dollar_volume"] if c in va.columns]].copy()
            base["model_score"] = score
            base["mom_z"] = _zscore_by_date(base, "mom_12_1")
            base["model_z"] = _zscore_by_date(base, "model_score")
            base["score"] = w * base["mom_z"] + (1.0 - w) * base["model_z"]
            blend_scored = base[["date", "ticker", "score"] + [c for c in ["amihud_illiq", "log_dollar_volume"] if c in base.columns]].copy()
            _, summ_b = run_overlap_topn_backtest(
                scored_panel=blend_scored,
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
                    "blend_weight": w,
                    "blend_sharpe": float(summ_b.sharpe),
                    "blend_cum_return": float(summ_b.cum_return),
                    "blend_max_drawdown": float(summ_b.max_drawdown),
                    "blend_avg_turnover": float(summ_b.avg_turnover),
                }
            )

        rows.append(r)
        dt = time.time() - t0
        msg = f"[{i}/{runs}] done {name} model_sharpe={r['model_sharpe']:.4f}"
        if "blend_sharpe" in r:
            msg += f" blend_sharpe={float(r['blend_sharpe']):.4f}"
        msg += f" elapsed_s={dt:.1f}"
        print(msg, flush=True)

        # Optionally save the exact trained model + meta for local validation without retraining.
        if save_dir is not None:
            safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(name))
            model_path = save_dir / f"{str(args.save_prefix)}_{safe}.json"
            # Save the exact Booster used for scoring/backtest (no retraining -> no mismatch).
            booster.save_model(str(model_path))
            meta_path = model_path.with_suffix(".meta.yaml")
            meta_out = {
                "created_at": ts,
                "task": "rank",
                "source": "minimal_gpu_tuner.feature_ablation_gpu",
                "dataset_path": str(ds_path),
                "target_col": tgt,
                "feature_cols": list(map(str, feat_cols)),
                "split": {
                    "train_start": str(args.train_start),
                    "valid_start": str(args.valid_start),
                    "valid_end": (str(args.valid_end) if str(args.valid_end).strip() else None),
                    "purge_days": int(args.purge_days),
                    "embargo_days": int(args.embargo_days),
                },
                "ranking": {"relevance_bins": int(args.bins)},
                "backtest": {
                    "topn": int(args.topn),
                    "hold_days": int(args.hold_days),
                    "cost_bps": float(args.cost_bps),
                    "slippage_k": float(args.slippage_k),
                    "slippage_cap_bps": float(args.slippage_cap_bps),
                    "liq_min_logdv": (float(args.liq_min_logdv) if args.liq_min_logdv is not None else None),
                    "price_col": price_col,
                    "blend_weight": float(args.blend_weight),
                },
                "env": _env_fingerprint(),
                "dataset_fingerprint": _dataset_fingerprint(df, target_col=tgt),
                "xgb_params": dict(params),
                "num_boost_round": int(num_boost_round),
                "best_iteration": int(getattr(booster, "best_iteration", best_it)),
                "result": dict(r),
            }
            _write_yaml(meta_path, meta_out)
            print(f"saved_model={model_path}", flush=True)

    i = 1
    _eval_one("baseline_all_features", base_cols, [], i=i)
    for g in group_names:
        i += 1
        drop = set(groups[g])
        cols = [c for c in base_cols if c not in drop]
        if len(cols) < 5:
            continue
        _eval_one(f"drop_{g}", cols, [g], i=i)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"output={out_path}")
    print(out_df.sort_values(["blend_sharpe", "model_sharpe"], ascending=False, na_position="last").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


