from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import platform
import sys


def _env_fingerprint() -> Dict[str, Any]:
    bi = {}
    try:
        bi = dict(xgb.build_info())
    except Exception:
        bi = {}
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "xgboost": getattr(xgb, "__version__", "unknown"),
        "xgboost_build_info": {
            # keep a small subset that's most useful
            "USE_CUDA": bi.get("USE_CUDA"),
            "USE_OPENMP": bi.get("USE_OPENMP"),
            "USE_DLOPEN_NCCL": bi.get("USE_DLOPEN_NCCL"),
        },
        "pandas": getattr(pd, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }


def _dataset_fingerprint(df: pd.DataFrame, *, target_col: str) -> Dict[str, Any]:
    """
    Create a lightweight fingerprint so two machines can confirm they're tuning the same dataset.
    Assumes df is already cleaned and includes date/ticker/target_col.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "ticker", target_col]).sort_values(["date", "ticker"]).reset_index(drop=True)

    cols_sorted = sorted(map(str, d.columns.tolist()))
    columns_sha1 = hashlib.sha1(("\n".join(cols_sorted)).encode("utf-8")).hexdigest()

    sample = d[["date", "ticker", target_col]].head(2000).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    sample_sha1 = hashlib.sha1(sample.to_csv(index=False).encode("utf-8")).hexdigest()

    return {
        "rows": int(len(d)),
        "cols": int(len(d.columns)),
        "date_min": str(d["date"].min().date()) if len(d) else None,
        "date_max": str(d["date"].max().date()) if len(d) else None,
        "tickers": int(d["ticker"].nunique()) if len(d) else 0,
        "columns_sha1": columns_sha1,
        "sample_sha1": sample_sha1,
    }


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


def _time_split(
    df: pd.DataFrame,
    *,
    train_start: str,
    valid_start: str,
    valid_end: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    t0 = pd.to_datetime(train_start)
    v0 = pd.to_datetime(valid_start)
    v1 = pd.to_datetime(valid_end) if valid_end else None
    tr = d[(d["date"] >= t0) & (d["date"] < v0)].copy()
    va = d[(d["date"] >= v0) & ((d["date"] <= v1) if v1 is not None else True)].copy()
    if tr.empty or va.empty:
        raise RuntimeError(f"empty split: train={len(tr)} valid={len(va)}")
    return tr, va


def _purge_embargo(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    purge_days: int,
    embargo_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    va = valid.copy()
    purge_days = int(purge_days or 0)
    embargo_days = int(embargo_days or 0)
    if purge_days > 0:
        dts = sorted(tr["date"].dropna().unique())
        keep = set(dts[:-purge_days]) if len(dts) > purge_days else set()
        tr = tr[tr["date"].isin(keep)].copy()
    if embargo_days > 0:
        dts = sorted(va["date"].dropna().unique())
        drop = set(dts[:embargo_days]) if len(dts) > embargo_days else set(dts)
        va = va[~va["date"].isin(drop)].copy()
    if tr.empty or va.empty:
        raise RuntimeError(f"purge/embargo empty: train={len(tr)} valid={len(va)}")
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def _pick_features(df: pd.DataFrame) -> List[str]:
    drop_cols = {"date", "ticker", "label"}
    drop_prefixes = ("future_return_", "future_excess_return_", "spy_future_return_")
    feats = []
    for c in df.select_dtypes(include=["number"]).columns:
        if c in drop_cols:
            continue
        if "future_" in c:
            continue
        if any(str(c).startswith(p) for p in drop_prefixes):
            continue
        feats.append(str(c))
    if not feats:
        raise RuntimeError("no numeric features found after filtering")
    return feats


def _make_groups(dates: pd.Series) -> np.ndarray:
    sizes = pd.Series(dates).groupby(dates, sort=True).size().values
    return sizes.astype(int)


def _to_relevance_bins(
    df: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    bins: int,
) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be >=2")
    out = np.empty(len(df), dtype=int)
    start = 0
    for _, g in df.groupby(date_col, sort=True):
        vals = pd.to_numeric(g[target_col], errors="coerce").astype(float).values
        n = len(vals)
        if np.nanstd(vals) == 0 or len(np.unique(vals[~np.isnan(vals)])) < bins:
            rk = pd.Series(vals).rank(method="average", na_option="keep")
            scaled = ((rk - 1) / max(1.0, (rk.max() - 1))) * (bins - 1)
            rel = np.floor(scaled.fillna(0).values).astype(int)
        else:
            try:
                rel = pd.qcut(pd.Series(vals), q=bins, labels=False, duplicates="drop")
                rel = rel.fillna(0).astype(int).values
            except Exception:
                rk = pd.Series(vals).rank(method="average", na_option="keep")
                scaled = ((rk - 1) / max(1.0, (rk.max() - 1))) * (bins - 1)
                rel = np.floor(scaled.fillna(0).values).astype(int)
        out[start : start + n] = rel
        start += n
    return out


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    aa = a.rank(method="average")
    bb = b.rank(method="average")
    return float(aa.corr(bb, method="pearson"))


def _evaluate_spread(
    panel: pd.DataFrame,
    *,
    date_col: str,
    ticker_col: str,
    score_col: str,
    target_col: str,
    top_n: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = panel[[date_col, ticker_col, score_col, target_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)
    rows = []
    prev_top: set[str] | None = None
    for dt, g in d.groupby(date_col, sort=True):
        gg = g.dropna(subset=[score_col, target_col])
        if len(gg) < max(10, top_n):
            continue
        ic = _spearman_corr(gg[score_col], gg[target_col])
        top = gg.nlargest(top_n, score_col)
        spread = float(top[target_col].mean() - gg[target_col].mean())
        top_set = set(top[ticker_col].astype(str).tolist())
        if prev_top is None:
            turnover = np.nan
        else:
            turnover = 1.0 - (len(top_set & prev_top) / float(top_n))
        prev_top = top_set
        rows.append({"date": dt.strftime("%Y-%m-%d"), "ic": float(ic), "spread": spread, "turnover": float(turnover) if not np.isnan(turnover) else np.nan})
    daily = pd.DataFrame(rows)
    if daily.empty:
        raise RuntimeError("no daily evaluation rows (check coverage/topn)")
    spread = pd.to_numeric(daily["spread"], errors="coerce")
    ic = pd.to_numeric(daily["ic"], errors="coerce")
    turnover = pd.to_numeric(daily["turnover"], errors="coerce")
    spread_mean = float(spread.mean())
    spread_std = float(spread.std(ddof=0))
    spread_tstat = float(spread_mean / (spread_std / np.sqrt(len(spread)))) if spread_std != 0 else float("nan")
    out = {
        "days": float(len(daily)),
        "ic_mean": float(ic.mean()),
        "spread_mean": spread_mean,
        "spread_tstat": spread_tstat,
        "turnover_mean": float(turnover.dropna().mean()) if turnover.notna().any() else float("nan"),
    }
    return daily, out


def _ndcg_at_k(rels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(rels) == 0:
        return float("nan")
    kk = int(min(k, len(rels)))
    if kk <= 0:
        return float("nan")
    order = np.argsort(-scores)[:kk]
    r = rels[order].astype(float)
    gains = np.power(2.0, r) - 1.0
    discounts = np.log2(np.arange(2, kk + 2))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(rels.astype(float))[::-1][:kk]
    gains_i = np.power(2.0, ideal) - 1.0
    idcg = float(np.sum(gains_i / discounts))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def _evaluate_ndcg(panel: pd.DataFrame, *, date_col: str, score_col: str, rel_col: str, k: int) -> Dict[str, float]:
    rows = []
    for dt, g in panel.groupby(date_col, sort=True):
        gg = g.dropna(subset=[score_col, rel_col])
        if len(gg) < max(10, k):
            continue
        ndcg = _ndcg_at_k(
            rels=gg[rel_col].astype(int).values,
            scores=pd.to_numeric(gg[score_col], errors="coerce").astype(float).values,
            k=k,
        )
        rows.append(float(ndcg))
    s = pd.Series(rows, dtype="float64")
    return {"ndcg_mean": float(s.mean()), "ndcg_std": float(s.std(ddof=0)), "ndcg_n_days": float(len(s))}


def main() -> int:
    ap = argparse.ArgumentParser(description="Minimal brute-force XGB ranker tuner (spread_tstat + ndcg@K).")
    ap.add_argument("--dataset", required=True, help="Parquet panel with date/ticker/features/target")
    ap.add_argument("--target-col", required=True, help="e.g. future_return_5d")
    ap.add_argument("--grid", required=True, help="YAML grid with xgboost params (lists)")
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--valid-start", default="2024-01-01")
    ap.add_argument("--valid-end", default="")
    ap.add_argument("--purge-days", type=int, default=0)
    ap.add_argument("--embargo-days", type=int, default=0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--ndcg-k", type=int, default=20)
    ap.add_argument("--early-stopping-rounds", type=int, default=0)
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--out", default="", help="Output CSV path (default: artifacts/logs/tune_ranker_<ts>.csv)")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    df = pd.read_parquet(ds_path)
    if "date" not in df.columns or "ticker" not in df.columns:
        raise RuntimeError("dataset must contain columns: date, ticker")
    tgt = str(args.target_col)
    if tgt not in df.columns:
        raise RuntimeError(f"dataset missing target col: {tgt}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)
    df[tgt] = pd.to_numeric(df[tgt], errors="coerce")
    df = df.dropna(subset=[tgt])

    tr, va = _time_split(df, train_start=args.train_start, valid_start=args.valid_start, valid_end=(args.valid_end or None))
    if int(args.purge_days) or int(args.embargo_days):
        tr, va = _purge_embargo(tr, va, purge_days=int(args.purge_days), embargo_days=int(args.embargo_days))

    feats = _pick_features(df)
    X_tr = tr[feats].replace([np.inf, -np.inf], np.nan)
    X_va = va[feats].replace([np.inf, -np.inf], np.nan)

    bins = int(args.bins)
    y_tr = _to_relevance_bins(tr, date_col="date", target_col=tgt, bins=bins)
    y_va = _to_relevance_bins(va, date_col="date", target_col=tgt, bins=bins)
    g_tr = _make_groups(tr["date"])
    g_va = _make_groups(va["date"])

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    dtr.set_group(g_tr)
    dva.set_group(g_va)

    rel_va = y_va  # for ndcg eval
    eval_panel = va[["date", "ticker", tgt]].copy()

    grid = _read_grid(args.grid)
    combos = list(_iter_grid(grid))
    if int(args.max_trials) > 0:
        combos = combos[: int(args.max_trials)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else (Path("artifacts") / "logs" / f"tune_ranker_{ts}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")

    meta = {
        "created_at": ts,
        "env": _env_fingerprint(),
        "args": vars(args),
    }
    meta["dataset"] = {
        "path": str(ds_path),
        "fingerprint": _dataset_fingerprint(df, target_col=tgt),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"meta={meta_path}")
    print(f"dataset={ds_path} rows={len(df)} train={len(tr)} valid={len(va)} feats={len(feats)} target={tgt}")
    print(f"trials={len(combos)} grid_keys={list(grid.keys())}")

    rows: List[Dict[str, Any]] = []
    best = None
    for i, ov in enumerate(combos, start=1):
        t0 = time.time()
        r: Dict[str, Any] = {"trial": int(i), "overrides": json.dumps(ov, sort_keys=True)}
        try:
            params = dict(ov)
            num_round = int(params.pop("num_boost_round", params.pop("n_estimators", 400)))
            # alias eta/learning_rate
            if "learning_rate" in params and "eta" not in params:
                params["eta"] = float(params.pop("learning_rate"))
            # alias lambda -> reg_lambda
            if "reg_lambda" in params and "lambda" not in params:
                params["lambda"] = float(params.pop("reg_lambda"))

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

            panel = eval_panel.copy()
            panel["score"] = score
            _, spread = _evaluate_spread(
                panel,
                date_col="date",
                ticker_col="ticker",
                score_col="score",
                target_col=tgt,
                top_n=int(args.topn),
            )

            nd = panel[["date", "ticker", "score"]].copy()
            nd["rel"] = rel_va
            ndcg = _evaluate_ndcg(nd, date_col="date", score_col="score", rel_col="rel", k=int(args.ndcg_k))

            r.update(spread)
            r.update(ndcg)
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"
        r["fit_seconds"] = float(time.time() - t0)
        rows.append(r)

        if "error" not in r and "spread_tstat" in r and "ndcg_mean" in r:
            key = (float(r["spread_tstat"]), float(r["ndcg_mean"]))
            if best is None or key > best[0]:
                best = (key, r)

        if i % 25 == 0 or i == 1 or i == len(combos):
            if best:
                print(f"[{i}/{len(combos)}] best_spread_t={best[1]['spread_tstat']:.3f} best_ndcg={best[1]['ndcg_mean']:.4f}")
            else:
                print(f"[{i}/{len(combos)}] no_success_yet")

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"output={out_path}")
    if best:
        print("best_overrides=", best[1]["overrides"])
        print(f"best_spread_tstat={best[1]['spread_tstat']:.3f} best_ndcg={best[1]['ndcg_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


