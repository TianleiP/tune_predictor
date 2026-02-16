from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestSummary:
    start_date: str
    end_date: str
    days: int
    top_n: int
    hold_days: int
    cost_bps: float
    avg_turnover: float
    cum_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def build_ret_matrix(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "adj_close",
) -> tuple[np.ndarray, List[pd.Timestamp], List[str], Dict[str, int]]:
    """
    Build a (n_dates, n_tickers) matrix of close-to-close returns.
    """
    d = df[[date_col, ticker_col, price_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)
    d["ret_1d"] = d.groupby(ticker_col)[price_col].pct_change()

    dates = sorted(d[date_col].unique())
    tickers = sorted(d[ticker_col].astype(str).unique())
    t2i = {t: i for i, t in enumerate(tickers)}

    mat = (
        d.pivot(index=date_col, columns=ticker_col, values="ret_1d")
        .reindex(index=dates, columns=tickers)
        .astype("float32")
        .to_numpy(copy=True)
    )
    return mat, dates, tickers, t2i


def overlap_topn_backtest_from_scores(
    *,
    score_mat: np.ndarray,
    ret_mat: np.ndarray,
    dates: List[pd.Timestamp],
    top_n: int,
    hold_days: int,
    cost_bps: float,
) -> tuple[pd.DataFrame, BacktestSummary]:
    """
    score_mat: (n_dates, n_tickers) score for each ticker per date (NaN allowed)
    ret_mat:   (n_dates, n_tickers) 1d returns for each ticker per date (NaN allowed)
    """
    if score_mat.shape != ret_mat.shape:
        raise ValueError("score_mat and ret_mat must have same shape")
    n_dates, n_tickers = score_mat.shape
    if n_dates != len(dates):
        raise ValueError("dates length mismatch")
    if n_dates < (hold_days + 5):
        raise RuntimeError("Not enough dates for backtest")

    w_pos = 1.0 / float(hold_days * top_n)
    cost_rate = float(cost_bps) / 10000.0

    equity = 1.0
    curve = []

    # Each cohort: numpy array of ticker indices
    cohorts: List[np.ndarray] = []
    prev_top: np.ndarray | None = None
    turnovers = []

    for i in range(n_dates):
        pnl = 0.0

        # PnL from active positions for today's return (ret_mat at i)
        if cohorts:
            active_idx = np.concatenate(cohorts) if len(cohorts) > 1 else cohorts[0]
            rets = ret_mat[i, active_idx]
            if rets.size:
                pnl += w_pos * float(np.nansum(rets))

        # Exit oldest cohort after earning today's return
        if len(cohorts) >= hold_days:
            old = cohorts.pop(0)
            pnl -= float(len(old)) * w_pos * cost_rate

        # Enter today's cohort
        scores = score_mat[i]
        if np.count_nonzero(~np.isnan(scores)) >= top_n:
            # select top_n indices
            # argpartition is O(n) and fast
            idx = np.argpartition(-np.nan_to_num(scores, nan=-np.inf), top_n - 1)[:top_n]
            # For turnover, we want consistent membership regardless of tie ordering:
            idx = np.sort(idx)

            pnl -= float(len(idx)) * w_pos * cost_rate
            cohorts.append(idx)

            if prev_top is None:
                turnover = np.nan
            else:
                inter = np.intersect1d(prev_top, idx, assume_unique=False).size
                turnover = 1.0 - (inter / float(top_n))
            prev_top = idx
            turnovers.append(turnover)
        else:
            turnovers.append(np.nan)

        equity *= (1.0 + pnl)
        curve.append({"date": dates[i].strftime("%Y-%m-%d"), "daily_return": pnl, "equity": equity})

    out = pd.DataFrame(curve)
    rets = out["daily_return"].astype(float).to_numpy()
    days = int(len(out))
    ann_factor = 252.0
    ann_vol = float(np.nanstd(rets) * np.sqrt(ann_factor))
    sharpe = float((np.nanmean(rets) * ann_factor) / (np.nanstd(rets) * np.sqrt(ann_factor))) if np.nanstd(rets) != 0 else float("nan")
    cum_return = float(out["equity"].iloc[-1] - 1.0)
    ann_return = float((out["equity"].iloc[-1]) ** (ann_factor / days) - 1.0) if days > 0 else float("nan")
    mdd = _max_drawdown(out["equity"].astype(float).to_numpy())
    avg_turnover = float(np.nanmean(np.asarray(turnovers, dtype="float64"))) if len(turnovers) else float("nan")

    summary = BacktestSummary(
        start_date=str(out["date"].iloc[0]),
        end_date=str(out["date"].iloc[-1]),
        days=days,
        top_n=int(top_n),
        hold_days=int(hold_days),
        cost_bps=float(cost_bps),
        avg_turnover=avg_turnover,
        cum_return=cum_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
    )
    return out, summary


