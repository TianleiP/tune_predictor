from __future__ import annotations

"""
Official backtest implementation copied from the main project (`src/backtest/overlap_topn.py`).

Goal: make the minimal GPU tuner compute portfolio Sharpe/cum_return with identical mechanics,
so tuning results are apples-to-apples with `scripts/backtest_compare.py`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def run_overlap_equal_weight_backtest(
    *,
    price_panel: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "adj_close",
    hold_days: int,
    cost_bps: float = 0.0,
    slippage_k: float = 0.0,
    slippage_cap_bps: float = 50.0,
    liquidity_min_log_dollar_volume: float | None = None,
    amihud_col: str = "amihud_illiq",
    log_dollar_volume_col: str = "log_dollar_volume",
) -> tuple[pd.DataFrame, BacktestSummary]:
    cols = [date_col, ticker_col, price_col]
    if amihud_col in price_panel.columns:
        cols.append(amihud_col)
    if log_dollar_volume_col in price_panel.columns:
        cols.append(log_dollar_volume_col)
    p = price_panel[cols].copy()
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce")
    p = p.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)
    p["ret_1d"] = p.groupby(ticker_col)[price_col].pct_change()

    dates = sorted(p[date_col].unique())
    if len(dates) < (hold_days + 5):
        raise RuntimeError("Not enough dates for baseline backtest.")

    cost_rate = float(cost_bps) / 10000.0
    cap_rate = float(slippage_cap_bps) / 10000.0
    equity = 1.0
    active_cohorts = 0
    rows = []
    for i, dt in enumerate(dates):
        pnl = 0.0
        if i > 0:
            day = p[p[date_col] == dt]
            if liquidity_min_log_dollar_volume is not None and log_dollar_volume_col in day.columns:
                day = day[pd.to_numeric(day[log_dollar_volume_col], errors="coerce") >= float(liquidity_min_log_dollar_volume)]
            mean_ret = float(pd.to_numeric(day["ret_1d"], errors="coerce").dropna().mean()) if len(day) else 0.0
            pnl += (active_cohorts / float(hold_days)) * mean_ret

        if i >= hold_days:
            pnl -= (1.0 / float(hold_days)) * cost_rate
            if slippage_k and amihud_col in p.columns:
                day = p[p[date_col] == dt]
                if liquidity_min_log_dollar_volume is not None and log_dollar_volume_col in day.columns:
                    day = day[pd.to_numeric(day[log_dollar_volume_col], errors="coerce") >= float(liquidity_min_log_dollar_volume)]
                a = float(pd.to_numeric(day[amihud_col], errors="coerce").dropna().mean()) if len(day) else 0.0
                pnl -= (1.0 / float(hold_days)) * min(cap_rate, float(slippage_k) * a)
            active_cohorts -= 1

        pnl -= (1.0 / float(hold_days)) * cost_rate
        if slippage_k and amihud_col in p.columns:
            day = p[p[date_col] == dt]
            if liquidity_min_log_dollar_volume is not None and log_dollar_volume_col in day.columns:
                day = day[pd.to_numeric(day[log_dollar_volume_col], errors="coerce") >= float(liquidity_min_log_dollar_volume)]
            a = float(pd.to_numeric(day[amihud_col], errors="coerce").dropna().mean()) if len(day) else 0.0
            pnl -= (1.0 / float(hold_days)) * min(cap_rate, float(slippage_k) * a)
        active_cohorts += 1

        equity *= (1.0 + pnl)
        rows.append({"date": dt.strftime("%Y-%m-%d"), "daily_return": pnl, "equity": equity, "n_active_cohorts": active_cohorts, "turnover": 0.0})

    curve = pd.DataFrame(rows)
    rets = curve["daily_return"].astype(float)
    days = int(len(curve))
    cum_return = float(curve["equity"].iloc[-1] - 1.0)
    ann_factor = 252.0
    ann_return = float((curve["equity"].iloc[-1]) ** (ann_factor / days) - 1.0) if days > 0 else float("nan")
    ann_vol = float(rets.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = float((rets.mean() * ann_factor) / (rets.std(ddof=0) * np.sqrt(ann_factor))) if rets.std(ddof=0) != 0 else float("nan")
    mdd = _max_drawdown(curve["equity"])
    summary = BacktestSummary(
        start_date=str(curve["date"].iloc[0]),
        end_date=str(curve["date"].iloc[-1]),
        days=days,
        top_n=-1,
        hold_days=int(hold_days),
        cost_bps=float(cost_bps),
        avg_turnover=0.0,
        cum_return=cum_return,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
    )
    return curve, summary


def run_overlap_topn_backtest(
    *,
    scored_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    score_col: str = "score",
    price_col: str = "adj_close",
    top_n: int,
    hold_days: int,
    cost_bps: float = 0.0,
    slippage_k: float = 0.0,
    slippage_cap_bps: float = 50.0,
    liquidity_min_log_dollar_volume: float | None = None,
    amihud_col: str = "amihud_illiq",
    log_dollar_volume_col: str = "log_dollar_volume",
) -> tuple[pd.DataFrame, BacktestSummary]:
    s_cols = [date_col, ticker_col, score_col]
    if amihud_col in scored_panel.columns:
        s_cols.append(amihud_col)
    if log_dollar_volume_col in scored_panel.columns:
        s_cols.append(log_dollar_volume_col)
    s = scored_panel[s_cols].copy()
    s[date_col] = pd.to_datetime(s[date_col], errors="coerce")
    s = s.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)

    p = price_panel[[date_col, ticker_col, price_col]].copy()
    p[date_col] = pd.to_datetime(p[date_col], errors="coerce")
    p = p.dropna(subset=[date_col]).sort_values([date_col, ticker_col]).reset_index(drop=True)
    p["ret_1d"] = p.groupby(ticker_col)[price_col].pct_change()

    dates = sorted(set(s[date_col].unique()) & set(p[date_col].unique()))
    if len(dates) < (hold_days + 5):
        raise RuntimeError("Not enough overlapping dates to run backtest.")

    top_by_date: Dict[pd.Timestamp, List[str]] = {}
    amihud_map: Dict[tuple[pd.Timestamp, str], float] = {}
    if amihud_col in s.columns:
        ss = s[[date_col, ticker_col, amihud_col]].copy()
        ss[amihud_col] = pd.to_numeric(ss[amihud_col], errors="coerce")
        amihud_map = {
            (r[date_col], str(r[ticker_col])): float(r[amihud_col])
            for r in ss.dropna(subset=[amihud_col]).to_dict(orient="records")
        }
    for dt, g in s.groupby(date_col, sort=True):
        gg = g.dropna(subset=[score_col])
        if liquidity_min_log_dollar_volume is not None and log_dollar_volume_col in gg.columns:
            gg = gg[pd.to_numeric(gg[log_dollar_volume_col], errors="coerce") >= float(liquidity_min_log_dollar_volume)]
        if len(gg) < top_n:
            continue
        top = gg.nlargest(top_n, score_col)[ticker_col].astype(str).tolist()
        top_by_date[dt] = top

    w_pos = 1.0 / float(hold_days * top_n)
    cost_rate = float(cost_bps) / 10000.0
    cap_rate = float(slippage_cap_bps) / 10000.0

    active: List[Tuple[int, str]] = []
    prev_top: Optional[set[str]] = None

    rows = []
    equity = 1.0
    for i, dt in enumerate(dates):
        pnl = 0.0
        if i > 0:
            day_rets = p[p[date_col] == dt].set_index(ticker_col)["ret_1d"]
            for entry_i, t in active:
                r = day_rets.get(t)
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                pnl += w_pos * float(r)

        exits = [(ei, t) for (ei, t) in active if (i - ei) >= hold_days]
        if exits:
            traded_notional = len(exits) * w_pos
            pnl -= traded_notional * cost_rate
            if slippage_k and amihud_map:
                slip = 0.0
                for _, t in exits:
                    a = amihud_map.get((dt, t))
                    if a is None:
                        continue
                    slip += w_pos * min(cap_rate, float(slippage_k) * float(a))
                pnl -= slip
        active = [(ei, t) for (ei, t) in active if (i - ei) < hold_days]

        todays = top_by_date.get(dt, [])
        if len(todays) == top_n:
            traded_notional = top_n * w_pos
            pnl -= traded_notional * cost_rate
            if slippage_k and amihud_map:
                slip = 0.0
                for t in todays:
                    a = amihud_map.get((dt, t))
                    if a is None:
                        continue
                    slip += w_pos * min(cap_rate, float(slippage_k) * float(a))
                pnl -= slip
            for t in todays:
                active.append((i, t))

        if len(todays) == top_n:
            top_set = set(todays)
            if prev_top is None:
                turnover = np.nan
            else:
                turnover = 1.0 - (len(top_set & prev_top) / float(top_n))
            prev_top = top_set
        else:
            turnover = np.nan

        equity *= (1.0 + pnl)
        rows.append({"date": dt.strftime("%Y-%m-%d"), "daily_return": pnl, "equity": equity, "n_active_positions": len(active), "turnover": float(turnover) if not np.isnan(turnover) else np.nan})

    curve = pd.DataFrame(rows)
    curve["daily_return"] = pd.to_numeric(curve["daily_return"], errors="coerce").fillna(0.0)
    curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")

    rets = curve["daily_return"]
    days = int(len(curve))
    cum_return = float(curve["equity"].iloc[-1] - 1.0)
    ann_factor = 252.0
    ann_return = float((curve["equity"].iloc[-1]) ** (ann_factor / days) - 1.0) if days > 0 else float("nan")
    ann_vol = float(rets.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = float((rets.mean() * ann_factor) / (rets.std(ddof=0) * np.sqrt(ann_factor))) if rets.std(ddof=0) != 0 else float("nan")
    mdd = _max_drawdown(curve["equity"])
    avg_turnover = float(pd.to_numeric(curve["turnover"], errors="coerce").dropna().mean()) if curve["turnover"].notna().any() else float("nan")

    summary = BacktestSummary(
        start_date=str(curve["date"].iloc[0]),
        end_date=str(curve["date"].iloc[-1]),
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
    return curve, summary


