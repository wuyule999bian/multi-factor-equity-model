"""
Factor construction: value, momentum, size, quality, and cross-sectional
z-scoring.

Point-in-time hygiene (the two things that quietly wreck a backtest):

1. No lookahead. Every fundamental row carries an `available_date`
   (fiscal period end + a conservative filing lag, see data.py). Rebalance
   dates are fixed at July 1 of the year *after* the fiscal year so that
   every ticker's fundamentals -- regardless of individual fiscal calendars
   -- are guaranteed to already be public. Momentum and forward returns are
   read directly off the price series at those same rebalance dates, so no
   factor or label ever depends on information from the future.

2. No survivorship bias beyond what's already inherent in the static
   universe (see data.load_universe docstring). Within that universe, we do
   not drop tickers for having a bad quarter, going on to underperform, etc.
   -- every ticker that has fundamentals data for a given period is scored
   in that period's cross-section, good or bad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REBALANCE_MONTH = 7   # July 1 of (fiscal_year + 1): safely after FILING_LAG_DAYS
REBALANCE_DAY = 1


# --------------------------------------------------------------------------- #
# Cross-sectional z-scoring
# --------------------------------------------------------------------------- #
def winsorize(s: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Clip a series to its [lo, hi] quantiles to blunt outlier influence."""
    lo, hi = s.quantile(limits[0]), s.quantile(limits[1])
    return s.clip(lower=lo, upper=hi)


def cross_sectional_zscore(
    df: pd.DataFrame, col: str, period_col: str = "period", winsorize_first: bool = True
) -> pd.Series:
    """Z-score `col` within each `period_col` group (not across periods).

    This is what makes the factors comparable period to period: a raw
    earnings yield of 5% means something different in 2021 than in 2023, but
    "1.2 standard deviations above that period's cross-sectional mean" is
    stable. Winsorizing before scoring keeps a single extreme ticker from
    dominating the mean/std of an entire period.
    """
    def _z(group: pd.Series) -> pd.Series:
        g = winsorize(group.dropna()) if winsorize_first else group.dropna()
        if g.std(ddof=0) == 0 or len(g) < 3:
            return pd.Series(np.nan, index=group.index)
        z = (group - g.mean()) / g.std(ddof=0)
        return z

    return df.groupby(period_col)[col].transform(_z)


# --------------------------------------------------------------------------- #
# Rebalance calendar
# --------------------------------------------------------------------------- #
def assign_rebalance_period(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Attach a `period` (rebalance date) to each fundamentals row.

    period = July 1 of (fiscal_year + 1). Using a fixed calendar date rather
    than each company's own available_date keeps every ticker on the same
    cross-sectional grid (required for decile sorts and Fama-MacBeth), at
    the cost of being less timely for non-December fiscal-year-end filers --
    a deliberate, documented trade-off, not an oversight.
    """
    out = fundamentals.copy()
    out["period"] = pd.to_datetime(
        {"year": out["fiscal_year"] + 1, "month": REBALANCE_MONTH, "day": REBALANCE_DAY}
    )
    # Safety check: the fixed rebalance date must fall after the row's
    # actual available_date, otherwise we'd be trading on unpublished data.
    bad = out["period"] < out["available_date"]
    if bad.any():
        out = out.loc[~bad].copy()
    return out


# --------------------------------------------------------------------------- #
# Price-derived factors: momentum + forward returns
# --------------------------------------------------------------------------- #
def _price_asof(prices: pd.DataFrame, date: pd.Timestamp, max_forward_slack_days: int = 7) -> pd.Series:
    """Last available price on or before `date`, per ticker.

    Returns NaN (rather than silently returning the most recent price) when
    `date` is more than `max_forward_slack_days` beyond the price panel's
    last date. Without this guard, asking for a price at a *future*
    rebalance date -- which happens for the most recent period, whose
    "next" rebalance date hasn't occurred yet -- would resolve to today's
    price and get mislabeled as a full-period forward return instead of the
    lookahead-tainted partial-period return it actually is.
    """
    last_available = prices.index[-1]
    if date > last_available + pd.Timedelta(days=max_forward_slack_days):
        return pd.Series(np.nan, index=prices.columns)
    idx = prices.index.searchsorted(date, side="right") - 1
    if idx < 0:
        return pd.Series(np.nan, index=prices.columns)
    return prices.iloc[idx]


def compute_momentum(
    prices: pd.DataFrame, periods: list[pd.Timestamp], skip_months: int = 1, lookback_months: int = 12
) -> pd.DataFrame:
    """12-1 month momentum: return from (t - 12mo) to (t - 1mo), per ticker per period.

    Skipping the most recent month avoids the well-documented short-term
    reversal effect contaminating the momentum signal.
    """
    rows = []
    for period in periods:
        end = period - pd.DateOffset(months=skip_months)
        start = period - pd.DateOffset(months=lookback_months)
        p_end = _price_asof(prices, end)
        p_start = _price_asof(prices, start)
        mom = (p_end / p_start) - 1.0
        rows.append(pd.DataFrame({"ticker": mom.index, "period": period, "momentum": mom.values}))
    return pd.concat(rows, ignore_index=True)


def compute_forward_returns(prices: pd.DataFrame, periods: list[pd.Timestamp]) -> pd.DataFrame:
    """Realized return from each period to the *next* period (the label for backtests/FM)."""
    periods = sorted(periods)
    rows = []
    for t0, t1 in zip(periods[:-1], periods[1:]):
        p0 = _price_asof(prices, t0)
        p1 = _price_asof(prices, t1)
        fwd = (p1 / p0) - 1.0
        rows.append(pd.DataFrame({"ticker": fwd.index, "period": t0, "fwd_return": fwd.values}))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Fundamental factors
# --------------------------------------------------------------------------- #
def compute_quality_stability(fundamentals: pd.DataFrame) -> pd.Series:
    """Expanding-window (point-in-time) std of net margin, per ticker, as of each row.

    Uses only fiscal periods up to and including the current row for each
    ticker -- an expanding window, not a centered one -- so no future margin
    data leaks into a past quality score.
    """
    fundamentals = fundamentals.sort_values(["ticker", "fiscal_date"])
    stability = fundamentals.groupby("ticker")["netProfitMargin"].apply(
        lambda s: s.expanding(min_periods=2).std()
    )
    return stability.reset_index(level=0, drop=True).reindex(fundamentals.index)


MIN_TICKERS_PER_PERIOD = 20  # below this, a cross-section is too thin to decile-sort meaningfully


def build_factor_panel(fundamentals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Assemble the full period x ticker factor panel used downstream.

    Returns one row per (ticker, period) with raw + z-scored value, size,
    quality, momentum, a composite z-score, and the realized forward return.

    Periods with fewer than `MIN_TICKERS_PER_PERIOD` tickers are dropped
    entirely -- these show up at the edges of the sample (e.g. a handful of
    early fiscal-year filers reporting before the rest of their peers) and
    are too thin a cross-section for a decile sort to mean anything.
    """
    fnd = assign_rebalance_period(fundamentals)
    fnd["margin_stability_raw"] = -compute_quality_stability(fnd)  # negate: lower vol = higher quality
    fnd["book_to_market"] = np.where(
        fnd["priceToBookRatio"] > 0, 1.0 / fnd["priceToBookRatio"], np.nan
    )
    fnd["size_raw"] = -np.log(fnd["marketCap"].where(fnd["marketCap"] > 0))

    periods = sorted(fnd["period"].unique())
    mom = compute_momentum(prices, periods)
    fwd = compute_forward_returns(prices, periods)

    panel = fnd.merge(mom, on=["ticker", "period"], how="left")
    panel = panel.merge(fwd, on=["ticker", "period"], how="left")

    # --- cross-sectional z-scores, one factor at a time ---
    panel["z_book_to_market"] = cross_sectional_zscore(panel, "book_to_market")
    panel["z_earnings_yield"] = cross_sectional_zscore(panel, "earningsYield")
    panel["z_value"] = panel[["z_book_to_market", "z_earnings_yield"]].mean(axis=1)

    panel["z_size"] = cross_sectional_zscore(panel, "size_raw")

    panel["z_roe"] = cross_sectional_zscore(panel, "returnOnEquity")
    panel["z_margin_stability"] = cross_sectional_zscore(panel, "margin_stability_raw")
    panel["z_quality"] = panel[["z_roe", "z_margin_stability"]].mean(axis=1)

    panel["z_momentum"] = cross_sectional_zscore(panel, "momentum")

    panel["z_composite"] = panel[["z_value", "z_size", "z_quality", "z_momentum"]].mean(axis=1)

    keep = [
        "ticker", "sector", "period", "fiscal_date", "available_date",
        "marketCap", "book_to_market", "earningsYield", "returnOnEquity",
        "margin_stability_raw", "momentum",
        "z_value", "z_size", "z_quality", "z_momentum", "z_composite",
        "fwd_return",
    ]
    panel = panel[keep].sort_values(["period", "ticker"]).reset_index(drop=True)

    counts = panel.groupby("period")["ticker"].transform("count")
    dropped_periods = sorted(panel.loc[counts < MIN_TICKERS_PER_PERIOD, "period"].dt.date.unique())
    if dropped_periods:
        print(f"Dropping thin cross-sections (< {MIN_TICKERS_PER_PERIOD} tickers): {dropped_periods}")
    return panel.loc[counts >= MIN_TICKERS_PER_PERIOD].reset_index(drop=True)
