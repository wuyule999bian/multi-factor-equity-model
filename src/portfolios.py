"""
Decile sorts, long-short spread portfolios, the composite tilt portfolio,
and simple backtest performance statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------------------- #
# Decile sorts
# --------------------------------------------------------------------------- #
def decile_sort(panel: pd.DataFrame, factor_col: str, period_col: str = "period", n: int = 10) -> pd.Series:
    """Assign each (ticker, period) row a decile (1 = lowest factor score, n = highest).

    Sorted independently within each period, which is the whole point of a
    cross-sectional factor: a stock is "cheap" or "expensive" relative to
    its peers *at that point in time*, not relative to its own history.
    """
    def _decile(group: pd.Series) -> pd.Series:
        valid = group.dropna()
        if len(valid) < n:
            return pd.Series(np.nan, index=group.index)
        ranks = valid.rank(method="first")
        bins = pd.qcut(ranks, n, labels=False) + 1
        return bins.reindex(group.index)

    return panel.groupby(period_col)[factor_col].transform(_decile)


def decile_returns(
    panel: pd.DataFrame, factor_col: str, return_col: str = "fwd_return",
    period_col: str = "period", n: int = 10,
) -> pd.DataFrame:
    """Mean forward return by decile, averaged across all periods."""
    deciles = decile_sort(panel, factor_col, period_col, n)
    tmp = panel.assign(decile=deciles)
    return (
        tmp.dropna(subset=["decile", return_col])
        .groupby("decile")[return_col]
        .agg(mean_return="mean", std_return="std", n_obs="count")
        .reset_index()
    )


def long_short_spread(
    panel: pd.DataFrame, factor_col: str, return_col: str = "fwd_return",
    period_col: str = "period", n: int = 10,
) -> pd.Series:
    """Per-period top-decile-minus-bottom-decile return spread (equal-weighted)."""
    deciles = decile_sort(panel, factor_col, period_col, n)
    tmp = panel.assign(decile=deciles).dropna(subset=["decile", return_col])

    def _spread(group: pd.DataFrame) -> float:
        top = group.loc[group["decile"] == n, return_col].mean()
        bottom = group.loc[group["decile"] == 1, return_col].mean()
        return top - bottom

    return tmp.groupby(period_col)[["decile", return_col]].apply(_spread).rename(f"{factor_col}_long_short")


# --------------------------------------------------------------------------- #
# Composite portfolio
# --------------------------------------------------------------------------- #
def composite_long_short_returns(
    panel: pd.DataFrame, score_col: str = "z_composite", return_col: str = "fwd_return",
    period_col: str = "period", n: int = 10,
) -> pd.Series:
    """Equal-weighted long top decile / short bottom decile on the composite score."""
    return long_short_spread(panel, score_col, return_col, period_col, n).rename("composite_long_short")


def decompose_returns(
    portfolio_returns: pd.Series, factor_spread_returns: pd.DataFrame
) -> tuple[pd.Series, float]:
    """Regress the composite portfolio's period returns on the individual
    single-factor long-short spreads to recover its factor loadings (beta to
    value, size, quality, momentum) and R^2 -- i.e. how much of the
    composite's return is explained by the four factors vs. left as alpha.
    Warns (rather than silently reporting a misleading R^2) when there are
    fewer observations than parameters -- with this project's ~3-4 usable
    annual rebalance periods (a free-tier data constraint, see README), a
    regression with an intercept + 4 factor loadings is easily
    underdetermined, in which case R^2 will trivially hit 1.0 regardless of
    whether the factors actually explain anything.
    """
    aligned = pd.concat([portfolio_returns, factor_spread_returns], axis=1).dropna()
    y = aligned.iloc[:, 0]
    X = sm.add_constant(aligned.iloc[:, 1:])
    if len(aligned) <= X.shape[1]:
        print(
            f"    WARNING: only {len(aligned)} periods for {X.shape[1]} regression "
            "parameters -- this decomposition is underdetermined/overfit, not a "
            "reliable estimate. Treat the loadings and R^2 below as illustrative "
            "of the method, not a statistically meaningful result."
        )
    model = sm.OLS(y, X).fit()
    return model.params, model.rsquared


# --------------------------------------------------------------------------- #
# Performance stats
# --------------------------------------------------------------------------- #
def cumulative_returns(return_series: pd.Series) -> pd.Series:
    return (1.0 + return_series.fillna(0)).cumprod() - 1.0


def performance_stats(return_series: pd.Series, periods_per_year: float = 1.0) -> pd.Series:
    """Basic annualized performance stats for a period return series.

    `periods_per_year` should reflect the actual rebalance frequency (1.0
    for the annual rebalancing used in this project's live data pull).

    Note: a long-short *spread* (top decile return minus bottom decile
    return) is not bounded below at -100% the way a single long-only
    position is -- if the bottom decile rallies hard enough, the spread in
    one period can be more negative than -100% (this actually happens in
    this project's real momentum spread: prior "losers" as of one rebalance
    date went on to massively outperform, a textbook momentum crash). That
    makes (1+r) go negative, which breaks naive geometric compounding to a
    fractional power. We floor compounded wealth at a small positive
    epsilon so `annualized_return` stays a real number instead of NaN --
    it will read as a large negative annualized return in that case, which
    is directionally honest even if the exact magnitude is an artifact of
    compounding an unbounded spread.
    """
    r = return_series.dropna()
    if r.empty:
        return pd.Series(dtype=float)

    wealth = max((1 + r).prod(), 1e-6)
    ann_return = wealth ** (periods_per_year / len(r)) - 1
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = cumulative_returns(r)
    running_max = (1 + cum).cummax()
    drawdown = (1 + cum) / running_max - 1
    max_dd = drawdown.min()

    return pd.Series(
        {
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": (r > 0).mean(),
            "n_periods": len(r),
        }
    )
