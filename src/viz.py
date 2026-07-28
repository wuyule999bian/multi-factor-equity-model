"""
Plotting functions. Every function takes data in, returns a matplotlib Axes
(or Figure for multi-panel plots) -- no function saves files or calls
plt.show() itself, so the notebook controls sizing/saving/display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = sns.color_palette("viridis", 10)


def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titleweight"] = "bold"


def plot_decile_returns(decile_df: pd.DataFrame, factor_name: str, ax: plt.Axes | None = None) -> plt.Axes:
    """Bar chart of mean forward return by decile for one factor."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    colors = sns.color_palette("RdYlGn", n_colors=int(decile_df["decile"].max()))
    ax.bar(decile_df["decile"], decile_df["mean_return"] * 100, color=colors)
    ax.set_xlabel("Decile (1 = lowest score, 10 = highest score)")
    ax.set_ylabel("Mean forward return (%)")
    ax.set_title(f"{factor_name}: return by decile")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(decile_df["decile"])
    return ax


def plot_cumulative_long_short(spreads: dict[str, pd.Series], ax: plt.Axes | None = None) -> plt.Axes:
    """Cumulative long-short return lines for one or more factors, overlaid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for (name, series), color in zip(spreads.items(), PALETTE):
        cum = (1 + series.fillna(0)).cumprod() - 1
        ax.plot(cum.index, cum.values * 100, marker="o", label=name, color=color)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Cumulative long-short return (%)")
    ax.set_xlabel("Rebalance period")
    ax.set_title("Cumulative long-short factor returns")
    ax.legend()
    return ax


def plot_fm_coefficients(pass1_coefs: pd.DataFrame, factor_cols: list[str], pass2: pd.DataFrame) -> plt.Figure:
    """One subplot per factor: per-period Fama-MacBeth coefficient, with the
    full-sample mean premium and t-stat (from pass 2) annotated.
    """
    n = len(factor_cols)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col, color in zip(axes, factor_cols, PALETTE):
        ax.plot(pass1_coefs.index, pass1_coefs[col] * 100, marker="o", color=color)
        ax.axhline(0, color="black", linewidth=0.8)
        mean_prem = pass2.loc[col, "mean_premium"] * 100
        t_stat = pass2.loc[col, "t_stat"]
        ax.axhline(mean_prem, color=color, linestyle="--", linewidth=1)
        ax.set_ylabel("Coef. (%)")
        ax.set_title(f"{col}: mean premium = {mean_prem:.2f}%,  t-stat = {t_stat:.2f}", fontsize=10)
    axes[-1].set_xlabel("Rebalance period")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(panel: pd.DataFrame, factor_cols: list[str], ax: plt.Axes | None = None) -> plt.Axes:
    """Heatmap of pairwise correlations between factor z-scores (pooled across periods)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    corr = panel[factor_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Factor correlation matrix (z-scores, pooled)")
    return ax
