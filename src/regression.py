"""
Fama-MacBeth two-pass regression, plus a sanity check against Kenneth
French's public factor library.

Pass 1: one cross-sectional OLS of forward return on factor z-scores per
rebalance period -> a time series of coefficients (one "risk premium"
estimate per factor per period).

Pass 2: average those per-period coefficients over time; the t-stat on the
mean tells you whether a factor's premium is distinguishable from zero given
how much it varies period to period -- the classic Fama-MacBeth (1973)
construction.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

FF_5FACTOR_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
FF_MOMENTUM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)

# Our factor z-scores map onto French's benchmark factors like so:
FACTOR_TO_FF = {
    "z_value": "HML",       # High-minus-low book-to-market
    "z_size": "SMB",        # Small-minus-big
    "z_quality": "RMW",     # Robust-minus-weak profitability
    "z_momentum": "Mom",    # Prior-return momentum
}


# --------------------------------------------------------------------------- #
# Pass 1: per-period cross-sectional regressions
# --------------------------------------------------------------------------- #
def fama_macbeth_pass1(
    panel: pd.DataFrame,
    factor_cols: list[str],
    return_col: str = "fwd_return",
    period_col: str = "period",
) -> pd.DataFrame:
    """Run one cross-sectional OLS per period: fwd_return ~ factor z-scores.

    Returns a period x (const, factor_cols..., n_obs, r_squared) DataFrame.
    """
    rows = []
    for period, grp in panel.groupby(period_col):
        sub = grp.dropna(subset=[return_col, *factor_cols])
        if len(sub) < len(factor_cols) + 5:  # need enough obs for a stable fit
            continue
        X = sm.add_constant(sub[factor_cols])
        y = sub[return_col]
        model = sm.OLS(y, X).fit()
        row = {period_col: period, "n_obs": int(model.nobs), "r_squared": model.rsquared}
        row.update(model.params.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).set_index(period_col).sort_index()


# --------------------------------------------------------------------------- #
# Pass 2: average premia + t-stats
# --------------------------------------------------------------------------- #
def fama_macbeth_pass2(pass1_coefs: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Average each period's coefficient over time; t-stat = mean / (std / sqrt(T)).

    This is the standard Fama-MacBeth t-stat: it treats each period's
    coefficient as one draw and asks whether the *time series* of estimates
    is reliably different from zero, which automatically accounts for
    period-to-period variation in the premium (unlike pooling all
    observations into one panel regression and ignoring cross-sectional
    correlation within a period).
    """
    cols = ["const", *factor_cols]
    out = []
    T = len(pass1_coefs)
    for c in cols:
        vals = pass1_coefs[c].dropna()
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        t_stat = mean / se if se and se > 0 else np.nan
        out.append({"factor": c, "mean_premium": mean, "std_error": se, "t_stat": t_stat, "n_periods": len(vals)})
    result = pd.DataFrame(out).set_index("factor")
    result.attrs["n_periods_total"] = T
    return result


# --------------------------------------------------------------------------- #
# Kenneth French data library benchmark
# --------------------------------------------------------------------------- #
def _parse_ff_annual_section(text: str, value_cols: list[str]) -> pd.DataFrame:
    """Pull the 'Annual Factors' block out of a French data-library CSV."""
    marker = "Annual Factors"
    idx = text.find(marker)
    if idx == -1:
        raise ValueError("Could not find 'Annual Factors' section in French data file.")
    tail = text[idx:]
    lines = tail.splitlines()

    header_idx = next(i for i, ln in enumerate(lines) if ln.strip().startswith(","))
    data_lines = []
    for ln in lines[header_idx + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < len(value_cols) + 1 or not parts[0].isdigit():
            if data_lines:
                break  # end of the annual block
            continue
        data_lines.append(parts)

    df = pd.DataFrame(data_lines, columns=["year", *value_cols]).astype(float)
    df["year"] = df["year"].astype(int)
    for c in value_cols:
        df[c] = df[c] / 100.0  # French's files are in percent
    return df.set_index("year")


def fetch_french_factors(data_dir: str | Path, force_refresh: bool = False) -> pd.DataFrame:
    """Download (or load cached) Kenneth French annual benchmark factors.

    Returns an annual DataFrame indexed by year with Mkt-RF, SMB, HML, RMW,
    CMA, RF, Mom -- all as decimal returns (not percent).
    """
    cache_path = Path(data_dir) / "raw" / "ff_factors_annual.csv"
    if cache_path.exists() and not force_refresh:
        return pd.read_csv(cache_path, index_col="year")

    r5 = requests.get(FF_5FACTOR_URL, timeout=20)
    r5.raise_for_status()
    z5 = zipfile.ZipFile(io.BytesIO(r5.content))
    text5 = z5.read(z5.namelist()[0]).decode("utf-8", errors="ignore")
    five = _parse_ff_annual_section(text5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])

    rmom = requests.get(FF_MOMENTUM_URL, timeout=20)
    rmom.raise_for_status()
    zmom = zipfile.ZipFile(io.BytesIO(rmom.content))
    textmom = zmom.read(zmom.namelist()[0]).decode("utf-8", errors="ignore")
    mom = _parse_ff_annual_section(textmom, ["Mom"])

    merged = five.join(mom, how="left")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path)
    return merged


def compare_to_french(
    fm_pass2: pd.DataFrame, french_annual: pd.DataFrame, sample_years: list[int]
) -> pd.DataFrame:
    """Line up our Fama-MacBeth premia against French's realized annual
    factor returns *over the same sample years*, so the comparison isn't
    apples-to-oranges against French's full 1963-present history.
    """
    french_sample = french_annual.loc[french_annual.index.isin(sample_years)]
    rows = []
    for our_factor, ff_factor in FACTOR_TO_FF.items():
        our_row = fm_pass2.loc[our_factor] if our_factor in fm_pass2.index else None
        ff_mean = french_sample[ff_factor].mean() if ff_factor in french_sample.columns else np.nan
        rows.append(
            {
                "our_factor": our_factor,
                "ff_benchmark": ff_factor,
                "our_mean_premium": our_row["mean_premium"] if our_row is not None else np.nan,
                "our_t_stat": our_row["t_stat"] if our_row is not None else np.nan,
                "ff_mean_annual_return": ff_mean,
                "same_sign": (
                    np.sign(our_row["mean_premium"]) == np.sign(ff_mean)
                    if our_row is not None and not np.isnan(ff_mean)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)
