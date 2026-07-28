"""
Data acquisition for the multi-factor equity model.

Two vendors:
  - Financial Modeling Prep (FMP): annual fundamentals. Requires an API key.
  - yfinance: daily adjusted prices (all tickers, full history, no key), and
    a *fallback* source for annual fundamentals.

Why a fallback: FMP's free tier turned out, empirically, to whitelist only
a subset of symbols for the `key-metrics` / `ratios` endpoints (roughly the
largest ~35-40 most-traded names in this universe returned HTTP 402 for
everything else -- confirmed by probing all ~200 candidate tickers before
committing to this design). Building a 100+ name cross-sectional universe on
FMP alone is not possible on this subscription tier. So: FMP is tried first
for every ticker (it has properly-computed diluted EPS, adjusted book value,
etc.); any ticker FMP won't serve falls back to fundamentals derived from
yfinance's own annual income statement + balance sheet + our own price
panel. Every row carries a `source` column ("fmp" or "yfinance") so this
vendor quirk is visible in the data, not hidden by it -- see the README
limitations section.

Everything is cached to disk under DATA_DIR so a rerun (or a Colab restart)
never re-hits the network for data it already has. The universe itself is a
static, hand-maintained CSV rather than a live index pull, since FMP's free
tier also rejects the S&P 500 constituents endpoint -- see the "Point-in-time
hygiene" markdown cell in the notebook for why that's a deliberate, documented
choice rather than an oversight.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
FMP_ANNUAL_LIMIT = 5          # hard cap on this FMP subscription tier
FILING_LAG_DAYS = 90          # conservative pad past typical 10-K filing deadlines


# --------------------------------------------------------------------------- #
# API key resolution: Colab secret -> environment variable -> interactive prompt
# --------------------------------------------------------------------------- #
def get_fmp_api_key() -> str:
    """Resolve the FMP API key without ever hardcoding it.

    Order of preference:
      1. google.colab.userdata (Colab's secrets manager)
      2. FMP_API_KEY environment variable
      3. getpass prompt (local, interactive sessions only)
    """
    try:
        from google.colab import userdata  # type: ignore

        key = userdata.get("FMP_API_KEY")
        if key:
            return key
    except Exception:
        pass

    key = os.environ.get("FMP_API_KEY")
    if key:
        return key

    import getpass

    key = getpass.getpass("Enter your Financial Modeling Prep API key: ")
    if not key:
        raise RuntimeError(
            "No FMP API key found. Set it via Colab secrets (name it "
            "FMP_API_KEY), `export FMP_API_KEY=...`, or enter it when prompted."
        )
    return key


# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def load_universe(data_dir: str | Path) -> pd.DataFrame:
    """Load the static ticker/sector universe.

    This is a fixed, hand-picked snapshot of ~100 large/mid-cap US equities
    across 10 GICS sectors -- NOT a point-in-time historical index membership
    list. Applying today's constituents retroactively over the backtest
    window introduces survivorship bias (companies that were removed from
    the S&P 500 for underperforming, or went bankrupt, are absent). This is
    documented explicitly in the README limitations section rather than
    quietly assumed away; a production system would use point-in-time index
    membership from a vendor like CRSP.
    """
    path = Path(data_dir) / "universe.csv"
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Generic cached GET helper
# --------------------------------------------------------------------------- #
# Circuit breaker: once we see this many consecutive 429s, FMP is globally
# rate-limited for the rest of this run. There is no point burning minutes
# retrying calls that all get throttled -- every remaining ticker should go
# straight to the yfinance fallback instead. Reset per fresh process.
_FMP_CONSECUTIVE_429S = 0
_FMP_CIRCUIT_OPEN_AFTER = 3


def fmp_circuit_open() -> bool:
    return _FMP_CONSECUTIVE_429S >= _FMP_CIRCUIT_OPEN_AFTER


def _cached_json_get(
    url: str, cache_path: Path, force_refresh: bool = False, max_retries: int = 2
) -> list | dict:
    """GET + cache a JSON endpoint.

    Returns [] (not an exception) for symbols this FMP subscription tier
    doesn't cover (402/403), so one gated ticker never aborts a 100+ ticker
    pull -- it just gets logged and falls back to yfinance upstream. A 429
    (rate limited) gets one short backoff retry; if the *global* rate limit
    trips (see `fmp_circuit_open`), we stop even trying FMP for the rest of
    the run rather than burning minutes on calls that will all be throttled.
    """
    global _FMP_CONSECUTIVE_429S

    if cache_path.exists() and not force_refresh:
        with open(cache_path) as f:
            return json.load(f)

    if fmp_circuit_open():
        return []

    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(url, timeout=20)
        if resp.status_code in (402, 403):
            _FMP_CONSECUTIVE_429S = 0
            print(f"    (subscription-gated, skipping) {url.split('?')[0]} -> HTTP {resp.status_code}")
            return []
        if resp.status_code == 429:
            _FMP_CONSECUTIVE_429S += 1
            if fmp_circuit_open():
                print("    (FMP rate limit circuit open -- switching to yfinance for the rest of this run)")
                return []
            print(f"    (rate limited, retry {attempt+1}/{max_retries} in {delay:.0f}s)")
            time.sleep(delay)
            delay *= 2
            continue
        _FMP_CONSECUTIVE_429S = 0
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict) and ("Error Message" in payload or "error" in payload):
            print(f"    (API error, skipping) {payload}")
            return []

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f)
        return payload

    print(f"    (still rate limited after {max_retries} retries, falling back) {url.split('?')[0]}")
    return []


# --------------------------------------------------------------------------- #
# FMP fundamentals
# --------------------------------------------------------------------------- #
def fetch_key_metrics_annual(
    ticker: str, api_key: str, data_dir: str | Path, force_refresh: bool = False
) -> pd.DataFrame:
    """Annual key-metrics: market cap, earnings yield, ROE, ROIC, FCF yield."""
    cache_path = Path(data_dir) / "raw" / "fmp" / f"{ticker}_key_metrics.json"
    url = f"{FMP_BASE_URL}/key-metrics?symbol={ticker}&limit={FMP_ANNUAL_LIMIT}&apikey={api_key}"
    payload = _cached_json_get(url, cache_path, force_refresh)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


def fetch_ratios_annual(
    ticker: str, api_key: str, data_dir: str | Path, force_refresh: bool = False
) -> pd.DataFrame:
    """Annual ratios: price/book, margins, book value per share, dividend yield."""
    cache_path = Path(data_dir) / "raw" / "fmp" / f"{ticker}_ratios.json"
    url = (
        f"{FMP_BASE_URL}/ratios?symbol={ticker}&period=annual"
        f"&limit={FMP_ANNUAL_LIMIT}&apikey={api_key}"
    )
    payload = _cached_json_get(url, cache_path, force_refresh)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    return pd.DataFrame(payload)


KEY_METRICS_COLS = [
    "symbol", "date", "fiscalYear", "marketCap", "earningsYield",
    "returnOnEquity", "returnOnInvestedCapital", "freeCashFlowYield",
]
RATIOS_COLS = [
    "symbol", "date", "priceToBookRatio", "netProfitMargin",
    "grossProfitMargin", "bookValuePerShare", "dividendYield",
]
# Common schema every fundamentals row is normalized to, regardless of source.
FUNDAMENTALS_SCHEMA = [
    "ticker", "sector", "fiscal_date", "marketCap", "earningsYield",
    "returnOnEquity", "priceToBookRatio", "netProfitMargin",
    "grossProfitMargin", "bookValuePerShare", "dividendYield", "source",
]


def _fetch_fmp_fundamentals(ticker: str, api_key: str, data_dir: str | Path, force_refresh: bool) -> pd.DataFrame:
    km = fetch_key_metrics_annual(ticker, api_key, data_dir, force_refresh)
    ratios = fetch_ratios_annual(ticker, api_key, data_dir, force_refresh)
    if km.empty or ratios.empty:
        return pd.DataFrame()

    km = km[[c for c in KEY_METRICS_COLS if c in km.columns]]
    ratios = ratios[[c for c in RATIOS_COLS if c in ratios.columns]]
    merged = km.merge(ratios, on=["symbol", "date"], how="inner")
    merged = merged.rename(columns={"symbol": "ticker", "date": "fiscal_date"})
    merged["source"] = "fmp"
    return merged


def _fetch_yfinance_fundamentals(
    ticker: str, prices: pd.DataFrame, data_dir: str | Path, force_refresh: bool = False
) -> pd.DataFrame:
    """Derive the same fundamentals schema from yfinance's raw statements.

    yfinance gives us the underlying line items (net income, shareholders'
    equity, shares outstanding, revenue, gross profit) but not pre-computed
    ratios, so we compute price/book, earnings yield, ROE and margins
    ourselves, using shares outstanding *as reported at each fiscal
    period end* (not today's share count) combined with the price on our
    own price panel as of that date -- this keeps the market-cap-derived
    ratios point-in-time consistent with the FMP-sourced rows.
    """
    import yfinance as yf

    inc_path = Path(data_dir) / "raw" / "yfinance" / f"{ticker}_income.csv"
    bal_path = Path(data_dir) / "raw" / "yfinance" / f"{ticker}_balance.csv"

    if inc_path.exists() and bal_path.exists() and not force_refresh:
        inc = pd.read_csv(inc_path, index_col=0, parse_dates=True)
        bal = pd.read_csv(bal_path, index_col=0, parse_dates=True)
    else:
        t = yf.Ticker(ticker)
        inc = t.get_income_stmt(freq="yearly").T
        bal = t.get_balance_sheet(freq="yearly").T
        inc_path.parent.mkdir(parents=True, exist_ok=True)
        inc.to_csv(inc_path)
        bal.to_csv(bal_path)

    need_inc = {"TotalRevenue", "NetIncome"}
    need_bal = {"StockholdersEquity", "OrdinarySharesNumber"}
    if not need_inc.issubset(inc.columns) or not need_bal.issubset(bal.columns):
        return pd.DataFrame()
    has_gross_profit = "GrossProfit" in inc.columns

    inc_cols = ["TotalRevenue", "NetIncome"] + (["GrossProfit"] if has_gross_profit else [])
    df = inc[inc_cols].join(bal[["StockholdersEquity", "OrdinarySharesNumber"]], how="inner")
    df = df.dropna(subset=["TotalRevenue", "NetIncome", "StockholdersEquity", "OrdinarySharesNumber"])
    if df.empty or ticker not in prices.columns:
        return pd.DataFrame()
    if not has_gross_profit:
        df["GrossProfit"] = np.nan

    df.index.name = "fiscal_date"
    df = df.reset_index()
    df["fiscal_date"] = pd.to_datetime(df["fiscal_date"])
    df["price"] = df["fiscal_date"].apply(lambda d: _price_asof_scalar(prices[ticker], d))
    df = df.dropna(subset=["price"])
    if df.empty:
        return pd.DataFrame()

    df["marketCap"] = df["price"] * df["OrdinarySharesNumber"]
    df["bookValuePerShare"] = df["StockholdersEquity"] / df["OrdinarySharesNumber"]
    df["priceToBookRatio"] = df["price"] / df["bookValuePerShare"]
    eps = df["NetIncome"] / df["OrdinarySharesNumber"]
    df["earningsYield"] = eps / df["price"]
    df["returnOnEquity"] = df["NetIncome"] / df["StockholdersEquity"]
    df["netProfitMargin"] = df["NetIncome"] / df["TotalRevenue"]
    df["grossProfitMargin"] = df["GrossProfit"] / df["TotalRevenue"]
    df["dividendYield"] = np.nan  # not reconstructed point-in-time from yfinance
    df["ticker"] = ticker
    df["source"] = "yfinance"
    return df[[c for c in FUNDAMENTALS_SCHEMA if c != "sector"]]


def _price_asof_scalar(series: pd.Series, date: pd.Timestamp) -> float:
    idx = series.index.searchsorted(date, side="right") - 1
    return series.iloc[idx] if idx >= 0 else np.nan


def build_fundamentals_panel(
    universe: pd.DataFrame,
    api_key: str,
    prices: pd.DataFrame,
    data_dir: str | Path,
    sleep_sec: float = 0.25,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull + cache annual fundamentals for every ticker in the universe.

    Tries FMP first, falls back to a yfinance-derived reconstruction for any
    ticker FMP's free tier won't serve (see module docstring). `prices` must
    already cover the full universe -- the yfinance fallback needs it to
    compute point-in-time market cap.

    Point-in-time hygiene: every row -- regardless of source -- is stamped
    with `available_date = fiscal_date + FILING_LAG_DAYS`, a deliberately
    conservative pad past the ~60-75 day 10-K deadline large caps typically
    meet. No factor or return computed downstream is allowed to use a
    fundamental row before its `available_date` -- see factors.py.
    """
    rows = []
    n = len(universe)
    n_fmp = n_yf = n_skipped = 0
    for i, r in enumerate(universe.itertuples(index=False)):
        ticker = r.ticker
        merged = _fetch_fmp_fundamentals(ticker, api_key, data_dir, force_refresh)
        if not merged.empty:
            n_fmp += 1
            if not (Path(data_dir) / "raw" / "fmp" / f"{ticker}_key_metrics.json").exists():
                time.sleep(sleep_sec)
        else:
            merged = _fetch_yfinance_fundamentals(ticker, prices, data_dir, force_refresh)
            if not merged.empty:
                n_yf += 1

        if merged.empty:
            n_skipped += 1
            print(f"  [{i+1}/{n}] {ticker}: no fundamentals from either source, skipping")
            continue

        merged["sector"] = r.sector
        rows.append(merged)

    print(f"Fundamentals sourced: {n_fmp} from FMP, {n_yf} from yfinance, {n_skipped} skipped entirely")
    if not rows:
        raise RuntimeError("No fundamentals data was retrieved for any ticker.")

    panel = pd.concat(rows, ignore_index=True)
    panel["fiscal_date"] = pd.to_datetime(panel["fiscal_date"])
    panel["available_date"] = panel["fiscal_date"] + pd.Timedelta(days=FILING_LAG_DAYS)
    panel["fiscal_year"] = panel["fiscal_date"].dt.year
    return panel.sort_values(["ticker", "fiscal_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prices (yfinance -- no key required, full history)
# --------------------------------------------------------------------------- #
def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    data_dir: str | Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Daily adjusted close for every ticker, wide format (date x ticker).

    Cached as a single parquet/csv so reruns are instant. yfinance is used
    (rather than FMP) for all price history: it needs no API key, has no
    meaningful rate limit for a universe this size, and covers years of
    daily history, which the free FMP tier's fundamentals endpoints do not.
    """
    cache_path = Path(data_dir) / "raw" / "prices.csv"
    if cache_path.exists() and not force_refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Only trust the cache if it already covers every requested ticker.
        if set(tickers).issubset(set(prices.columns)):
            return prices

    import yfinance as yf

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.dropna(how="all").sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path)
    return prices
