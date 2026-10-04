"""
Daily adjusted closing prices for the candidate pool, from yfinance, cached in data/prices/.

Run:  python src/prices.py            (download if no cache yet, then print a summary)
      python src/prices.py --refresh  (download again and overwrite the cache)

The trading calendar used for labelling is the set of dates in this price table, so
exchange holidays are handled correctly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRICES_DIR = PROJECT_ROOT / "data" / "prices"
CACHE_PATH = PRICES_DIR / "adj_close.csv"

# Covers both text sources (tweets 2017-01..2018-10, news 2009-02..2020-06) plus room for the
# 10-trading-day windows before and after each item.
START, END = "2009-01-01", "2020-07-31"


def download_adj_close(tickers, start: str = START, end: str = END) -> pd.DataFrame:
    """Adjusted closes (split- and dividend-adjusted), one column per ticker, one row per trading day."""
    import yfinance as yf
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    closes = data["Close"]
    if isinstance(closes, pd.Series):               # a single ticker comes back as a Series
        closes = closes.to_frame(tickers[0])
    closes.index = pd.to_datetime(closes.index)
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index.name = "date"
    return closes.dropna(how="all").sort_index()[sorted(closes.columns)]


def load_adj_close(tickers=None, refresh: bool = False, cache_path: Path = CACHE_PATH,
                   downloader=download_adj_close) -> pd.DataFrame:
    """Read prices from the cache; download (and cache) them if there is no cache or refresh=True."""
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, index_col="date", parse_dates=["date"])
    if tickers is None:
        raise ValueError("no price cache yet: pass the tickers to download")
    prices = downloader(list(tickers))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, float_format="%.6f")
    cache_path.with_suffix(".info.txt").write_text(
        f"Adjusted closes from yfinance (auto_adjust=True), {START} to {END}.\n"
        f"Downloaded {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC.\n", encoding="utf-8")
    return pd.read_csv(cache_path, index_col="date", parse_dates=["date"])


def summarize(prices: pd.DataFrame) -> pd.DataFrame:
    """First/last date with a price and number of trading days, per ticker."""
    return pd.DataFrame({
        "first": prices.apply(lambda s: s.first_valid_index()),
        "last": prices.apply(lambda s: s.last_valid_index()),
        "days": prices.notna().sum(),
    })


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inspect_data import POOL

    parser = argparse.ArgumentParser(description="Download/cache adjusted closes for the pool")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    prices = load_adj_close([t for t, *_ in POOL], refresh=args.refresh)
    print(f"{prices.shape[1]} tickers x {prices.shape[0]} trading days "
          f"({prices.index.min().date()} -> {prices.index.max().date()}), cached in {CACHE_PATH.relative_to(PROJECT_ROOT)}")
    print(summarize(prices).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
