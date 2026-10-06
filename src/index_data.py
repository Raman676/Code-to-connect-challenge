"""
Market data for the S&P 100 index backtest, plus a data-check report.

Run:  python src/index_data.py            (download what is not cached yet, write docs/index_data_checks.txt)
      python src/index_data.py --refresh  (download everything again)

Universe: data/universe/sp100_constituents.csv = current S&P 100 members (Wikipedia "S&P 100"
constituents table, as of 2026-09-21; 101 rows because Alphabet has two share classes).

Downloads (yfinance, cached in data/prices/):
  sp100_close.csv / sp100_adj_close.csv  daily unadjusted / adjusted closes, 2009-2020
                                         (unadjusted x shares = market cap; adjusted -> returns)
  sp100_shares_history.csv               shares outstanding over time (Ticker.get_shares_full)
  sp100_info.csv                         today's snapshot: price, market cap, shares, float shares, sector
  benchmarks.csv                         ^OEX (S&P 100, price index) and ^IRX (13-week T-bill yield, %)
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_data import NEWS_FILES, Tee, indent, read_chunks, section  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = PROJECT_ROOT / "data" / "universe" / "sp100_constituents.csv"
PRICES_DIR = PROJECT_ROOT / "data" / "prices"
CLOSE_CSV = PRICES_DIR / "sp100_close.csv"
ADJ_CLOSE_CSV = PRICES_DIR / "sp100_adj_close.csv"
SHARES_CSV = PRICES_DIR / "sp100_shares_history.csv"
INFO_CSV = PRICES_DIR / "sp100_info.csv"
BENCHMARK_CSV = PRICES_DIR / "benchmarks.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "index_data_checks.txt"

START, END = "2009-01-01", "2020-12-31"
BACKTEST_START, BACKTEST_END = "2012-01-01", "2020-06-30"
BENCHMARKS = {"^OEX": "sp100", "^IRX": "tbill_13w_pct"}
FLOAT_RATIO_OK = (0.5, 1.05)        # float shares / shares outstanding outside this range -> check by hand


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def load_universe(path: Path = UNIVERSE_CSV) -> pd.DataFrame:
    """Constituents with news_symbols split into a list (old tickers used in the 2009-2020 news data)."""
    df = pd.read_csv(path)
    df["news_symbols"] = df["news_symbols"].str.split("|")
    return df


# ---------------------------------------------------------------------------
# Downloads (cached)
# ---------------------------------------------------------------------------

def download_closes(symbols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unadjusted and adjusted daily closes in one call."""
    import yfinance as yf
    data = yf.download(list(symbols), start=START, end=END, auto_adjust=False, progress=False)
    close, adj = data["Close"], data["Adj Close"]
    for df in (close, adj):
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
        df.index.name = "date"
    return close.dropna(how="all").sort_index(), adj.dropna(how="all").sort_index()


def download_shares_history(symbols) -> pd.DataFrame:
    """Long table (symbol, date, shares); one value per symbol and day (the last reported)."""
    import yfinance as yf
    parts = []
    for symbol in symbols:
        print(f"  shares {symbol} ...", file=sys.__stdout__, flush=True)
        for attempt in range(3):
            try:
                s = yf.Ticker(symbol).get_shares_full(start="2005-01-01")
                break
            except Exception:  # noqa: BLE001 - network hiccups: retry, then give up on this symbol
                s = None
                time.sleep(2)
        if s is None or len(s) == 0:
            continue
        idx = pd.to_datetime(s.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        df = pd.DataFrame({"symbol": symbol, "date": idx.normalize(), "shares": s.to_numpy(dtype=float)})
        parts.append(df.groupby("date", as_index=False).last().assign(symbol=symbol))
    return pd.concat(parts, ignore_index=True)[["symbol", "date", "shares"]]


def download_info(symbols) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for symbol in symbols:
        print(f"  info {symbol} ...", file=sys.__stdout__, flush=True)
        info = {}
        for attempt in range(3):
            try:
                info = yf.Ticker(symbol).info
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        rows.append({"symbol": symbol, "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                     "market_cap": info.get("marketCap"), "shares_out": info.get("sharesOutstanding"),
                     "implied_shares_out": info.get("impliedSharesOutstanding"),
                     "float_shares": info.get("floatShares"), "yf_sector": info.get("sector")})
    df = pd.DataFrame(rows)
    df["fetched_utc"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}"
    return df


def download_benchmarks() -> pd.DataFrame:
    import yfinance as yf
    data = yf.download(list(BENCHMARKS), start=START, end=END, auto_adjust=False, progress=False)
    out = data["Close"].rename(columns=BENCHMARKS)
    out["sp100_adj"] = data["Adj Close"]["^OEX"]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def ensure_downloads(universe: pd.DataFrame, refresh: bool = False):
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    symbols = universe["yf_symbol"].tolist()
    if refresh or not (CLOSE_CSV.exists() and ADJ_CLOSE_CSV.exists()):
        print("  downloading closes ...", file=sys.__stdout__, flush=True)
        close, adj = download_closes(symbols)
        close.to_csv(CLOSE_CSV, float_format="%.6f")
        adj.to_csv(ADJ_CLOSE_CSV, float_format="%.6f")
    if refresh or not SHARES_CSV.exists():
        download_shares_history(symbols).to_csv(SHARES_CSV, index=False)
    if refresh or not INFO_CSV.exists():
        download_info(symbols).to_csv(INFO_CSV, index=False)
    if refresh or not BENCHMARK_CSV.exists():
        download_benchmarks().to_csv(BENCHMARK_CSV, float_format="%.6f")


def load_prices():
    read = lambda p: pd.read_csv(p, index_col="date", parse_dates=["date"])  # noqa: E731
    return read(CLOSE_CSV), read(ADJ_CLOSE_CSV)


def load_shares() -> pd.DataFrame:
    return pd.read_csv(SHARES_CSV, parse_dates=["date"])


# ---------------------------------------------------------------------------
# Helpers used by the checks (and later by the index)
# ---------------------------------------------------------------------------

def shares_asof(history: pd.DataFrame, symbol: str, dates, max_age_days: int | None = None) -> pd.Series:
    """Last reported share count on or before each date (NaN if none, or older than max_age_days)."""
    h = history[history["symbol"] == symbol].sort_values("date")
    dates = pd.DatetimeIndex(dates)
    if h.empty:
        return pd.Series(np.nan, index=dates)
    pos = h["date"].searchsorted(dates, side="right") - 1
    values = np.where(pos >= 0, h["shares"].to_numpy()[np.clip(pos, 0, None)], np.nan)
    if max_age_days is not None:
        age = (dates - h["date"].to_numpy()[np.clip(pos, 0, None)]).days
        values = np.where((pos >= 0) & (age <= max_age_days), values, np.nan)
    return pd.Series(values, index=dates)


def float_ratio(info: pd.DataFrame) -> pd.Series:
    """float shares / shares outstanding, as reported by yfinance (before any manual fixes)."""
    return (info["float_shares"] / info["shares_out"]).rename("float_ratio")


def quarter_end_reviews(calendar: pd.DatetimeIndex, start=BACKTEST_START, end=BACKTEST_END) -> pd.DatetimeIndex:
    """Last trading day of March, June, September and December."""
    days = calendar[(calendar >= start) & (calendar <= end)]
    days = days[days.month.isin([3, 6, 9, 12])]          # partial quarters at the end get no review
    s = pd.Series(days, index=days)
    last = s.groupby([days.year, days.month]).max()
    return pd.DatetimeIndex(last.to_numpy())


def news_counts_by_year(universe: pd.DataFrame) -> pd.DataFrame:
    """Headlines per company and year in analyst_ratings_processed.csv (all symbols a company used)."""
    parts = []
    for chunk in read_chunks(NEWS_FILES["analyst_ratings_processed"]):
        parts.append(pd.DataFrame({"symbol": chunk["stock"], "year": chunk["date"].str[:4].astype(int)}))
    counts = pd.concat(parts).groupby(["symbol", "year"]).size()
    years = range(2009, 2021)
    rows, aliases_found = {}, {}
    for _, r in universe.iterrows():
        total = pd.Series(0, index=years)
        found = {}
        for sym in r["news_symbols"]:
            if sym in counts.index.get_level_values(0):
                per_year = counts.xs(sym, level="symbol").reindex(years, fill_value=0)
                total = total + per_year
                found[sym] = int(per_year.sum())
        rows[r["ticker"]] = total
        aliases_found[r["ticker"]] = found
    return pd.DataFrame(rows).T, aliases_found


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download S&P 100 market data and check it")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)

    universe = load_universe()
    ensure_downloads(universe, refresh=args.refresh)
    close, adj = load_prices()
    shares = load_shares()
    info = pd.read_csv(INFO_CSV)
    bench = pd.read_csv(BENCHMARK_CSV, index_col="date", parse_dates=["date"])
    sym_to_ticker = dict(zip(universe["yf_symbol"], universe["ticker"]))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Index data checks - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/index_data.py")
        print(f"Universe: {len(universe)} symbols from {UNIVERSE_CSV.relative_to(PROJECT_ROOT)}; "
              f"backtest window {BACKTEST_START} -> {BACKTEST_END}")

        section("1. PRICES (yfinance daily closes)")
        first_price = close.apply(lambda s: s.first_valid_index())
        missing = [s for s in universe["yf_symbol"] if s not in close.columns or close[s].notna().sum() == 0]
        print(f"{close.shape[1]} symbols x {len(close):,} days ({close.index.min().date()} -> {close.index.max().date()})")
        print(f"No prices at all: {missing or 'none'}")
        late = first_price[first_price > pd.Timestamp(BACKTEST_START)].sort_values()
        print(f"Prices start after {BACKTEST_START} ({len(late)}): "
              f"{ {sym_to_ticker.get(s, s): str(d.date()) for s, d in late.items()} }")

        section("2. HISTORICAL SHARES OUTSTANDING (Ticker.get_shares_full)")
        per_symbol = shares.groupby("symbol").agg(first=("date", "min"), last=("date", "max"), points=("date", "size"))
        no_shares = sorted(set(universe["yf_symbol"]) - set(per_symbol.index))
        print(f"Symbols with any share history: {len(per_symbol)} / {len(universe)}; none: {no_shares or 'none'}")
        print(f"First year of share history: {per_symbol['first'].dt.year.value_counts().sort_index().to_dict()}")
        print(f"Median data points per symbol: {per_symbol['points'].median():.0f}")
        jumps = (shares.sort_values(["symbol", "date"]).groupby("symbol")["shares"].pct_change().abs() > 0.5)
        jumpy = shares.loc[jumps, "symbol"].value_counts()
        print(f"Symbols with day-to-day changes > 50% (possible data errors or splits): {len(jumpy)} "
              f"{dict(jumpy.head(15))}")
        reviews = quarter_end_reviews(close.index)
        both = []
        for review in reviews:
            n_price = int(close.loc[review].notna().sum())
            n_shares = sum(np.isfinite(shares_asof(shares, s, [review]).iloc[0])
                           and np.isfinite(close.loc[review].get(s, np.nan)) for s in universe["yf_symbol"])
            n_fresh = sum(np.isfinite(shares_asof(shares, s, [review], max_age_days=200).iloc[0])
                          and np.isfinite(close.loc[review].get(s, np.nan)) for s in universe["yf_symbol"])
            both.append({"review": review.date(), "with price": n_price, "price + any shares": n_shares,
                         "price + shares <= 200 days old": n_fresh})
        print("\nCompanies we can rank by market cap at each quarterly review (need at least 20; 25 for the buffer):")
        print(indent(pd.DataFrame(both).to_string(index=False)))

        section("3. CURRENT SNAPSHOT: shares and free float (yfinance .info)")
        info = info.assign(ticker=info["symbol"].map(sym_to_ticker), float_ratio=float_ratio(info).round(3))
        latest = shares.sort_values("date").groupby("symbol").last()
        info["hist_shares_x_price / market_cap"] = (
            info["symbol"].map(latest["shares"]) * info["price"] / info["market_cap"]).round(3)
        print(f"Snapshot fetched {info['fetched_utc'].iloc[0]} UTC; float shares missing for "
              f"{int(info['float_shares'].isna().sum())} symbols")
        print(f"float / shares outstanding: median {info['float_ratio'].median():.3f}; "
              f"distribution {pd.cut(info['float_ratio'], [0, 0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 10]).value_counts().sort_index().to_dict()}")
        lo, hi = FLOAT_RATIO_OK
        flagged = info[(info["float_ratio"] < lo) | (info["float_ratio"] > hi) | info["float_ratio"].isna()]
        cols = ["ticker", "price", "market_cap", "shares_out", "implied_shares_out", "float_shares", "float_ratio",
                "hist_shares_x_price / market_cap"]
        with pd.option_context("display.width", 250, "display.float_format", "{:,.3f}".format):
            print(f"\nFlagged (float ratio outside {FLOAT_RATIO_OK} or missing):")
            print(indent(flagged[cols].to_string(index=False)))
            off = info[(info["hist_shares_x_price / market_cap"] - 1).abs() > 0.1]
            print("\nLatest historical share count x price differs from market cap by > 10% "
                  "(share class issues for historical market caps):")
            print(indent(off[cols].to_string(index=False)))
            print("\nAlphabet (one company, traded as GOOGL):")
            print(indent(info[info["ticker"].isin(["GOOG", "GOOGL"])][cols].to_string(index=False)))

        section("4. SECTORS")
        merged = universe.merge(info[["ticker", "yf_sector"]], on="ticker")
        print(f"GICS sector (Wikipedia list): {universe['gics_sector'].notna().sum()} / {len(universe)}; "
              f"yfinance sector: {merged['yf_sector'].notna().sum()} / {len(merged)}")
        print(f"GICS sectors: {universe['gics_sector'].value_counts().to_dict()}")

        section("5. BENCHMARKS")
        for col in ("sp100", "tbill_13w_pct"):
            s = bench[col].dropna()
            print(f"{col}: {len(s):,} days, {s.index.min().date()} -> {s.index.max().date()}")
        same = np.allclose(bench["sp100"].dropna(), bench["sp100_adj"].dropna())
        print(f"^OEX Close == Adj Close: {same} -> {'price index only (no dividends)' if same else 'adjusted series differs'}")

        section("6. NEWS COVERAGE PER YEAR (analyst_ratings_processed.csv, S&P 100 incl. old tickers)")
        counts, aliases = news_counts_by_year(universe)
        print("Old/alternative tickers found: "
              f"{ {t: a for t, a in aliases.items() if len(a) > 1 or (a and list(a)[0] != t)} }")
        summary = pd.DataFrame({"headlines": counts.sum(), "companies >= 1": (counts >= 1).sum(),
                                "companies >= 20": (counts >= 20).sum(), "companies >= 50": (counts >= 50).sum(),
                                "median per company": counts.median()})
        summary.index.name = "year"
        print(indent(summary.reset_index().to_string(index=False)))
        total = counts.sum(axis=1).sort_values()
        print(f"\nNo headlines at all ({int((total == 0).sum())}): {sorted(total[total == 0].index)}")
        print(f"Exactly 10 headlines in total (suspected cap in the dataset): {sorted(total[total == 10].index)}")
        print(f"Fewest headlines (non-zero, 15 lowest): {total[total > 0].head(15).to_dict()}")

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
