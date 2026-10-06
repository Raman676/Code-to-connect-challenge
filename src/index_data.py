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
SPLITS_CSV = PRICES_DIR / "sp100_splits.csv"
SHARES_CLEAN_CSV = PRICES_DIR / "sp100_shares_clean.csv"

# Share-count cleaning
JUMP = 0.25                  # a share count changing by more than 25% between observations is a "jump"
REVERT_DAYS = 30             # ... and a spike if it comes back within 30 days
REVERT_TOLERANCE = 0.10      # "comes back" = within 10% of the value before the jump
KNOWN_EVENTS = {             # real share-count changes from corporate events (kept as reported);
                             # keyed by the month the new count first appears in the data
    ("AVGO", "2016-03"): "Broadcom Ltd merger (Avago acquires Broadcom, closed Feb 2016)",
    ("BMY", "2019-11"): "Celgene acquisition",
    ("TMUS", "2020-04"): "Sprint merger",
    ("CVS", "2018-12"): "Aetna acquisition (closed Nov 2018)",
}
# Single bad share-count observations removed by hand (reviewed 2026-10-06)
MANUAL_DROPS = {
    # 260M shares between ~628M (Dec 2015) and ~624M (Mar 2016); next value is 3 months later, so
    # the spike rule misses it. It is the value used at the first review (2015-12-31).
    ("ACN", "2015-12-17"): "bad value (~40% of the real share count)",
    # Post-split count (3-for-1, effective 2017-10-06) reported before the split date; next value is
    # 14 months later, so market cap would be tripled from Aug 2017 to Nov 2018.
    ("ISRG", "2017-08-28"): "post-split count reported before the split date",
    # One-day value of 750M between ~612M values; the +23% jump is just under the 25% spike threshold.
    ("NVDA", "2020-03-17"): "one-day outlier (750M vs ~612M)",
}
# yfinance share history starts when a company took its CURRENT ticker. For these two, earlier
# counts are back-filled with the first known split-adjusted count (approved 2026-10-06): their
# share counts barely changed in 2016-2020. LIN, RTX and DELL are NOT back-filled (mergers or
# restructurings changed their share counts a lot), so they stay unranked before their data starts.
BACKFILL_SHARES = {
    "META": "ticker was FB until June 2022; share history starts 2022-06",
    "BKNG": "ticker was PCLN until Feb 2018; share history starts 2017-08",
}
BACKFILL_FROM = "2009-01-01"
# Free-float factors that replace yfinance's floatShares / sharesOutstanding (see the report)
FLOAT_OVERRIDES = {
    "BRK.B": (0.85, "our estimate: excludes roughly Buffett's ~15% economic stake; yfinance float unusable"),
}
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


def download_splits(symbols) -> pd.DataFrame:
    """Long table (symbol, date, ratio); ratio 2.0 = a 2-for-1 split."""
    import yfinance as yf
    rows = []
    for symbol in symbols:
        for attempt in range(3):
            try:
                s = yf.Ticker(symbol).splits
                break
            except Exception:  # noqa: BLE001
                s = None
                time.sleep(2)
        if s is None:
            continue
        idx = pd.to_datetime(s.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        rows += [{"symbol": symbol, "date": d.normalize(), "ratio": float(r)} for d, r in zip(idx, s.to_numpy())]
    return pd.DataFrame(rows, columns=["symbol", "date", "ratio"])


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
    if refresh or not SPLITS_CSV.exists():
        print("  downloading splits ...", file=sys.__stdout__, flush=True)
        download_splits(symbols).to_csv(SPLITS_CSV, index=False)


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


def split_adjust_shares(shares: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Put share counts on today's split basis (like yfinance closes): multiply each count by the
    product of all split ratios that happen AFTER its date."""
    out = shares.copy()
    out["split_factor"] = 1.0
    for symbol, sp in splits.groupby("symbol"):
        rows = out["symbol"] == symbol
        for date, ratio in zip(sp["date"], sp["ratio"]):
            out.loc[rows & (out["date"] < date), "split_factor"] *= ratio
    out["shares_adj"] = out["shares"] * out["split_factor"]
    return out


def clean_shares(shares: pd.DataFrame, known_events: dict = KNOWN_EVENTS,
                 manual_drops: dict = MANUAL_DROPS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove MANUAL_DROPS and isolated spikes from split-adjusted share counts, and list every remaining jump.
    A spike: a jump of more than JUMP that returns to within REVERT_TOLERANCE of the earlier value
    within REVERT_DAYS -> the observations in between are dropped.
    Remaining jumps are 'kept (known event)' if listed in known_events, else 'flagged for review'
    (kept as reported until reviewed). Returns (cleaned shares, events table).
    """
    keep, events = [], []
    manual = pd.Series([(s, d.strftime("%Y-%m-%d")) in manual_drops for s, d in zip(shares["symbol"], shares["date"])],
                       index=shares.index)
    for (symbol, date), reason in manual_drops.items():
        hit = shares[(shares["symbol"] == symbol) & (shares["date"] == pd.Timestamp(date))]
        if len(hit):
            events.append({"symbol": symbol, "date": pd.Timestamp(date).date(), "before": np.nan,
                           "after": hit["shares_adj"].iloc[0], "change %": np.nan, "action": f"manual drop: {reason}"})
    shares = shares[~manual]
    for symbol, g in shares.sort_values("date").groupby("symbol"):
        dates, values = g["date"].to_numpy(), g["shares_adj"].to_numpy()
        drop = np.zeros(len(g), dtype=bool)
        last_kept = 0                                           # compare with the last value we keep
        i = 1
        while i < len(g):
            before = values[last_kept]
            if abs(values[i] / before - 1) > JUMP:
                horizon = dates[i] + np.timedelta64(REVERT_DAYS, "D")
                j = i
                while j < len(g) and dates[j] <= horizon and abs(values[j] / before - 1) > REVERT_TOLERANCE:
                    j += 1
                if j < len(g) and dates[j] <= horizon:              # came back: a spike
                    drop[i:j] = True
                    events.append({"symbol": symbol, "date": pd.Timestamp(dates[i]).date(), "before": before,
                                   "after": values[i], "change %": round(100 * (values[i] / before - 1), 1),
                                   "action": f"spike removed ({j - i} obs)"})
                    last_kept, i = j, j + 1
                    continue
                month = pd.Timestamp(dates[i]).strftime("%Y-%m")
                known = known_events.get((symbol, month))
                events.append({"symbol": symbol, "date": pd.Timestamp(dates[i]).date(), "before": before,
                               "after": values[i], "change %": round(100 * (values[i] / before - 1), 1),
                               "action": f"kept (known event: {known})" if known else "flagged for review (kept)"})
            last_kept = i
            i += 1
        keep.append(g[~drop])
    return pd.concat(keep, ignore_index=True), pd.DataFrame(events)


def backfill_shares(clean: pd.DataFrame, symbols: dict = BACKFILL_SHARES, since: str = BACKFILL_FROM) -> pd.DataFrame:
    """Add one observation dated `since` holding each listed symbol's first known split-adjusted count,
    so the as-of lookup has a value before the share history starts. Marked backfilled=True."""
    out = clean.assign(backfilled=False)
    extra = []
    for symbol in symbols:
        h = out[out["symbol"] == symbol].sort_values("date")
        if h.empty or h["date"].iloc[0] <= pd.Timestamp(since):
            continue
        first = h.iloc[0]
        extra.append({**first.to_dict(), "date": pd.Timestamp(since), "backfilled": True})
    return pd.concat([out, pd.DataFrame(extra)], ignore_index=True) if extra else out


def float_factors(info: pd.DataFrame, overrides: dict = FLOAT_OVERRIDES) -> pd.DataFrame:
    """
    Free-float factor per ticker, assumed constant over time (today's value).
    Default: floatShares / sharesOutstanding, capped at 1. Alphabet: float over ALL share classes
    (impliedSharesOutstanding), since sharesOutstanding counts one class only. Plus manual overrides.
    """
    out = info[["ticker", "float_shares", "shares_out", "implied_shares_out"]].copy()
    out["factor"] = (out["float_shares"] / out["shares_out"]).clip(upper=1.0)
    out["source"] = "yfinance floatShares / sharesOutstanding"
    alphabet = out["ticker"].isin(["GOOGL", "GOOG"])
    out.loc[alphabet, "factor"] = out.loc[alphabet, "float_shares"] / out.loc[alphabet, "implied_shares_out"]
    out.loc[alphabet, "source"] = "Alphabet: floatShares / impliedSharesOutstanding (all share classes)"
    for ticker, (factor, why) in overrides.items():
        out.loc[out["ticker"] == ticker, ["factor", "source"]] = [factor, why]
    return out[["ticker", "factor", "source"]]


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

        section("7. SHARE-DATA CLEANING (split basis, spikes, remaining jumps) AND FREE-FLOAT FACTORS")
        splits = pd.read_csv(SPLITS_CSV, parse_dates=["date"])
        in_data = splits[(splits["date"] >= "2015-01-01") & (splits["date"] <= "2026-12-31")]
        print(f"Splits downloaded: {len(splits)} for {splits['symbol'].nunique()} symbols; "
              f"from 2015 on: {len(in_data)} "
              f"{ {f'{s} {d.date()}': r for s, d, r in zip(in_data['symbol'], in_data['date'], in_data['ratio'])} }")
        adjusted = split_adjust_shares(shares, splits)
        clean, events = clean_shares(adjusted)
        clean.to_csv(SHARES_CLEAN_CSV, index=False)
        print(f"Share observations: {len(adjusted):,} -> {len(clean):,} after removing spikes "
              f"(saved to {SHARES_CLEAN_CSV.relative_to(PROJECT_ROOT)})")
        if len(events):
            events["ticker"] = events["symbol"].map(sym_to_ticker)
            events["in backtest window"] = (pd.to_datetime(events["date"]) >= "2015-10-01") & \
                                           (pd.to_datetime(events["date"]) <= BACKTEST_END)
            with pd.option_context("display.width", 250, "display.float_format", "{:,.0f}".format,
                                   "display.max_colwidth", 80):
                for label, part in (("Manual drops (reviewed)", events[events["action"].str.startswith("manual")]),
                                    ("Spikes removed", events[events["action"].str.startswith("spike")]),
                                    ("Known corporate events (kept)", events[events["action"].str.startswith("kept")]),
                                    ("FLAGGED for review (kept as reported until you decide)",
                                     events[events["action"].str.startswith("flagged")])):
                    print(f"\n{label}: {len(part)}")
                    if len(part):
                        cols = ["ticker", "date", "before", "after", "change %", "in backtest window", "action"]
                        print(indent(part[cols].to_string(index=False)))

        # Re-scan: market-cap jumps without a matching price move, after cleaning (backtest window)
        window = close.loc["2015-12-31":BACKTEST_END]
        residual = []
        for sym in universe["yf_symbol"]:
            if sym not in window or window[sym].notna().sum() == 0:
                continue
            h = clean[clean["symbol"] == sym].assign(shares=lambda d: d["shares_adj"])
            mcap = window[sym] * shares_asof(h, sym, window.index).to_numpy()
            change, price_change = mcap.pct_change().abs(), window[sym].pct_change().abs()
            for d in change[(change > 0.3) & (price_change < 0.2)].index:
                residual.append(f"{sym_to_ticker.get(sym, sym)} {d.date()} ({100 * change[d]:+.0f}%)")
        print(f"\nAfter cleaning, market-cap jumps > 30% without a price move > 20% (2016 - Jun 2020): "
              f"{residual or 'none'}")

        factors = float_factors(info)
        print("\nFree-float factors (today's value, assumed constant 2016-2020):")
        print(f"  distribution: {pd.cut(factors['factor'], [0, 0.5, 0.8, 0.9, 0.95, 1.0]).value_counts().sort_index().to_dict()}")
        special = factors[factors["source"] != "yfinance floatShares / sharesOutstanding"]
        low = factors[factors["factor"] < 0.8]
        with pd.option_context("display.width", 250, "display.max_colwidth", 110):
            print("  overrides / special cases:")
            print(indent(special.round(3).to_string(index=False), "    "))
            print("  factors below 0.8 (genuine insider/parent holdings unless overridden):")
            print(indent(low.round(3).to_string(index=False), "    "))

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
