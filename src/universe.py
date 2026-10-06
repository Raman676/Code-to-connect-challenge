"""
Index universe: free-float market cap (FFMC), ranking, Top-20 selection with a 20/25 buffer,
and quarterly constituent reviews (methodology sections 3-7 and 20).

Universe   current S&P 100 members (data/universe/sp100_constituents.csv); Alphabet counted once (GOOGL).
FFMC       close x split-adjusted shares outstanding (last value on or before the date) x free-float
           factor (today's value, assumed constant; BRK.B and Alphabet fixed by hand, see index_data.py).
           META and BKNG share counts are back-filled before their history starts (index_data.BACKFILL_SHARES);
           LIN, RTX and DELL are unranked until their share history starts.
Selection  exact-20 rule at each review:
             1. keep current members ranked 25 or better (the buffer),
             2. add every non-member ranked 20 or better (entry rule),
             3. if that gives more than 20, drop the lowest-ranked holdovers;
                if fewer, fill from the best-ranked remaining non-members.
           (The methodology's section 7 example ends with 21 names; step 3 resolves this.)
Reviews    last trading day of March, June, September and December; the new membership is traded
           at the close of the next trading day (the same one-day lag as the sentiment signals).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_data as idx  # noqa: E402

N_CONSTITUENTS = 20
ENTRY_RANK = 20
EXIT_RANK = 25
FIRST_REVIEW = "2015-12-31"
LAST_DAY = "2020-06-30"


@dataclass
class MarketData:
    close: pd.DataFrame          # split-adjusted closes (yfinance "Close"), dates x tickers
    adj_close: pd.DataFrame      # split- and dividend-adjusted closes, for returns
    shares: pd.DataFrame         # split-adjusted shares outstanding, as of each date (dates x tickers)
    float_factor: pd.Series      # per ticker
    sectors: pd.Series           # GICS sector per ticker
    companies: pd.Series         # company name per ticker


def load_market_data() -> MarketData:
    """Everything the index needs, with index tickers as columns (BRK-B -> BRK.B; GOOG dropped)."""
    universe = idx.load_universe()
    to_ticker = dict(zip(universe["yf_symbol"], universe["ticker"]))
    close, adj = idx.load_prices()
    close, adj = close.rename(columns=to_ticker).drop(columns="GOOG"), adj.rename(columns=to_ticker).drop(columns="GOOG")

    clean = idx.backfill_shares(pd.read_csv(idx.SHARES_CLEAN_CSV, parse_dates=["date"]))   # META, BKNG
    clean["ticker"] = clean["symbol"].map(to_ticker)
    shares = shares_as_of(clean[clean["ticker"] != "GOOG"], close.index)

    info = pd.read_csv(idx.INFO_CSV)
    info["ticker"] = info["symbol"].map(to_ticker)
    factors = idx.float_factors(info).set_index("ticker")["factor"].drop("GOOG")

    u = universe[universe["ticker"] != "GOOG"].set_index("ticker")
    return MarketData(close, adj, shares.reindex(columns=close.columns), factors, u["gics_sector"], u["company"])


def shares_as_of(history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Dates x tickers: the last reported (split-adjusted) share count on or before each date."""
    wide = history.pivot_table(index="date", columns="ticker", values="shares_adj", aggfunc="last")
    all_dates = wide.index.union(dates)
    return wide.reindex(all_dates).ffill().reindex(dates)


def ffmc(data: MarketData) -> pd.DataFrame:
    """Free-float market cap, dates x tickers (NaN until a company has both a price and a share count)."""
    return data.close * data.shares * data.float_factor.reindex(data.close.columns)


def rank_universe(ffmc_on_day: pd.Series) -> pd.Series:
    """Rank 1 = largest FFMC; companies without an FFMC are not ranked."""
    return ffmc_on_day.dropna().rank(ascending=False, method="first").astype(int)


def select_constituents(ranks: pd.Series, current=None, n: int = N_CONSTITUENTS,
                        entry: int = ENTRY_RANK, exit_: int = EXIT_RANK) -> list:
    """Exact-n selection with the entry/exit buffer (see the module docstring). Returns tickers by rank."""
    if not current:
        return list(ranks.nsmallest(n).index)
    current = set(current)
    holdovers = [t for t in current if t in ranks.index and ranks[t] <= exit_]
    entrants = [t for t in ranks.index if t not in current and ranks[t] <= entry]
    members = sorted(holdovers + entrants, key=lambda t: ranks[t])
    while len(members) > n:                                   # drop the lowest-ranked holdover
        worst_holdover = max((t for t in members if t in current), key=lambda t: ranks[t])
        members.remove(worst_holdover)
    if len(members) < n:                                      # fill from the best-ranked non-members
        fill = [t for t in ranks.sort_values().index if t not in members][: n - len(members)]
        members = sorted(members + fill, key=lambda t: ranks[t])
    return members


def review_dates(calendar: pd.DatetimeIndex, first: str = FIRST_REVIEW, last: str = LAST_DAY) -> pd.DatetimeIndex:
    """Quarter-end trading days from the first review on (a review on the last day itself is skipped)."""
    reviews = idx.quarter_end_reviews(calendar, start=first, end=last)
    return reviews[reviews < pd.Timestamp(last)]


def effective_date(review: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """The next trading day after a review: membership changes are traded at its close."""
    return calendar[calendar.searchsorted(review, side="right")]


def run_reviews(ffmc_table: pd.DataFrame, reviews: pd.DatetimeIndex) -> pd.DataFrame:
    """Constituents at every review, with ranks, additions and deletions."""
    rows, current = [], None
    for review in reviews:
        ranks = rank_universe(ffmc_table.loc[review])
        members = select_constituents(ranks, current)
        added = sorted(set(members) - set(current or []))
        removed = sorted(set(current or []) - set(members))
        rows.append({"review": review, "effective": effective_date(review, ffmc_table.index), "members": members,
                     "added": added if current else [], "removed": removed,
                     "ranks": {t: int(ranks[t]) for t in members}, "ranked_companies": len(ranks),
                     "removed_ranks": {t: (int(ranks[t]) if t in ranks.index else None) for t in removed}})
        current = members
    return pd.DataFrame(rows)
