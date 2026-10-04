"""
Inspection report for the raw text datasets (tweets + news). Read-only: builds no
labels or models and changes no data.

Run:   python src/inspect_data.py              (prints the report, saves docs/data_inspection.txt)
       python src/inspect_data.py --no-fetch   (skip the yfinance market-cap lookup)

Files (data/raw/, gitignored):
  tweets/full_dataset-release.csv, tweets/reduced_dataset-release.csv
      Kaggle "Tweet Sentiment's Impact on Stock Returns"
  news/raw_partner_headlines.csv, news/raw_analyst_ratings.csv, news/analyst_ratings_processed.csv
      Kaggle "Daily Financial News for 6000+ Stocks"

Parsing note: tweet texts contain bare '\r' characters, which pandas' default parser
treats as line breaks (that splits tweets into broken rows). We therefore split rows on
'\n' only, then replace any leftover '\r' (from CRLF line endings or inside text) by a space.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_PATH = PROJECT_ROOT / "docs" / "data_inspection.txt"
CHUNKSIZE = 200_000
SEED = 0

TWEET_FILES = {
    "tweets_full": RAW_DIR / "tweets" / "full_dataset-release.csv",
    "tweets_reduced": RAW_DIR / "tweets" / "reduced_dataset-release.csv",
}
NEWS_FILES = {
    "raw_partner_headlines": RAW_DIR / "news" / "raw_partner_headlines.csv",
    "raw_analyst_ratings": RAW_DIR / "news" / "raw_analyst_ratings.csv",
    "analyst_ratings_processed": RAW_DIR / "news" / "analyst_ratings_processed.csv",
}

# Candidate pool for the mock index: today's largest S&P 500 companies plus names that
# were among the largest in 2017-2020. The final 20 are chosen after this report.
# (current ticker, company, names used in the tweets' STOCK column, symbols used in the news data)
POOL = [
    ("AAPL", "Apple", ["Apple"], ["AAPL"]),
    ("MSFT", "Microsoft", ["Microsoft"], ["MSFT"]),
    ("NVDA", "NVIDIA", [], ["NVDA"]),
    ("AMZN", "Amazon", ["Amazon"], ["AMZN"]),
    ("GOOGL", "Alphabet", ["Google"], ["GOOGL", "GOOG"]),        # two share classes, one company
    ("META", "Meta Platforms", ["Facebook"], ["META", "FB"]),    # ticker FB until June 2022
    ("AVGO", "Broadcom", [], ["AVGO"]),
    ("TSLA", "Tesla", [], ["TSLA"]),
    ("BRK-B", "Berkshire Hathaway", [], ["BRK-B", "BRK.B", "BRK/B", "BRK-A", "BRK.A", "BRK/A"]),
    ("JPM", "JPMorgan Chase", ["JPMorgan"], ["JPM"]),
    ("WMT", "Walmart", ["Walmart"], ["WMT"]),
    ("LLY", "Eli Lilly", [], ["LLY"]),
    ("ORCL", "Oracle", ["Oracle"], ["ORCL"]),
    ("V", "Visa", ["Visa"], ["V"]),
    ("MA", "Mastercard", ["Mastercard"], ["MA"]),
    ("NFLX", "Netflix", ["Netflix"], ["NFLX"]),
    ("XOM", "Exxon Mobil", ["Exxon"], ["XOM"]),
    ("COST", "Costco", ["Costco"], ["COST"]),
    ("JNJ", "Johnson & Johnson", [], ["JNJ"]),
    ("HD", "Home Depot", ["Home Depot"], ["HD"]),
    ("PG", "Procter & Gamble", ["P&G"], ["PG"]),
    ("ABBV", "AbbVie", [], ["ABBV"]),
    ("BAC", "Bank of America", ["Bank of America"], ["BAC"]),
    ("PLTR", "Palantir", [], ["PLTR"]),
    ("AMD", "AMD", [], ["AMD"]),
    ("UNH", "UnitedHealth", [], ["UNH"]),
    ("KO", "Coca-Cola", ["CocaCola"], ["KO"]),
    ("CSCO", "Cisco", ["Cisco"], ["CSCO"]),
    ("CVX", "Chevron", ["Chevron"], ["CVX"]),
    ("WFC", "Wells Fargo", ["Wells Fargo"], ["WFC"]),
    ("CRM", "Salesforce", ["salesforce.com"], ["CRM"]),
    ("IBM", "IBM", ["IBM"], ["IBM"]),
    ("MRK", "Merck", [], ["MRK"]),
    ("PEP", "PepsiCo", ["Pepsi"], ["PEP"]),
    ("ADBE", "Adobe", ["Adobe"], ["ADBE"]),
    ("DIS", "Disney", ["Disney"], ["DIS"]),
    ("T", "AT&T", ["AT&T"], ["T"]),
    ("VZ", "Verizon", ["Verizon"], ["VZ"]),
    ("INTC", "Intel", ["Intel"], ["INTC"]),
    ("PFE", "Pfizer", ["Pfizer"], ["PFE"]),
    ("CMCSA", "Comcast", ["Comcast"], ["CMCSA"]),
    ("NKE", "Nike", ["Nike"], ["NKE"]),
    ("MCD", "McDonald's", ["McDonald's"], ["MCD"]),
    ("PYPL", "PayPal", ["PayPal"], ["PYPL"]),
    ("BA", "Boeing", ["Boeing"], ["BA"]),
    ("C", "Citigroup", ["Citigroup"], ["C"]),
    ("GE", "General Electric", ["General Electric"], ["GE"]),
]

# Recognised date layouts: name -> (regex, strptime format, characters to parse)
DATE_PATTERNS = {
    "DD/MM/YYYY": (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y", 10),
    "YYYY-MM-DD HH:MM:SS+HH:MM": (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S", 19),
    "YYYY-MM-DD HH:MM:SS": (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", "%Y-%m-%d %H:%M:%S", 19),
    "YYYY-MM-DD": (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d", 10),
}
TIME_PATTERNS = ["YYYY-MM-DD HH:MM:SS+HH:MM", "YYYY-MM-DD HH:MM:SS"]
OFFSET_RE = r"([+-]\d{2}:\d{2})$"

RETURN_COLS = ["1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
VOL_COLS = ["VOLATILITY_10D", "VOLATILITY_30D"]
TWEET_NUMERIC_COLS = ["LAST_PRICE", *RETURN_COLS, "PX_VOLUME", *VOL_COLS, "LSTM_POLARITY", "TEXTBLOB_POLARITY"]


# ---------------------------------------------------------------------------
# Reading and parsing helpers
# ---------------------------------------------------------------------------

def read_chunks(path: Path, chunksize: int = CHUNKSIZE):
    """Yield cleaned chunks of a CSV, all columns as text (see the parsing note at the top)."""
    for chunk in pd.read_csv(path, lineterminator="\n", chunksize=chunksize, dtype=str):
        chunk.columns = [c.replace("\r", "").strip() for c in chunk.columns]
        for col in chunk.columns:
            s = chunk[col].str.replace("\r", " ", regex=False).str.strip()
            chunk[col] = s.mask(s == "")
        yield chunk


def find_col(columns, *names):
    """Return the first column whose lower-cased name is in `names`, else None."""
    lower = {c.lower(): c for c in columns}
    return next((lower[n] for n in names if n in lower), None)


def classify_dates(dates: pd.Series) -> pd.Series:
    """Label each date string with the name of the DATE_PATTERNS layout it matches ('other' if none)."""
    labels = pd.Series("other", index=dates.index, dtype=object)
    for name, (regex, _, _) in DATE_PATTERNS.items():
        labels[dates.str.match(regex, na=False)] = name
    return labels


def parse_local_dates(dates: pd.Series, labels: pd.Series) -> pd.Series:
    """Parse dates as local (exchange) time, ignoring any UTC offset suffix."""
    parts = [pd.to_datetime(dates[labels == name].str[:width], format=fmt, errors="coerce")
             for name, (_, fmt, width) in DATE_PATTERNS.items() if (labels == name).any()]
    if not parts:
        return pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    return pd.concat(parts).reindex(dates.index)


def hash_strings(s: pd.Series) -> np.ndarray:
    """64-bit hash per string: lets us count duplicates without keeping the texts in memory."""
    return pd.util.hash_pandas_object(s, index=False).to_numpy()


def looks_like_fractions(values: pd.Series) -> bool:
    """Daily returns as fractions have a typical size around 0.01; as percents around 1."""
    return float(values.abs().median()) < 0.2


def pool_count(counter: Counter, names) -> int:
    """Total rows across all aliases (e.g. FB + META) of one company."""
    return sum(counter.get(n, 0) for n in names)


# ---------------------------------------------------------------------------
# Per-file statistics, accumulated chunk by chunk
# ---------------------------------------------------------------------------

class FileStats:
    def __init__(self, name: str, path: Path, keep_cols=None, window=None):
        self.name, self.path = name, path
        self.keep_cols = keep_cols or []
        self.window = window                     # (first, last) date: count rows per ticker inside it
        self.rows = 0
        self.columns = None
        self.missing, self.numeric, self.non_numeric, self.non_int = Counter(), Counter(), Counter(), Counter()
        self.date_patterns, self.offsets = Counter(), Counter()
        self.other_dates = []
        self.date_min = self.date_max = None
        self.with_time = self.midnight = self.seconds_zero = self.weekend = 0
        self.dayfirst_max = [0, 0]               # max of the 1st and 2nd field in DD/MM/YYYY dates
        self.tickers, self.tickers_in_window, self.publishers = Counter(), Counter(), Counter()
        self.text_hashes, self.text_ticker_hashes, self.key_hashes, self.has_at = [], [], [], []
        self.examples, self.kept = [], []

    def update(self, chunk: pd.DataFrame, chunk_no: int):
        if self.columns is None:
            self.columns = list(chunk.columns)
            self.text_col = find_col(self.columns, "tweet", "headline", "title")
            self.ticker_col = find_col(self.columns, "stock", "ticker")
            self.date_col = find_col(self.columns, "date")
            self.publisher_col = find_col(self.columns, "publisher")
        self.rows += len(chunk)

        # Missing values and inferred types (everything is read as text first)
        for col in chunk.columns:
            s = chunk[col]
            num = pd.to_numeric(s, errors="coerce")
            self.missing[col] += int(s.isna().sum())
            self.numeric[col] += int(num.notna().sum())
            self.non_numeric[col] += int((s.notna() & num.isna()).sum())
            # Written with a decimal point or exponent ("100.0", "1e-3") -> float, even if whole-valued
            self.non_int[col] += int((num.notna() & s.str.contains(r"[.eE]", na=False)).sum())

        # Dates
        d = chunk[self.date_col]
        labels = classify_dates(d)
        self.date_patterns.update(labels[d.notna()].value_counts().to_dict())
        if len(self.other_dates) < 5:
            self.other_dates += d[(labels == "other") & d.notna()].head(5).tolist()
        self.offsets.update(d.str.extract(OFFSET_RE)[0].value_counts().to_dict())
        local = parse_local_dates(d, labels)
        if local.notna().any():
            lo, hi = local.min(), local.max()
            self.date_min = lo if self.date_min is None else min(self.date_min, lo)
            self.date_max = hi if self.date_max is None else max(self.date_max, hi)
        has_time = labels.isin(TIME_PATTERNS)
        self.with_time += int(has_time.sum())
        self.midnight += int((has_time & (d.str[11:19] == "00:00:00")).sum())
        self.seconds_zero += int((has_time & (d.str[17:19] == "00")).sum())
        self.weekend += int((local.dt.dayofweek >= 5).sum())
        dm = labels == "DD/MM/YYYY"
        if dm.any():
            self.dayfirst_max[0] = max(self.dayfirst_max[0], int(d[dm].str[:2].astype(int).max()))
            self.dayfirst_max[1] = max(self.dayfirst_max[1], int(d[dm].str[3:5].astype(int).max()))

        # Tickers / companies
        tick = chunk[self.ticker_col]
        self.tickers.update(tick.value_counts().to_dict())
        if self.window:
            inside = local.dt.normalize().between(*self.window)
            self.tickers_in_window.update(tick[inside].value_counts().to_dict())
        if self.publisher_col:
            self.publishers.update(chunk[self.publisher_col].value_counts().to_dict())

        # Hashes for duplicates and cross-file overlap
        text = chunk[self.text_col]
        self.text_hashes.append(hash_strings(text.dropna()))
        text, tick, d = text.fillna(""), tick.fillna(""), d.fillna("")
        self.text_ticker_hashes.append(hash_strings(text + "\x1f" + tick))
        self.key_hashes.append(hash_strings(text + "\x1f" + tick + "\x1f" + d))
        self.has_at.append(text.str.contains("@", regex=False).to_numpy())

        self.examples.append(chunk.sample(min(5, len(chunk)), random_state=SEED + chunk_no))
        if self.keep_cols:
            self.kept.append(chunk[self.keep_cols])

    def finish(self):
        for attr in ("text_hashes", "text_ticker_hashes", "key_hashes", "has_at"):
            setattr(self, attr, np.concatenate(getattr(self, attr)))
        examples = pd.concat(self.examples)
        self.examples = examples.sample(min(5, len(examples)), random_state=SEED)
        self.kept = pd.concat(self.kept, ignore_index=True) if self.keep_cols else None
        return self

    def dtype_label(self, col) -> str:
        filled = self.rows - self.missing[col]
        if filled == 0 or self.numeric[col] / filled < 0.5:
            return "text"
        kind = "int" if self.non_int[col] == 0 else "float"
        return kind if self.non_numeric[col] == 0 else f"{kind} ({self.non_numeric[col]:,} non-numeric)"

    def report(self):
        size_mb = self.path.stat().st_size / 1e6
        section(f"FILE: {self.name}  ({self.path.relative_to(PROJECT_ROOT)}, {size_mb:,.0f} MB)")
        print(f"Rows: {self.rows:,}")
        print("\nColumns:")
        table = pd.DataFrame({
            "dtype": [self.dtype_label(c) for c in self.columns],
            "% missing": [round(100 * self.missing[c] / self.rows, 2) for c in self.columns],
        }, index=self.columns)
        print(indent(table.to_string()))

        print(f"\nDates (column '{self.date_col}'):")
        print(f"  formats:        {dict(self.date_patterns)}")
        if self.other_dates:
            print(f"  unparsed examples: {self.other_dates[:5]}")
        if self.dayfirst_max[0]:
            print(f"  day-first check: max of 1st field = {self.dayfirst_max[0]}, of 2nd field = "
                  f"{self.dayfirst_max[1]}  (1st > 12 confirms DD/MM)")
        print(f"  timezone:       {('UTC offsets ' + str(dict(self.offsets))) if self.offsets else 'none (naive dates)'}")
        print(f"  range (local):  {self.date_min} -> {self.date_max}")
        if self.with_time:
            print(f"  time of day:    {self.with_time:,} rows have a time; "
                  f"{100 * self.midnight / self.with_time:.1f}% are exactly 00:00:00, "
                  f"{100 * self.seconds_zero / self.with_time:.1f}% have seconds == 00")
        print(f"  weekend dates:  {100 * self.weekend / self.rows:.1f}% of rows")

        print(f"\nTickers/companies (column '{self.ticker_col}'): {len(self.tickers):,} unique")
        print(f"  top 10: {dict(self.tickers.most_common(10))}")
        if self.publishers:
            print(f"\nPublishers: {len(self.publishers):,} unique; top 10: {dict(self.publishers.most_common(10))}")

        n_text = len(self.text_hashes)
        dup_text = n_text - len(np.unique(self.text_hashes))
        dup_pair = self.rows - len(np.unique(self.text_ticker_hashes))
        dup_key = self.rows - len(np.unique(self.key_hashes))
        print(f"\nDuplicate texts (column '{self.text_col}'):")
        print(f"  same text:                 {dup_text:,} rows ({100 * dup_text / max(n_text, 1):.1f}%) repeat an earlier text")
        print(f"  same text + ticker:        {dup_pair:,} rows ({100 * dup_pair / self.rows:.1f}%)")
        print(f"  same text + ticker + date: {dup_key:,} rows ({100 * dup_key / self.rows:.1f}%)")

        print("\n5 example rows (random, seed 0):")
        with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 70):
            print(indent(self.examples.to_string(index=False)))


def inspect_file(name, path, chunksize, keep_cols=None, window=None) -> FileStats:
    print(f"  reading {name} ...", file=sys.__stdout__, flush=True)
    stats = FileStats(name, path, keep_cols=keep_cols, window=window)
    for i, chunk in enumerate(read_chunks(path, chunksize)):
        stats.update(chunk, i)
    return stats.finish()


# ---------------------------------------------------------------------------
# Tweets: full vs reduced, returns, volatility
# ---------------------------------------------------------------------------

def to_numeric_frame(kept: pd.DataFrame) -> pd.DataFrame:
    out = kept.copy()
    for col in TWEET_NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["DATE"] = pd.to_datetime(out["DATE"], format="%d/%m/%Y", errors="coerce")
    return out


def daily_series(tweets: pd.DataFrame):
    """One row per (stock, date). Price/return/vol columns repeat on every tweet of that day."""
    cols = ["LAST_PRICE", *RETURN_COLS, *VOL_COLS]
    grouped = tweets.groupby(["STOCK", "DATE"])
    not_unique = {c: float((grouped[c].nunique() > 1).mean()) for c in cols}
    daily = grouped[cols].first().reset_index().sort_values(["STOCK", "DATE"])
    # Position of each date in the Mon-Fri calendar (exchange holidays not removed), so a gap
    # between two tweet dates shows up as a jump > 1. Weekend dates are dropped: they would
    # otherwise sit between Friday and Monday and break that link.
    calendar = pd.bdate_range(daily["DATE"].min(), daily["DATE"].max())
    daily["t"] = daily["DATE"].map({d: i for i, d in enumerate(calendar)})
    return daily.dropna(subset=["t"]).reset_index(drop=True), not_unique


def return_alignment(daily: pd.DataFrame, col: str, max_k: int = 7) -> pd.DataFrame:
    """How often `col` equals the price change over k trading days, forward (t -> t+k) or backward (t-k -> t)."""
    rows = []
    groups = [g for _, g in daily.groupby("STOCK")]
    for direction in ("forward", "backward"):
        for k in range(1, max_k + 1):
            n = matched = 0
            for g in groups:
                p, t, reported = g["LAST_PRICE"], g["t"], g[col]
                if direction == "forward":
                    implied, ok = p.shift(-k) / p - 1, (t.shift(-k) - t) == k
                else:
                    implied, ok = p / p.shift(k) - 1, (t - t.shift(k)) == k
                valid = ok & reported.notna() & implied.notna()
                n += int(valid.sum())
                matched += int(np.isclose(reported[valid], implied[valid], rtol=1e-3, atol=1e-4).sum())
            rows.append({"column": col, "direction": direction, "k (trading days)": k,
                         "pairs": n, "match %": round(100 * matched / n, 1) if n else np.nan})
    return pd.DataFrame(rows)


def volatility_check(daily: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    """Compare `col` with annualised realised volatility (%) over three possible windows."""
    pairs = {"trailing, ending on tweet date": [], "trailing, ending the day before": [], "forward, after tweet date": []}
    for _, g in daily.groupby("STOCK"):
        p, t, reported = g["LAST_PRICE"], g["t"], g[col]
        consecutive = t.diff() == 1
        log_ret = np.log(p).diff().where(consecutive)
        vol = log_ret.rolling(window).std() * np.sqrt(252) * 100   # NaN unless all `window` returns exist
        candidates = {
            "trailing, ending on tweet date": vol,
            "trailing, ending the day before": vol.shift(1).where(consecutive),
            "forward, after tweet date": vol.shift(-window).where((t.shift(-window) - t) == window),
        }
        for name, computed in candidates.items():
            valid = computed.notna() & reported.notna() & t.notna()
            pairs[name].append(pd.DataFrame({"reported": reported[valid], "computed": computed[valid]}))
    rows = []
    for name, frames in pairs.items():
        df = pd.concat(frames)
        rows.append({"column": col, "window tested": name, "pairs": len(df),
                     "correlation": round(df["reported"].corr(df["computed"]), 3),
                     "median |diff|": round((df["reported"] - df["computed"]).abs().median(), 2),
                     "median reported": round(df["reported"].median(), 2),
                     "median computed": round(df["computed"].median(), 2)})
    return pd.DataFrame(rows)


def report_tweets(full: FileStats, reduced: FileStats):
    section("TWEETS: full vs reduced")
    print(f"Rows: full {full.rows:,}, reduced {reduced.rows:,} ({100 * reduced.rows / full.rows:.1f}% of full)")
    print(f"Columns only in full:    {sorted(set(full.columns) - set(reduced.columns))}")
    print(f"Columns only in reduced: {sorted(set(reduced.columns) - set(full.columns))}")
    print(f"Date range: full {full.date_min.date()} -> {full.date_max.date()}, "
          f"reduced {reduced.date_min.date()} -> {reduced.date_max.date()}")
    print(f"Companies: full {len(full.tickers)}, reduced {len(reduced.tickers)}; "
          f"only in full: {sorted(set(full.tickers) - set(reduced.tickers))}")

    in_full = np.isin(reduced.key_hashes, full.key_hashes)
    print(f"Reduced rows found in full (same tweet + company + date): {100 * in_full.mean():.1f}%")
    full_only = ~np.isin(full.key_hashes, reduced.key_hashes)
    print(f"Tweets containing '@': reduced {100 * reduced.has_at.mean():.1f}%, "
          f"full rows not in reduced {100 * full.has_at[full_only].mean():.1f}%")
    r = reduced.kept
    mention_filled = r["MENTION"].notna()
    in_text = [str(m).lower() in str(tw).lower() for tw, m in zip(r.loc[mention_filled, "TWEET"], r.loc[mention_filled, "MENTION"])]
    print(f"Reduced MENTION column: filled in {100 * mention_filled.mean():.1f}% of rows; "
          f"the mention appears in the tweet text in {100 * np.mean(in_text):.1f}% of those")
    print(f"  most common mentions: { {k: int(v) for k, v in r['MENTION'].value_counts().head(10).items()} }")
    top = pd.DataFrame({"full": pd.Series(full.tickers), "reduced": pd.Series(reduced.tickers)}).fillna(0).astype(int)
    top["reduced/full %"] = (100 * top["reduced"] / top["full"]).round(1)
    print("\nRows per company (top 15 in full):")
    print(indent(top.sort_values("full", ascending=False).head(15).to_string()))

    for label, stats in (("full", full), ("reduced", reduced)):
        tw = to_numeric_frame(stats.kept)
        section(f"TWEETS ({label}): value ranges")
        desc = pd.DataFrame({
            "min": tw[TWEET_NUMERIC_COLS].min(), "p1": tw[TWEET_NUMERIC_COLS].quantile(0.01),
            "median": tw[TWEET_NUMERIC_COLS].median(), "p99": tw[TWEET_NUMERIC_COLS].quantile(0.99),
            "max": tw[TWEET_NUMERIC_COLS].max(), "median |x|": tw[TWEET_NUMERIC_COLS].abs().median(),
            "missing": tw[TWEET_NUMERIC_COLS].isna().sum(),
        })
        with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 200):
            print(indent(desc.to_string()))
        verdicts = {c: "fractions" if looks_like_fractions(tw[c].dropna()) else "percents" for c in RETURN_COLS}
        print(f"\nReturns look like: {verdicts}  (median |1-day return| = {tw['1_DAY_RETURN'].abs().median():.4f})")
        print(f"LSTM_POLARITY values: { {k: int(v) for k, v in tw['LSTM_POLARITY'].value_counts().items()} }")
        if label == "reduced":
            continue   # alignment and volatility checks use the full file (more price history)

        daily, not_unique = daily_series(tw)
        section("TWEETS (full): what the return and volatility columns measure")
        dates = pd.Series(sorted(tw["DATE"].dropna().unique()))
        n_weekdays = len(pd.bdate_range(dates.min(), dates.max()))
        print(f"Stock-days: {len(daily):,}. Distinct tweet dates: {len(dates)} "
              f"({int((dates.dt.dayofweek < 5).sum())} weekdays) out of {n_weekdays} weekdays in the date range.")
        per_month = dates.dt.strftime("%Y-%m").value_counts().sort_index()
        print(f"  dates per month: { {k: int(v) for k, v in per_month.items()} }")
        print("  The checks below compare only truly consecutive weekdays, so sparse dates mean few pairs.")
        print("Share of stock-days where a column takes more than one value across that day's tweets:")
        print(f"  { {c: round(100 * v, 2) for c, v in not_unique.items()} } (%)")

        print("\nReturn alignment: % of stock-days where the column equals the LAST_PRICE change over k trading days")
        align = pd.concat([return_alignment(daily, c) for c in RETURN_COLS])
        best = align.sort_values("match %", ascending=False).groupby("column", sort=False).head(2)
        print(indent(best.sort_values(["column", "match %"], ascending=[True, False]).to_string(index=False)))

        print("\nVolatility: reported vs realised volatility computed from LAST_PRICE "
              "(std of daily log returns x sqrt(252) x 100):")
        vol = pd.concat([volatility_check(daily, "VOLATILITY_10D", 10), volatility_check(daily, "VOLATILITY_30D", 30)])
        with pd.option_context("display.width", 200):
            print(indent(vol.to_string(index=False)))


# ---------------------------------------------------------------------------
# News: compare the three files
# ---------------------------------------------------------------------------

def report_news(news: dict):
    section("NEWS: comparison of the three files")
    rows = []
    for name, s in news.items():
        rows.append({
            "file": name, "rows": f"{s.rows:,}", "columns": ", ".join(c for c in s.columns if not c.startswith("Unnamed")),
            "date range": f"{s.date_min.date()} -> {s.date_max.date()}",
            "UTC offset": "yes" if s.offsets else "no",
            "time is 00:00:00 %": round(100 * s.midnight / s.with_time, 1) if s.with_time else "no time",
            "tickers": f"{len(s.tickers):,}", "publishers": f"{len(s.publishers):,}" if s.publishers else "-",
            "dup text+ticker %": round(100 * (s.rows - len(np.unique(s.text_ticker_hashes))) / s.rows, 1),
        })
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(indent(pd.DataFrame(rows).to_string(index=False)))

    def overlap(a, b):
        return 100 * np.isin(np.unique(a), np.unique(b)).mean()

    p, a, q = news["raw_partner_headlines"], news["raw_analyst_ratings"], news["analyst_ratings_processed"]
    print("\nOverlap (share of unique headline+ticker pairs in the first file also found in the second):")
    print(f"  analyst_ratings_processed in raw_analyst_ratings: {overlap(q.text_ticker_hashes, a.text_ticker_hashes):.1f}%")
    print(f"  raw_analyst_ratings in analyst_ratings_processed: {overlap(a.text_ticker_hashes, q.text_ticker_hashes):.1f}%")
    print(f"  raw_partner_headlines in raw_analyst_ratings:     {overlap(p.text_ticker_hashes, a.text_ticker_hashes):.1f}%")
    print(f"  (headline only) partner in analyst:               {overlap(p.text_hashes, a.text_hashes):.1f}%")


# ---------------------------------------------------------------------------
# Candidate pool: market caps and coverage
# ---------------------------------------------------------------------------

def fetch_market_caps(pool) -> pd.DataFrame:
    """Total and free-float market cap per company from yfinance (needs internet)."""
    import yfinance as yf
    rows = []
    for ticker, company, _, _ in pool:
        print(f"  fetching {ticker} ...", file=sys.__stdout__, flush=True)
        try:
            info = yf.Ticker(ticker).info
            error = ""
        except Exception as exc:  # noqa: BLE001 - report and continue with the next ticker
            info, error = {}, f"{type(exc).__name__}: {exc}"[:60]
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        rows.append({"ticker": ticker, "company": company, "price": price,
                     "market_cap": info.get("marketCap"), "shares_out": info.get("sharesOutstanding"),
                     "float_shares": info.get("floatShares"), "error": error})
    caps = pd.DataFrame(rows)
    for col in ("price", "market_cap", "shares_out", "float_shares"):
        caps[col] = pd.to_numeric(caps[col], errors="coerce")
    caps["ff_market_cap"] = caps["float_shares"] * caps["price"]
    caps["float/shares_out"] = (caps["float_shares"] / caps["shares_out"]).round(3)
    caps["rank_total"] = caps["market_cap"].rank(ascending=False, method="min").astype("Int64")
    caps["rank_ff"] = caps["ff_market_cap"].rank(ascending=False, method="min").astype("Int64")
    return caps


def report_market_caps(caps: pd.DataFrame):
    section(f"CANDIDATE POOL ({len(POOL)} companies): market caps from yfinance, fetched "
            f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC")
    print("Free-float market cap = floatShares x current price.")
    print("Caveats: for multi-class companies (GOOGL, BRK-B, META) yfinance's share counts may cover only one class,")
    print("so check rows where float/shares_out is far from ~0.8-1.0.")
    show = caps.copy()
    show["market_cap ($bn)"] = (show["market_cap"] / 1e9).round(1)
    show["ff_market_cap ($bn)"] = (show["ff_market_cap"] / 1e9).round(1)
    cols = ["rank_total", "rank_ff", "ticker", "company", "price", "market_cap ($bn)", "ff_market_cap ($bn)", "float/shares_out"]
    with pd.option_context("display.width", 200):
        print("\nRanking by total market cap:")
        print(indent(show.sort_values("market_cap", ascending=False)[cols].to_string(index=False)))
        print("\nRanking by free-float market cap:")
        print(indent(show.sort_values("ff_market_cap", ascending=False)[cols].to_string(index=False)))
    errors = caps[caps["error"] != ""]
    if len(errors):
        print(f"\nFetch errors: {dict(zip(errors['ticker'], errors['error']))}")


def report_pool_coverage(tweets: dict, news: dict, caps: pd.DataFrame | None, window):
    section("CANDIDATE POOL: tweet and news coverage per company")
    print(f"'in window' = rows dated within the tweets' date range ({window[0].date()} -> {window[1].date()}).")
    rows = []
    for ticker, company, tweet_names, news_symbols in POOL:
        row = {"ticker": ticker, "company": company,
               "tweets_full": pool_count(tweets["tweets_full"].tickers, tweet_names),
               "tweets_reduced": pool_count(tweets["tweets_reduced"].tickers, tweet_names)}
        for name, s in news.items():
            short = {"raw_partner_headlines": "partner", "raw_analyst_ratings": "analyst",
                     "analyst_ratings_processed": "processed"}[name]
            row[short] = pool_count(s.tickers, news_symbols)
            row[f"{short}_in_window"] = pool_count(s.tickers_in_window, news_symbols)
        rows.append(row)
    cov = pd.DataFrame(rows)
    if caps is not None:
        cov = caps[["ticker", "rank_total", "rank_ff"]].merge(cov, on="ticker").sort_values("rank_ff")
    with pd.option_context("display.width", 250):
        print(indent(cov.to_string(index=False)))

    print("\nSymbol aliases found in the news files (rows per symbol):")
    for ticker, company, _, news_symbols in POOL:
        if len(news_symbols) > 1:
            found = {name: {sym: s.tickers[sym] for sym in news_symbols if s.tickers.get(sym)}
                     for name, s in news.items()}
            print(f"  {ticker:6} {company:20} {found}")


# ---------------------------------------------------------------------------
# Output helpers and main
# ---------------------------------------------------------------------------

class Tee(io.TextIOBase):
    """Write to several streams at once (console + report buffer)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def section(title: str):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--no-fetch", action="store_true", help="skip the yfinance market-cap lookup")
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE)
    args = parser.parse_args(argv)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Raw data inspection report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/inspect_data.py")

        caps = None if args.no_fetch else fetch_market_caps(POOL)
        if caps is not None:
            report_market_caps(caps)

        keep_reduced = ["STOCK", "DATE", "TWEET", "MENTION", *TWEET_NUMERIC_COLS]
        keep_full = ["STOCK", "DATE", *TWEET_NUMERIC_COLS]
        tweets = {
            "tweets_full": inspect_file("tweets_full", TWEET_FILES["tweets_full"], args.chunksize, keep_full),
            "tweets_reduced": inspect_file("tweets_reduced", TWEET_FILES["tweets_reduced"], args.chunksize, keep_reduced),
        }
        window = (tweets["tweets_full"].date_min.normalize(), tweets["tweets_full"].date_max.normalize())
        news = {name: inspect_file(name, path, args.chunksize, window=window) for name, path in NEWS_FILES.items()}

        for stats in [*tweets.values(), *news.values()]:
            stats.report()
        report_tweets(tweets["tweets_full"], tweets["tweets_reduced"])
        report_news(news)
        report_pool_coverage(tweets, news, caps, window)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
