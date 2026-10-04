"""
Labelling pipeline: give each tweet / news headline about a pool company a sentiment
and impact label from how the stock moved afterwards (market-reaction labelling,
rules in DECISIONS.md).

Run:  python src/labels.py     (prints the report, saves docs/labels_report.txt)

Steps per source:
  1. Load, keep pool companies, drop duplicate text+ticker (keep the earliest).
  2. Anchor each item to a trading-day close:
       date-only items -> last close strictly before the item's date (weekend -> Friday)
       timed news      -> last close at or before the timestamp (close = 16:00 ET)
     News counts as timed only if its timestamps pass the spot-check against
     raw_analyst_ratings.csv; otherwise it is treated as date-only.
  3. Returns r1, r2, r3, r7 = price k trading days after the anchor / anchor price - 1.
  4. Volatility filter: keep if the volatility of daily log returns over the 10 trading
     days after the anchor is higher than over the 10 days up to the anchor.
  5. Sentiment = sign of r1. Impact from the signs of (r1, r2, r3, r7), see IMPACT_TABLE.
     Items whose sign pattern is not in the table, or with a zero return, are dropped.

The tweet file's own return columns are NOT used for labels; they are only cross-checked.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prices as px  # noqa: E402
from inspect_data import NEWS_FILES, POOL, TWEET_FILES, Tee, indent, read_chunks, section  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"      # gitignored: outputs over the size limit
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
REPORT_PATH = PROJECT_ROOT / "docs" / "labels_report.txt"
MAX_TRACKED_MB = 50
SAMPLE_ROWS = 1000
SEED = 0

HORIZONS = [1, 2, 3, 7]           # trading days
VOL_WINDOW = 10                   # trading days before / after the anchor
CLOSE_HOUR = 16                   # US market close, Eastern time
TIMESTAMP_AGREEMENT_MIN = 0.95    # share of spot-checked news times that must agree within 1 minute

# Sign pattern of (r1, r2, r3, r7) -> impact (1 = bearish ... 10 = bullish). Others are dropped.
IMPACT_TABLE = {
    "++++": 10,   # mentor
    "--++": 7,    # mirror extension
    "+++-": 6,    # mentor
    "---+": 5,    # mirror extension
    "++--": 4,    # mentor
    "----": 1,    # mentor
}

TWEET_NAME_TO_TICKER = {name: ticker for ticker, _, names, _ in POOL for name in names}
NEWS_SYMBOL_TO_TICKER = {sym: ticker for ticker, _, _, symbols in POOL for sym in symbols}
COMPANY = {ticker: company for ticker, company, _, _ in POOL}

OUTPUT_COLS = ["item_id", "source", "ticker", "company", "text", "item_time", "anchor_date",
               "r1", "r2", "r3", "r7", "vol_before", "vol_after", "pattern", "sentiment", "impact"]


class StepLog:
    """Rows remaining after each pipeline step, per source (for the report)."""
    def __init__(self):
        self.rows = []

    def add(self, source: str, step: str, n: int):
        self.rows.append({"source": source, "step": step, "rows": n})

    def table(self, source: str) -> pd.DataFrame:
        df = pd.DataFrame([r for r in self.rows if r["source"] == source])
        df["dropped"] = (-df["rows"].diff()).fillna(0).astype(int)
        return df[["step", "rows", "dropped"]]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def drop_duplicate_texts(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Drop repeated text+ticker, keeping the earliest item (ties: earliest file row)."""
    ordered = df.sort_values([time_col, "row_id"], kind="stable")
    return ordered.drop_duplicates(subset=["text", "ticker"], keep="first").sort_index()


def load_tweets(log: StepLog) -> pd.DataFrame:
    keep = ["Unnamed: 0", "TWEET", "STOCK", "DATE", "1_DAY_RETURN", "2_DAY_RETURN", "3_DAY_RETURN", "7_DAY_RETURN"]
    frames, n_raw = [], 0
    for chunk in read_chunks(TWEET_FILES["tweets_full"]):
        n_raw += len(chunk)
        chunk = chunk[keep].rename(columns={"Unnamed: 0": "row_id", "TWEET": "text", "STOCK": "company_name",
                                            "DATE": "item_time", "1_DAY_RETURN": "file_r1", "2_DAY_RETURN": "file_r2",
                                            "3_DAY_RETURN": "file_r3", "7_DAY_RETURN": "file_r7"})
        chunk["ticker"] = chunk["company_name"].map(TWEET_NAME_TO_TICKER)
        frames.append(chunk[chunk["ticker"].notna()])
    log.add("tweets", "raw rows (full_dataset-release.csv)", n_raw)
    df = pd.concat(frames, ignore_index=True)
    df["row_id"] = df["row_id"].astype(int)
    df["item_date"] = pd.to_datetime(df["item_time"], format="%d/%m/%Y")
    for k in HORIZONS:
        df[f"file_r{k}"] = pd.to_numeric(df[f"file_r{k}"], errors="coerce")
    log.add("tweets", "pool companies (mapped name -> ticker)", len(df))
    df = df[df["text"].notna()]
    df = drop_duplicate_texts(df, "item_date")
    log.add("tweets", "drop duplicate text+ticker (keep earliest)", len(df))
    df["item_id"] = "tw_" + df["row_id"].astype(str)
    df["source"] = "tweets"
    return df


def load_news(log: StepLog) -> pd.DataFrame:
    frames, n_raw = [], 0
    for chunk in read_chunks(NEWS_FILES["analyst_ratings_processed"]):
        n_raw += len(chunk)
        chunk = chunk.rename(columns={"Unnamed: 0": "row_id", "title": "text", "date": "item_time", "stock": "symbol"})
        chunk["ticker"] = chunk["symbol"].map(NEWS_SYMBOL_TO_TICKER)
        frames.append(chunk[chunk["ticker"].notna()])
    log.add("news", "raw rows (analyst_ratings_processed.csv)", n_raw)
    df = pd.concat(frames, ignore_index=True)
    df["row_id"] = df["row_id"].astype(int)
    # Local (US Eastern) time; the UTC offset suffix (-04:00/-05:00) is dropped.
    df["local_time"] = pd.to_datetime(df["item_time"].str[:19], format="%Y-%m-%d %H:%M:%S")
    df["item_date"] = df["local_time"].dt.normalize()
    log.add("news", "pool tickers (FB -> META, GOOG -> GOOGL)", len(df))
    df = df[df["text"].notna()]
    df = drop_duplicate_texts(df, "local_time")
    log.add("news", "drop duplicate headline+ticker (keep earliest)", len(df))
    df["item_id"] = "nw_" + df["row_id"].astype(str)
    df["source"] = "news"
    return df


# ---------------------------------------------------------------------------
# News timestamp spot-check
# ---------------------------------------------------------------------------

def compare_timestamps(raw: pd.DataFrame, processed: pd.DataFrame) -> dict:
    """
    Compare times for the same headline+ticker. Both frames need columns key, time (naive local).
    If a key occurs several times in `processed`, the closest time is used.
    """
    merged = raw.merge(processed, on="key", suffixes=("_raw", "_proc"))
    merged["diff"] = merged["time_proc"] - merged["time_raw"]
    merged["abs_diff"] = merged["diff"].abs()
    best = merged.sort_values("abs_diff").drop_duplicates(subset=["raw_id"], keep="first")
    within = lambda limit: float((best["abs_diff"] <= limit).mean()) if len(best) else float("nan")  # noqa: E731
    minutes = (best["diff"].dt.total_seconds() / 60).round().astype("Int64")
    return {
        "raw_rows": len(raw),
        "matched": len(best),
        "exact_minute_floor": float((best["time_proc"] == best["time_raw"].dt.floor("min")).mean()) if len(best) else np.nan,
        "exact_minute_round": float((best["time_proc"] == best["time_raw"].dt.round("min")).mean()) if len(best) else np.nan,
        "within_1min": within(pd.Timedelta(minutes=1)),
        "within_1h": within(pd.Timedelta(hours=1)),
        "same_date": float((best["time_proc"].dt.normalize() == best["time_raw"].dt.normalize()).mean()) if len(best) else np.nan,
        "median_abs_diff": best["abs_diff"].median(),
        "top_diffs_min": {int(k): int(v) for k, v in minutes.value_counts().head(5).items()},
    }


def spot_check_news_times() -> dict:
    """Processed vs raw_analyst_ratings, for raw rows that carry a real time (UTC offset present)."""
    raw_parts = []
    for chunk in read_chunks(NEWS_FILES["raw_analyst_ratings"]):
        timed = chunk[chunk["date"].str.match(r".*[+-]\d{2}:\d{2}$", na=False)]
        raw_parts.append(pd.DataFrame({
            "raw_id": timed["Unnamed: 0"], "key": timed["headline"] + "\x1f" + timed["stock"],
            "time": pd.to_datetime(timed["date"].str[:19], format="%Y-%m-%d %H:%M:%S"),
            "offset": timed["date"].str[-6:]}))
    raw = pd.concat(raw_parts, ignore_index=True)
    keys = set(raw["key"])
    proc_parts = []
    for chunk in read_chunks(NEWS_FILES["analyst_ratings_processed"]):
        key = chunk["title"] + "\x1f" + chunk["stock"]
        hit = chunk[key.isin(keys)]
        proc_parts.append(pd.DataFrame({
            "key": key[key.isin(keys)],
            "time": pd.to_datetime(hit["date"].str[:19], format="%Y-%m-%d %H:%M:%S")}))
    processed = pd.concat(proc_parts, ignore_index=True)
    result = compare_timestamps(raw[["raw_id", "key", "time"]], processed)
    result["raw_date_range"] = (raw["time"].min(), raw["time"].max())
    result["raw_offsets"] = {k: int(v) for k, v in raw["offset"].value_counts().items()}
    result["reliable"] = bool(result["within_1min"] >= TIMESTAMP_AGREEMENT_MIN)
    return result


def time_of_day_by_year(news: pd.DataFrame) -> pd.DataFrame:
    """Plausibility of news times in years the spot-check cannot cover."""
    t = news["local_time"]
    minutes = t.dt.hour * 60 + t.dt.minute
    hhmm = t.dt.strftime("%H:%M")
    rows = []
    for year, idx in news.groupby(t.dt.year).groups.items():
        m, h = minutes[idx], hhmm[idx]
        rows.append({"year": year, "rows": len(idx),
                     "% before 09:30": round(100 * (m < 570).mean(), 1),
                     "% 09:30-16:00": round(100 * ((m >= 570) & (m < 960)).mean(), 1),
                     "% from 16:00": round(100 * (m >= 960).mean(), 1),
                     "most common time": h.value_counts().index[0],
                     "its share %": round(100 * h.value_counts().iloc[0] / len(idx), 1),
                     "distinct times": h.nunique()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anchors, returns, volatility
# ---------------------------------------------------------------------------

def anchor_date_only(item_dates: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Position in `calendar` of the last trading day strictly before each date (-1 if none)."""
    return calendar.searchsorted(item_dates.to_numpy(), side="left") - 1


def anchor_timed(local_times: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Position of the last close at or before each timestamp; a trading day closes at 16:00 ET."""
    days = local_times.dt.normalize().to_numpy()
    last_on_or_before = calendar.searchsorted(days, side="right") - 1   # counts that day's close
    last_before = calendar.searchsorted(days, side="left") - 1          # previous close
    return np.where(local_times.dt.hour.to_numpy() >= CLOSE_HOUR, last_on_or_before, last_before)


def market_features(tickers: pd.Series, anchor_pos: np.ndarray, prices: pd.DataFrame) -> pd.DataFrame:
    """Returns over HORIZONS and volatility before/after the anchor, for each item."""
    p = prices.to_numpy()
    n_days = len(prices)
    col = prices.columns.get_indexer(tickers)            # -1 if we have no prices for the ticker
    log_ret = np.log(prices).diff()
    # rolling std at row i covers daily log returns i-9 .. i (needs prices i-10 .. i)
    vol = log_ret.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().to_numpy()

    def at(matrix, offset):
        rows = anchor_pos + offset
        ok = (col >= 0) & (anchor_pos >= 0) & (rows >= 0) & (rows < n_days)
        out = np.full(len(anchor_pos), np.nan)
        out[ok] = matrix[rows[ok], col[ok]]
        return out

    anchor_price = at(p, 0)
    out = pd.DataFrame(index=tickers.index)
    valid = anchor_pos >= 0
    anchor_dates = np.full(len(anchor_pos), np.datetime64("NaT"), dtype="datetime64[ns]")
    anchor_dates[valid] = prices.index.to_numpy(dtype="datetime64[ns]")[anchor_pos[valid]]
    out["anchor_date"] = anchor_dates
    for k in HORIZONS:
        out[f"r{k}"] = at(p, k) / anchor_price - 1
    out["vol_before"] = at(vol, 0)              # returns over the 10 days ending at the anchor
    out["vol_after"] = at(vol, VOL_WINDOW)      # returns over the 10 days after the anchor
    return out


def sign_pattern(returns: pd.DataFrame) -> pd.Series:
    """'+'/'-' per horizon, e.g. '++-+'. Missing if any return is zero or missing."""
    signs = np.sign(returns[[f"r{k}" for k in HORIZONS]])
    ok = (signs != 0).all(axis=1) & signs.notna().all(axis=1)
    pattern = signs.replace({1.0: "+", -1.0: "-"}).astype(str).agg("".join, axis=1)
    return pattern.where(ok)


def apply_labels(df: pd.DataFrame, source: str, log: StepLog) -> pd.DataFrame:
    returns_cols = [f"r{k}" for k in HORIZONS]
    df = df[df[returns_cols + ["vol_before", "vol_after"]].notna().all(axis=1)]
    log.add(source, "anchor found + full price window available", len(df))
    df = df[df["vol_after"] > df["vol_before"]]
    log.add(source, "volatility filter (10d after > 10d before)", len(df))
    df = df.assign(pattern=sign_pattern(df))
    df = df[df["pattern"].notna()]
    log.add(source, "no zero return (sign defined for all horizons)", len(df))
    df = df[df["pattern"].isin(IMPACT_TABLE)].copy()
    log.add(source, "sign pattern in impact table (6 of 16)", len(df))
    df["sentiment"] = np.where(df["r1"] > 0, 1, -1)
    df["impact"] = df["pattern"].map(IMPACT_TABLE)
    df["company"] = df["ticker"].map(COMPANY)
    return df


# ---------------------------------------------------------------------------
# Cross-check of the tweet file's return columns
# ---------------------------------------------------------------------------

def crosscheck_file_returns(tweets: pd.DataFrame, prices: pd.DataFrame, anchor_pos: np.ndarray) -> pd.DataFrame:
    """How often the file's k-day return equals ours, with the anchor shifted by -2..+2 trading days."""
    rows = []
    for shift in range(-2, 3):
        shifted = np.where(anchor_pos >= 0, anchor_pos + shift, -1)   # keep "no anchor" as -1
        feats = market_features(tweets["ticker"], shifted, prices)
        for k in HORIZONS:
            ours, theirs = feats[f"r{k}"], tweets[f"file_r{k}"]
            ok = ours.notna() & theirs.notna()
            rows.append({"anchor shift (trading days)": shift, "column": f"{k}_DAY_RETURN", "pairs": int(ok.sum()),
                         "match % (|diff| < 0.002)": round(100 * ((ours[ok] - theirs[ok]).abs() < 0.002).mean(), 1),
                         "correlation": round(ours[ok].corr(theirs[ok]), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output and report
# ---------------------------------------------------------------------------

def write_output(df: pd.DataFrame, source: str) -> str:
    """Write labels to data/processed/ if small enough to track, else to data/interim/ plus a sample."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    path = INTERIM_DIR / f"labels_{source}.csv"
    df[OUTPUT_COLS].to_csv(path, index=False, encoding="utf-8")
    size_mb = path.stat().st_size / 1e6
    if size_mb <= MAX_TRACKED_MB:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        final = path.replace(PROCESSED_DIR / path.name)
        return f"{final.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB)"
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    sample = SAMPLES_DIR / f"labels_{source}_sample.csv"
    df[OUTPUT_COLS].sample(min(SAMPLE_ROWS, len(df)), random_state=SEED).to_csv(sample, index=False, encoding="utf-8")
    return (f"{path.relative_to(PROJECT_ROOT)} ({size_mb:.1f} MB, gitignored) "
            f"+ sample {sample.relative_to(PROJECT_ROOT)}")


def label_counts(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    """Rows, sentiment counts and impact counts, overall or grouped by `by`."""
    groups = [(None, df)] if by is None else df.groupby(by)
    rows = []
    for key, g in groups:
        row = {} if by is None else {by: key}
        row.update({"rows": len(g), "sent +1": int((g["sentiment"] == 1).sum()), "sent -1": int((g["sentiment"] == -1).sum())})
        row.update({f"imp {v}": int((g["impact"] == v).sum()) for v in sorted(IMPACT_TABLE.values())})
        rows.append(row)
    return pd.DataFrame(rows)


def report_source(df: pd.DataFrame, source: str, log: StepLog, where: str):
    section(f"{source.upper()}: rows kept and dropped at each step")
    print(indent(log.table(source).to_string(index=False)))
    print(f"\nOutput: {where}")
    section(f"{source.upper()}: label counts")
    print(indent(label_counts(df, None).to_string(index=False)))
    print("\nPer company:")
    per_company = label_counts(df, "ticker")
    per_company.insert(1, "company", per_company["ticker"].map(COMPANY))
    print(indent(per_company.sort_values("rows", ascending=False).to_string(index=False)))
    print("\nPer year (of the item date):")
    print(indent(label_counts(df.assign(year=df["item_date"].dt.year), "year").to_string(index=False)))


def main() -> int:
    buffer = io.StringIO()
    log = StepLog()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Labelling report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/labels.py")
        print(f"Rules: horizons {HORIZONS} trading days; volatility window {VOL_WINDOW} trading days; "
              f"impact table {IMPACT_TABLE}")

        prices = px.load_adj_close([t for t, *_ in POOL])
        calendar = prices.index
        section("PRICES")
        print(f"{prices.shape[1]} tickers x {len(calendar):,} trading days "
              f"({calendar.min().date()} -> {calendar.max().date()}), from {px.CACHE_PATH.relative_to(PROJECT_ROOT)}")
        summary = px.summarize(prices)
        print(f"Tickers with prices starting after 2009-01-31: "
              f"{ {t: str(d.date()) for t, d in summary['first'].items() if d > pd.Timestamp('2009-01-31')} }")
        missing = sorted(set(t for t, *_ in POOL) - set(prices.columns[prices.notna().any()]))
        print(f"Tickers without any prices: {missing or 'none'}")

        # --- News timestamps
        print("  spot-checking news timestamps ...", file=sys.__stdout__, flush=True)
        check = spot_check_news_times()
        section("NEWS TIMESTAMP SPOT-CHECK (analyst_ratings_processed vs raw_analyst_ratings rows with real times)")
        print(f"Raw rows with a real time: {check['raw_rows']:,} "
              f"({check['raw_date_range'][0]} -> {check['raw_date_range'][1]}), offsets {check['raw_offsets']}")
        print(f"Matched to a processed row (same headline + ticker): {check['matched']:,}")
        for name in ("same_date", "within_1h", "within_1min", "exact_minute_floor", "exact_minute_round"):
            print(f"  {name:20} {100 * check[name]:.1f}%")
        print(f"  median |difference|: {check['median_abs_diff']}")
        print(f"  most common differences (processed - raw, minutes): {check['top_diffs_min']}")
        print(f"Verdict: {'RELIABLE' if check['reliable'] else 'NOT reliable'} "
              f"(rule: at least {100 * TIMESTAMP_AGREEMENT_MIN:.0f}% within 1 minute) -> news treated as "
              f"{'timed (16:00 ET cut-off)' if check['reliable'] else 'date-only'}")

        # --- Tweets
        print("  labelling tweets ...", file=sys.__stdout__, flush=True)
        tweets = load_tweets(log)
        tweet_anchor = anchor_date_only(tweets["item_date"], calendar)
        tweets = tweets.join(market_features(tweets["ticker"], tweet_anchor, prices))
        crosscheck = crosscheck_file_returns(tweets, prices, tweet_anchor)
        tweet_labels = apply_labels(tweets, "tweets", log)
        tweet_where = write_output(tweet_labels, "tweets")

        # --- News
        print("  labelling news ...", file=sys.__stdout__, flush=True)
        news = load_news(log)
        news_anchor = (anchor_timed(news["local_time"], calendar) if check["reliable"]
                       else anchor_date_only(news["item_date"], calendar))
        news = news.join(market_features(news["ticker"], news_anchor, prices))
        news_labels = apply_labels(news, "news", log)
        news_where = write_output(news_labels, "news")

        section("NEWS: time of day by year (plausibility of times the spot-check cannot cover; pool tickers)")
        print(indent(time_of_day_by_year(news).to_string(index=False)))

        report_source(tweet_labels, "tweets", log, tweet_where)
        report_source(news_labels, "news", log, news_where)

        section("CROSS-CHECK: tweet file's return columns vs returns from yfinance adjusted closes")
        print("Shift 0 = our anchor (last close before the tweet date). Not used for labels.")
        with pd.option_context("display.width", 200):
            print(indent(crosscheck.sort_values(["column", "anchor shift (trading days)"]).to_string(index=False)))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
