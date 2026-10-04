"""
Tests for the helper functions in src/inspect_data.py, using tiny made-up CSV files.
No network access and no real data needed.

Run:  python tests/test_inspect_data.py        (plain script)
 or:  pytest tests/test_inspect_data.py
"""
from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import inspect_data as ins  # noqa: E402

# A tweet whose text contains a bare '\r', in a file with CRLF line endings
# (the two quirks found in the real files).
TWEETS_CSV = (
    b",TWEET,STOCK,DATE,LAST_PRICE,1_DAY_RETURN\r\n"
    b'0,"Great quarter\rfor $AAPL",Apple,31/01/2017,100.0,0.01\r\n'
    b"1,Plain tweet,Nike,01/02/2017,50.0,-0.02\r\n"
    b"2,Plain tweet,Nike,13/02/2017,51.0,\r\n"
)


def _write_tmp(content: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_read_chunks_handles_bare_cr_and_crlf():
    path = _write_tmp(TWEETS_CSV)
    df = pd.concat(ins.read_chunks(path, chunksize=2))
    assert len(df) == 3                                    # the '\r' did not split the first tweet
    assert list(df.columns)[1:] == ["TWEET", "STOCK", "DATE", "LAST_PRICE", "1_DAY_RETURN"]
    assert df["TWEET"].iloc[0] == "Great quarter for $AAPL"
    assert df["1_DAY_RETURN"].iloc[1] == "-0.02"          # no trailing '\r' from CRLF
    assert pd.isna(df["1_DAY_RETURN"].iloc[2])            # empty field becomes missing


def test_classify_and_parse_dates():
    dates = pd.Series(["31/01/2017", "2020-06-05 10:30:54-04:00", "2020-06-01 00:00:00",
                       "2020-06-01", "June 5", None])
    labels = ins.classify_dates(dates)
    assert labels.tolist() == ["DD/MM/YYYY", "YYYY-MM-DD HH:MM:SS+HH:MM", "YYYY-MM-DD HH:MM:SS",
                               "YYYY-MM-DD", "other", "other"]
    local = ins.parse_local_dates(dates, labels)
    assert local.iloc[0] == pd.Timestamp("2017-01-31")
    assert local.iloc[1] == pd.Timestamp("2020-06-05 10:30:54")   # local time, offset ignored
    assert pd.isna(local.iloc[4]) and pd.isna(local.iloc[5])
    offsets = dates.str.extract(ins.OFFSET_RE)[0]
    assert offsets.iloc[1] == "-04:00" and pd.isna(offsets.iloc[0])


def test_find_col_and_pool_count():
    assert ins.find_col(["Unnamed: 0", "TWEET", "STOCK"], "tweet", "headline") == "TWEET"
    assert ins.find_col(["title", "date"], "tweet", "headline", "title") == "title"
    assert ins.find_col(["a"], "b") is None
    counts = Counter({"FB": 10, "META": 2, "AAPL": 5})
    assert ins.pool_count(counts, ["META", "FB"]) == 12
    assert ins.pool_count(counts, ["BRK-B", "BRK.B"]) == 0


def test_pool_is_well_formed():
    tickers = [p[0] for p in ins.POOL]
    assert len(tickers) == len(set(tickers))
    assert 40 <= len(ins.POOL) <= 50
    assert all(len(p) == 4 and p[3] for p in ins.POOL)   # every company has at least one news symbol


def test_looks_like_fractions():
    assert ins.looks_like_fractions(pd.Series([0.01, -0.02, 0.005]))
    assert not ins.looks_like_fractions(pd.Series([1.0, -2.0, 0.5]))


def test_file_stats_on_small_file():
    path = _write_tmp(TWEETS_CSV)
    stats = ins.inspect_file("tiny", path, chunksize=2, keep_cols=["STOCK", "DATE"])
    assert stats.rows == 3
    assert stats.tickers == Counter({"Nike": 2, "Apple": 1})
    assert stats.date_patterns == Counter({"DD/MM/YYYY": 3})
    assert stats.dayfirst_max == [31, 2]
    assert stats.dtype_label("LAST_PRICE") == "float"
    assert stats.dtype_label("TWEET") == "text"
    assert stats.missing["1_DAY_RETURN"] == 1
    assert len(stats.text_hashes) - len(np.unique(stats.text_hashes)) == 1   # "Plain tweet" twice


def _daily_from_prices(prices):
    dates = pd.bdate_range("2017-01-02", periods=len(prices))
    daily = pd.DataFrame({"STOCK": "X", "DATE": dates, "LAST_PRICE": prices})
    daily["t"] = range(len(prices))
    return daily


def test_daily_series_keeps_gaps_between_sparse_dates():
    # Tweets on Tue 31 Jan, Wed 1 Feb, Fri 3 Feb, Sat 4 Feb, Mon 6 Feb and Tue 28 Feb 2017.
    # The Saturday is dropped, so Friday -> Monday counts as consecutive; the 28th is a gap.
    dates = ["2017-01-31", "2017-02-01", "2017-02-03", "2017-02-04", "2017-02-06", "2017-02-28"]
    tweets = pd.DataFrame({"STOCK": "X", "DATE": pd.to_datetime(dates), "LAST_PRICE": 1.0})
    for col in [*ins.RETURN_COLS, *ins.VOL_COLS]:
        tweets[col] = 0.0
    daily, _ = ins.daily_series(tweets)
    assert daily["t"].diff().tolist()[1:] == [1, 2, 1, 16]


def test_return_alignment_finds_forward_one_day():
    prices = np.array([100, 101, 99, 102, 103, 101, 104, 105, 103, 106], dtype=float)
    daily = _daily_from_prices(prices)
    daily["1_DAY_RETURN"] = pd.Series(prices).shift(-1) / prices - 1   # defined as t -> t+1
    result = ins.return_alignment(daily, "1_DAY_RETURN", max_k=2)
    best = result.sort_values("match %", ascending=False).iloc[0]
    assert (best["direction"], best["k (trading days)"], best["match %"]) == ("forward", 1, 100.0)


def test_volatility_check_identifies_trailing_window():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 120)))
    daily = _daily_from_prices(prices)
    log_ret = np.log(daily["LAST_PRICE"]).diff()
    daily["VOLATILITY_10D"] = log_ret.rolling(10).std() * np.sqrt(252) * 100   # trailing, incl. today
    result = ins.volatility_check(daily, "VOLATILITY_10D", 10).set_index("window tested")
    assert result.loc["trailing, ending on tweet date", "correlation"] == 1.0
    assert result.loc["forward, after tweet date", "correlation"] < 0.9


def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
