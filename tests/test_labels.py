"""
Tests for src/labels.py on small made-up data (no real files, no network).

Run:  python tests/test_labels.py     or     pytest tests/test_labels.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import labels as lb  # noqa: E402

# Fri 3 Feb, Mon 6 Feb, Tue 7 Feb 2017 (the weekend is not a trading day)
CALENDAR = pd.DatetimeIndex(["2017-02-03", "2017-02-06", "2017-02-07"])


def test_name_and_symbol_mapping():
    assert lb.TWEET_NAME_TO_TICKER["Facebook"] == "META"
    assert lb.TWEET_NAME_TO_TICKER["Google"] == "GOOGL"
    assert lb.NEWS_SYMBOL_TO_TICKER["FB"] == "META"
    assert lb.NEWS_SYMBOL_TO_TICKER["GOOG"] == "GOOGL"
    assert "Gillette" not in lb.TWEET_NAME_TO_TICKER


def test_anchor_date_only():
    dates = pd.Series(pd.to_datetime(["2017-02-04", "2017-02-05", "2017-02-06", "2017-02-07", "2017-02-03"]))
    pos = lb.anchor_date_only(dates, CALENDAR)
    # Sat -> Fri, Sun -> Fri, Mon -> Fri, Tue -> Mon, Fri -> (no earlier trading day) -1
    assert pos.tolist() == [0, 0, 0, 1, -1]


def test_anchor_timed():
    times = pd.Series(pd.to_datetime([
        "2017-02-06 15:59:00",   # Mon before close -> Fri close
        "2017-02-06 16:00:00",   # Mon at close     -> Mon close
        "2017-02-04 17:00:00",   # Sat evening      -> Fri close
        "2017-02-04 09:00:00",   # Sat morning      -> Fri close
        "2017-02-07 08:00:00",   # Tue morning      -> Mon close
    ]))
    assert lb.anchor_timed(times, CALENDAR).tolist() == [0, 1, 0, 0, 1]


def test_market_features_returns_and_volatility_windows():
    days = pd.bdate_range("2017-01-02", periods=40)
    # Calm before day 20 (tiny alternating moves), volatile after it (large alternating moves).
    moves = np.where(np.arange(40) <= 20, 0.001, 0.05) * np.where(np.arange(40) % 2 == 0, 1, -1)
    prices = pd.DataFrame({"AAA": 100 * np.exp(np.cumsum(moves))}, index=days)
    feats = lb.market_features(pd.Series(["AAA", "AAA", "ZZZ"]), np.array([20, 35, 20]), prices)
    p = prices["AAA"]
    for k in lb.HORIZONS:
        assert np.isclose(feats.loc[0, f"r{k}"], p.iloc[20 + k] / p.iloc[20] - 1)
    assert feats.loc[0, "vol_after"] > 10 * feats.loc[0, "vol_before"]
    assert feats.loc[0, "anchor_date"] == days[20]
    assert feats.loc[1, ["r7", "vol_after"]].isna().all()     # window runs past the last price
    assert feats.loc[2].drop("anchor_date").isna().all()       # unknown ticker


def test_sign_pattern_and_impact_table():
    combos = list(itertools.product([1, -1], repeat=4))
    returns = pd.DataFrame([[0.01 * s for s in c] for c in combos], columns=["r1", "r2", "r3", "r7"])
    patterns = lb.sign_pattern(returns)
    assert patterns.nunique() == 16
    assert sorted(set(patterns) & set(lb.IMPACT_TABLE)) == sorted(lb.IMPACT_TABLE)
    zero = pd.DataFrame({"r1": [0.0], "r2": [0.01], "r3": [0.01], "r7": [0.01]})
    assert lb.sign_pattern(zero).isna().all()


def test_apply_labels_steps():
    df = pd.DataFrame({
        "ticker": ["AAPL"] * 5,
        "r1": [0.01, -0.01, 0.01, 0.01, 0.0],
        "r2": [0.01, -0.01, -0.01, 0.01, 0.01],
        "r3": [0.01, -0.01, 0.01, 0.01, 0.01],
        "r7": [0.01, 0.01, 0.01, 0.01, 0.01],
        "vol_before": [1.0, 1.0, 1.0, 2.0, 1.0],
        "vol_after": [2.0, 2.0, 2.0, 1.0, 2.0],
    })
    log = lb.StepLog()
    log.add("t", "start", len(df))
    out = lb.apply_labels(df, "t", log)
    # row 3 fails the volatility filter, row 4 has a zero return, row 2 (+-++) is not in the table
    assert out.index.tolist() == [0, 1]
    assert out["impact"].tolist() == [10, 5] and out["sentiment"].tolist() == [1, -1]
    assert log.table("t")["rows"].tolist() == [5, 5, 4, 3, 2]


def test_drop_duplicate_texts_keeps_earliest():
    df = pd.DataFrame({"row_id": [5, 1, 2, 3], "text": ["a", "a", "a", "b"], "ticker": ["X", "X", "Y", "X"],
                       "t": pd.to_datetime(["2017-01-03", "2017-01-03", "2017-01-01", "2017-01-02"])})
    out = lb.drop_duplicate_texts(df, "t")
    # "a"/X appears twice on the same date -> keep the earlier file row (row_id 1)
    assert sorted(out["row_id"].tolist()) == [1, 2, 3]


def test_compare_timestamps():
    raw = pd.DataFrame({"raw_id": [0, 1, 2], "key": ["a", "b", "c"],
                        "time": pd.to_datetime(["2020-06-01 09:30:45", "2020-06-01 12:00:10", "2020-06-02 08:00:00"])})
    proc = pd.DataFrame({"key": ["a", "b", "b", "c"],
                         "time": pd.to_datetime(["2020-06-01 09:30:00", "2020-06-01 12:00:00",
                                                 "2020-05-01 12:00:00", "2020-06-02 12:00:00"])})
    result = lb.compare_timestamps(raw, proc)
    assert result["matched"] == 3                    # closest of the two "b" rows is used
    assert np.isclose(result["within_1min"], 2 / 3)
    assert np.isclose(result["exact_minute_floor"], 2 / 3)
    assert np.isclose(result["same_date"], 1.0)
    assert result["top_diffs_min"][240] == 1         # "c" is 4 hours off


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
