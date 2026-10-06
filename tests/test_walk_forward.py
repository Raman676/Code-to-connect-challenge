"""
Tests for src/walk_forward.py on small made-up data (no real files, no network).

Run:  python tests/test_walk_forward.py     or     pytest tests/test_walk_forward.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import walk_forward as wf  # noqa: E402

CALENDAR = pd.bdate_range("2013-01-01", "2016-12-31")


def test_symbol_map():
    universe = pd.DataFrame({"ticker": ["GOOG", "GOOGL", "META", "BNY"],
                             "news_symbols": [["GOOG"], ["GOOGL"], ["META", "FB"], ["BK", "BNY"]]})
    m = wf.symbol_map(universe)
    assert m == {"GOOG": "GOOGL", "GOOGL": "GOOGL", "META": "META", "FB": "META", "BK": "BNY", "BNY": "BNY"}


def test_year_windows_have_gaps_and_no_leakage():
    anchor = pd.Series(np.arange(len(CALENDAR)))
    dates = pd.Series(CALENDAR)
    w = wf.year_windows(anchor, dates, CALENDAR, 2016)
    first_test = CALENDAR.searchsorted(pd.Timestamp("2016-01-01"))
    train_pos = anchor[w["train"]]
    # the label window (anchor + 10 trading days) of every training item ends before the test year
    assert train_pos.max() + wf.GAP_DAYS < first_test
    assert train_pos.max() + wf.GAP_DAYS + 1 == first_test
    # validation = the last VAL_DAYS of the training window; fit ends GAP_DAYS before validation starts
    val_pos = anchor[w["val"]]
    assert len(val_pos) == wf.VAL_DAYS and val_pos.max() == train_pos.max()
    assert anchor[w["fit"]].max() + wf.GAP_DAYS + 1 == val_pos.min()
    assert not (w["fit"] & w["val"]).any()
    # test = items dated in 2016, none of which is in training
    assert dates[w["test"]].dt.year.eq(2016).all() and not (w["test"] & w["train"]).any()


def test_tradable_returns_timing():
    prices = pd.DataFrame({"X": [100.0, 101.0, 103.0, 102.0]},
                          index=pd.DatetimeIndex(["2017-02-02", "2017-02-03", "2017-02-06", "2017-02-07"]))
    dates = pd.Series(pd.to_datetime(["2017-02-02", "2017-02-04", "2017-02-06"]))   # Thu, Sat, Mon
    r = wf.tradable_returns(dates, pd.Series(["X", "X", "X"]), prices)
    # Thu item -> trade Fri close, earn Fri->Mon; Sat item -> trade Mon close, earn Mon->Tue;
    # Mon item -> trade Tue close, no next price -> NaN
    assert np.allclose(r[:2], [103 / 101 - 1, 102 / 103 - 1]) and np.isnan(r[2])
    assert np.isnan(wf.tradable_returns(dates[:1], pd.Series(["Y"]), prices)[0])      # unknown ticker


def test_previous_day_counts_are_per_company():
    df = pd.DataFrame({"ticker": ["A", "A", "A", "B", "B"], "anchor_pos": [5, 5, 6, 5, 6]})
    # A on day 6 sees A's 2 headlines from day 5; B on day 6 sees B's 1; day-5 rows see nothing from day 4
    assert wf.previous_day_counts(df).tolist() == [0, 0, 2, 0, 1]


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
