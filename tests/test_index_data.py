"""
Tests for src/index_data.py helpers (no network).

Run:  python tests/test_index_data.py     or     pytest tests/test_index_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import index_data as idx  # noqa: E402


def test_universe_file():
    u = idx.load_universe()
    assert len(u) == 101 and u["ticker"].is_unique
    assert {"GOOG", "GOOGL", "BRK.B", "META"} <= set(u["ticker"])
    assert u.set_index("ticker").loc["META", "news_symbols"] == ["META", "FB"]
    assert u.set_index("ticker").loc["BRK.B", "yf_symbol"] == "BRK-B"
    assert u["gics_sector"].notna().all()


def test_shares_asof():
    history = pd.DataFrame({"symbol": "X", "date": pd.to_datetime(["2015-01-10", "2015-06-30"]), "shares": [100.0, 120.0]})
    dates = pd.to_datetime(["2015-01-01", "2015-03-31", "2015-07-01", "2016-06-30"])
    assert np.allclose(idx.shares_asof(history, "X", dates).to_numpy(), [np.nan, 100, 120, 120], equal_nan=True)
    fresh = idx.shares_asof(history, "X", dates, max_age_days=200).to_numpy()
    assert np.allclose(fresh, [np.nan, 100, 120, np.nan], equal_nan=True)     # last value is 366 days old
    assert idx.shares_asof(history, "Y", dates).isna().all()


def test_float_ratio():
    info = pd.DataFrame({"float_shares": [90.0, 5.0, np.nan], "shares_out": [100.0, 2.0, 10.0]})
    assert np.allclose(idx.float_ratio(info).to_numpy(), [0.9, 2.5, np.nan], equal_nan=True)


def test_quarter_end_reviews():
    calendar = pd.bdate_range("2011-12-01", "2012-07-15")
    reviews = idx.quarter_end_reviews(calendar, start="2012-01-01", end="2012-07-15")
    assert [d.date().isoformat() for d in reviews] == ["2012-03-30", "2012-06-29"]


def test_split_adjust_shares():
    shares = pd.DataFrame({"symbol": "X", "date": pd.to_datetime(["2017-01-01", "2017-06-01", "2018-06-01"]),
                           "shares": [100.0, 200.0, 200.0]})
    splits = pd.DataFrame({"symbol": ["X", "X"], "date": pd.to_datetime(["2017-03-01", "2020-01-01"]), "ratio": [2.0, 4.0]})
    out = idx.split_adjust_shares(shares, splits)
    # 2017-01 is before both splits (x8); the later rows are only before the 2020 split (x4)
    assert out["shares_adj"].tolist() == [800.0, 800.0, 800.0]


def _shares(values, start="2017-01-02", step_days=7, symbol="X"):
    dates = pd.date_range(start, periods=len(values), freq=f"{step_days}D")
    return pd.DataFrame({"symbol": symbol, "date": dates, "shares_adj": np.asarray(values, dtype=float)})


def test_clean_shares_removes_spikes_keeps_known_events_and_flags_the_rest():
    spike = _shares([100, 100, 160, 101, 100], symbol="S")                 # back within 7 days -> spike
    merger = _shares([100, 100, 150, 150, 150], start="2019-11-01", symbol="M")
    other = _shares([100, 100, 40, 40, 40], symbol="F")                    # persists -> flagged
    clean, events = idx.clean_shares(pd.concat([spike, merger, other]), known_events={("M", "2019-11"): "deal"})
    assert len(clean) == 14 and 160 not in clean["shares_adj"].tolist()
    actions = dict(zip(events["symbol"], events["action"]))
    assert actions["S"].startswith("spike removed")
    assert actions["M"] == "kept (known event: deal)"
    assert actions["F"] == "flagged for review (kept)"


def test_clean_shares_manual_drops():
    data = _shares([100, 100, 40, 100, 100], step_days=60, symbol="A")   # 40 is too far apart for the spike rule
    bad_date = data["date"].iloc[2].strftime("%Y-%m-%d")
    clean, events = idx.clean_shares(data, known_events={}, manual_drops={("A", bad_date): "bad value"})
    assert clean["shares_adj"].tolist() == [100, 100, 100, 100]
    assert events["action"].tolist() == ["manual drop: bad value"]       # nothing else flagged


def test_float_factors():
    info = pd.DataFrame({"ticker": ["A", "B", "GOOGL", "BRK.B"], "float_shares": [90.0, 120.0, 89.0, 1.0],
                         "shares_out": [100.0, 100.0, 50.0, 1000.0], "implied_shares_out": [100.0, 100.0, 100.0, 1000.0]})
    f = idx.float_factors(info).set_index("ticker")["factor"]
    assert f["A"] == 0.9 and f["B"] == 1.0                 # capped at 1
    assert f["GOOGL"] == 0.89                              # all share classes
    assert f["BRK.B"] == 0.85                              # manual override


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
