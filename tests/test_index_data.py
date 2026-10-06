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
