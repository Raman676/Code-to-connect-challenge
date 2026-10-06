"""
Tests for src/universe.py on made-up data.

Run:  python tests/test_universe.py     or     pytest tests/test_universe.py
"""
from __future__ import annotations

import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import universe as uv  # noqa: E402

LETTERS = list(string.ascii_uppercase)          # A ... Z


def test_first_selection_is_top_20():
    ranks = pd.Series(range(1, 27), index=LETTERS)
    assert uv.select_constituents(ranks, None) == LETTERS[:20]


def test_methodology_section_7_example():
    # Members A..T. New ranks: S 21 (holdover inside the buffer), T 27 (falls out), U 18 and V 20 (entrants).
    current = LETTERS[:20]
    ranks = {t: r for t, r in zip(LETTERS[:18], list(range(1, 18)) + [19])}   # A..R hold ranks 1-17 and 19
    ranks.update({"S": 21, "T": 27, "U": 18, "V": 20, "W": 22, "X": 23, "Y": 24, "Z": 25})
    members = uv.select_constituents(pd.Series(ranks), current)
    # The document keeps S as well, ending with 21 names; the exact-20 rule drops the lowest-ranked holdover (S).
    assert len(members) == 20
    assert set(members) == set(LETTERS[:18]) | {"U", "V"}


def test_buffer_keeps_rank_25_and_drops_26():
    current = LETTERS[:20]
    ranks = pd.Series({t: i + 1 for i, t in enumerate(LETTERS[2:20])} | {"A": 25, "B": 26, "U": 19, "V": 20,
                                                                          "W": 21, "X": 22})
    # 18 members ranked 1-18, A at 25 (kept), B at 26 (exits); U(19) and V(20) enter -> 21 -> A is dropped
    members = uv.select_constituents(ranks, current)
    assert len(members) == 20 and "B" not in members and {"U", "V"} <= set(members)
    assert "A" not in members                       # the lowest-ranked holdover makes way
    ranks2 = ranks.copy()
    ranks2["V"] = 30                                # only one entrant -> A (rank 25) now stays
    members2 = uv.select_constituents(ranks2, current)
    assert "A" in members2 and "V" not in members2 and len(members2) == 20


def test_fill_from_best_non_members_when_short():
    current = LETTERS[:20]
    ranks = pd.Series({t: i + 1 for i, t in enumerate(LETTERS[:17])} | {"R": 30, "S": 31, "T": 32,
                                                                          "U": 21, "V": 22, "W": 23})
    # 17 holdovers, 3 members fall below 25, no entrant ranked <= 20 -> fill with U, V, W (ranks 21-23)
    members = uv.select_constituents(ranks, current)
    assert members[-3:] == ["U", "V", "W"] and len(members) == 20


def test_unranked_member_is_removed():
    current = LETTERS[:20]
    ranks = pd.Series({t: i + 1 for i, t in enumerate(LETTERS[1:22])})   # A has no FFMC (e.g. delisted)
    assert "A" not in uv.select_constituents(ranks, current)


def test_review_and_effective_dates():
    calendar = pd.bdate_range("2015-12-01", "2016-07-15")
    reviews = uv.review_dates(calendar, first="2015-12-31", last="2016-06-30")
    assert [d.date().isoformat() for d in reviews] == ["2015-12-31", "2016-03-31"]   # 2016-06-30 is the last day
    assert uv.effective_date(pd.Timestamp("2015-12-31"), calendar) == pd.Timestamp("2016-01-01")


def test_shares_as_of_and_ffmc():
    history = pd.DataFrame({"ticker": ["X", "X"], "date": pd.to_datetime(["2016-01-04", "2016-01-07"]),
                            "shares_adj": [10.0, 12.0]})
    dates = pd.bdate_range("2016-01-01", "2016-01-08")
    shares = uv.shares_as_of(history, dates)
    assert np.isnan(shares.loc["2016-01-01", "X"]) and shares.loc["2016-01-06", "X"] == 10.0
    assert shares.loc["2016-01-08", "X"] == 12.0
    data = uv.MarketData(close=pd.DataFrame({"X": 2.0}, index=dates), adj_close=None, shares=shares,
                         float_factor=pd.Series({"X": 0.5}), sectors=None, companies=None)
    assert uv.ffmc(data).loc["2016-01-08", "X"] == 2.0 * 12.0 * 0.5


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
