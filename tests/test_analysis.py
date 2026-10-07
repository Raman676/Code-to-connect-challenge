"""
Tests for src/analysis.py (daily IC, Fama-MacBeth, news-event and window returns).

Run:  python tests/test_analysis.py     or     pytest tests/test_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis as an  # noqa: E402

DAYS = pd.bdate_range("2019-01-02", periods=40)
NAMES = list("ABCDEFGH")
RNG = np.random.default_rng(0)


def test_daily_ic_perfect_and_skips_flat_days():
    s = pd.DataFrame(RNG.normal(size=(40, 8)), index=DAYS, columns=NAMES)
    s.iloc[0] = 0.0                                          # no variation -> skipped
    members = pd.DataFrame(True, index=DAYS, columns=NAMES)
    ic = an.daily_ic(s, 2 * s, members)                      # next return ranks exactly like S
    assert len(ic) == 39 and np.allclose(ic, 1.0)


def test_fama_macbeth_recovers_known_slope():
    s = pd.DataFrame(RNG.normal(size=(40, 8)), index=DAYS, columns=NAMES)
    p1 = pd.DataFrame(RNG.normal(size=(40, 8)), index=DAYS, columns=NAMES)
    p5 = pd.DataFrame(RNG.normal(size=(40, 8)), index=DAYS, columns=NAMES)
    y = 0.01 + 0.5 * s - 0.2 * p1                            # no noise: exact coefficients
    members = pd.DataFrame(True, index=DAYS, columns=NAMES)
    fm = an.fama_macbeth(y, {"S_eff": s, "p1": p1, "p5": p5}, members)
    assert np.isclose(fm.loc["S_eff", "mean coef"], 0.5) and np.isclose(fm.loc["p1", "mean coef"], -0.2)
    assert np.isclose(fm.loc["p5", "mean coef"], 0.0, atol=1e-9) and fm.loc["S_eff", "days"] == 40


def test_top_news_days_ignore_non_members():
    s = pd.DataFrame(0.0, index=DAYS, columns=NAMES)
    s.loc[DAYS[5], "A"], s.loc[DAYS[9], "B"] = 0.9, -0.9
    members = pd.DataFrame(True, index=DAYS, columns=NAMES)
    members.loc[DAYS[9], "B"] = False
    top = an.top_news_days(s, pd.DataFrame(1.0, index=DAYS, columns=NAMES), members, share=0.025)
    assert list(top) == [DAYS[5]]


def test_active_on_days_uses_next_day_return():
    navs = pd.DataFrame({"A": [1.0, 1.0, 1.01], "D": [1.0, 1.0, 1.03]}, index=DAYS[:3])
    out = an.active_on_days(navs, DAYS[1:2])
    assert np.isclose(out.loc["D", "mean active bp per day"], 200)   # 3% - 1% earned from day 1 to day 2


def test_window_returns():
    navs = pd.DataFrame({"A": [1.0, 1.1], "D": [1.0, 1.2]}, index=DAYS[:2])
    w = an.window_returns(navs, str(DAYS[0].date()), str(DAYS[1].date()))
    assert np.isclose(w.loc["D", "vs A (pp)"], 10)


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
