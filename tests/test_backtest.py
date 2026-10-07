"""
Tests for the backtest engine and metrics in src/backtest.py.

Run:  python tests/test_backtest.py     or     pytest tests/test_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest as bt  # noqa: E402
import signals  # noqa: E402,F401  (used in the timing test)

DAYS = pd.bdate_range("2016-01-04", periods=4)
PRICES = pd.DataFrame({"X": [100.0, 110.0, 99.0, 99.0], "Y": [50.0, 50.0, 55.0, 55.0]}, index=DAYS)


def test_buy_and_hold_drifts_without_cost():
    target = pd.Series({"X": 0.5, "Y": 0.5})
    res = bt.simulate(PRICES, DAYS, lambda day, held: target if day == DAYS[0] else None, cost_bps=10)
    nav = res["daily"]["nav_after_cost"]
    # 50/50 bought on day 0 (no cost for the initial build), then held: value = 0.5*X/100 + 0.5*Y/50
    expected = 0.5 * PRICES["X"] / 100 + 0.5 * PRICES["Y"] / 50
    assert np.allclose(nav.to_numpy(), expected.to_numpy())
    assert res["daily"]["cost"].sum() == 0


def test_daily_rebalancing_pays_costs():
    target = pd.Series({"X": 0.5, "Y": 0.5})
    res = bt.simulate(PRICES, DAYS, lambda day, held: target, cost_bps=10)
    d = res["daily"]
    # day 1: X up 10% -> weights drift to 0.5238/0.4762 -> trade 2 x 0.0238 back to 50/50
    drift_x = 0.5 * 1.1 / (0.5 * 1.1 + 0.5)
    assert np.isclose(d["traded"].iloc[1], 2 * (drift_x - 0.5))
    assert np.isclose(d["cost"].iloc[1], d["traded"].iloc[1] * 0.001)
    nav1 = 1.05 * (1 - d["cost"].iloc[1])                              # +5% on day 1, then costs
    assert np.isclose(d["nav_after_cost"].iloc[1], nav1)


def test_performance_metrics():
    nav = pd.Series([1.0, 1.1, 0.88, 0.99], index=DAYS)
    rf = pd.Series(0.0, index=DAYS)
    p = bt.performance(nav, rf)
    assert np.isclose(p["total return %"], -1.0)
    assert np.isclose(p["max drawdown %"], -20.0)
    with_bench = bt.performance(nav, rf, nav.pct_change())
    assert np.isclose(with_bench["tracking error %"], 0.0) and np.isclose(with_bench["beta"], 1.0)


def test_daily_rf():
    rf = bt.daily_rf(pd.Series([5.0], index=DAYS[:1]), DAYS)
    assert np.allclose(rf, 1.05 ** (1 / 252) - 1)                       # forward-filled


def test_one_lag_single_fake_headline():
    """News dated d tilts the weights set at d+1's close and earns the return from d+1 to d+2 - one lag only."""
    import signals as sg
    cal = pd.bdate_range("2019-01-07", periods=6)              # Mon..Mon
    d = cal[1]
    headline = pd.DataFrame({"date": [d], "ticker": ["X"], "score": [0.9], "severity": [10], "model_conf": [1.0]})
    sig = sg.security_signals(sg.build_events(headline, cal), cal, ["X", "Y"])
    assert sig["S_eff"].loc[d, "X"] == 0                        # not usable on its own date
    assert sig["S_eff"].loc[cal[2], "X"] == 0.9                 # first usable at d+1
    # X rises 10% on d+1 (before the tilt can be used) and 10% on d+2 (earned with the tilted weights)
    prices = pd.DataFrame({"X": [100, 100, 110, 121, 121, 121.0], "Y": [100.0] * 6}, index=cal)
    base = pd.DataFrame({"X": 0.5, "Y": 0.5}, index=cal)
    res = bt.simulate(prices, cal, bt.sentiment_target_fn(base, sig["S_eff"], sig["C"], [cal[0]], "B"), cost_bps=0)
    w = res["weights"]
    assert np.isclose(w[d]["X"], 0.5) and w[cal[2]]["X"] > 0.5
    nav = res["daily"]["nav_after_cost"]
    assert np.isclose(nav[cal[2]] / nav[d] - 1, 0.05)           # d+1's jump earned with base weights
    assert np.isclose(nav[cal[3]] / nav[cal[2]] - 1, 0.10 * w[cal[2]]["X"])   # d+2's with the tilt


def test_sentiment_targets_b_c_d():
    days = pd.bdate_range("2019-01-07", periods=3)
    base = pd.DataFrame({"X": 0.6, "Y": 0.4}, index=days)
    s = pd.DataFrame({"X": 1.0, "Y": 0.0}, index=days)
    c = pd.DataFrame({"X": 0.5, "Y": 0.0}, index=days)
    b = bt.sentiment_target_fn(base, s, c, [days[0]], "B")(days[1], None)
    cc = bt.sentiment_target_fn(base, s, c, [days[0]], "C")(days[1], None)
    assert b["X"] > cc["X"] > 0.6 and np.isclose(b.sum(), 1) and np.isclose(cc.sum(), 1)
    # D: ten equal stocks, X has full positive sentiment. Raw tilt 13.5% -> band/cap allow 12%, but from
    # yesterday's 10% the 1 pp daily limit lets X rise only to 11%.
    names = list("XABCDEFGHI")
    base10 = pd.DataFrame(0.1, index=days, columns=names)
    s10, c10 = pd.DataFrame(0.0, index=days, columns=names), pd.DataFrame(0.0, index=days, columns=names)
    s10["X"], c10["X"] = 1.0, 1.0
    relaxed = []
    dw = bt.sentiment_target_fn(base10, s10, c10, [days[0]], "D", relaxed)(days[1], base10.loc[days[0]])
    assert np.isclose(dw["X"], 0.11) and np.isclose(dw.sum(), 1) and relaxed == []
    review = bt.sentiment_target_fn(base10, s10, c10, [days[1]], "D")(days[1], base10.loc[days[0]])
    assert np.isclose(review["X"], 0.12)                        # review day: no daily limit, cap binds


def test_daily_base_weights_follow_membership():
    days = pd.bdate_range("2019-01-07", periods=3)
    ffmc = pd.DataFrame({"X": [3.0, 3, 3], "Y": [1.0, 1, 1], "Z": [1.0, 1, 1]}, index=days)
    reviews = pd.DataFrame({"effective": [days[0], days[2]], "members": [["X", "Y"], ["X", "Z"]]})
    base = bt.daily_base_weights(ffmc, reviews, days)
    assert np.isclose(base.loc[days[1], "X"], 0.75) and np.isnan(base.loc[days[1], "Z"])
    assert np.isclose(base.loc[days[2], "Z"], 0.25)


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
