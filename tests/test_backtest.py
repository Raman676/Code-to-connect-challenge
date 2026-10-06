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
