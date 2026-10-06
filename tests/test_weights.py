"""
Tests for src/weights.py.

Run:  python tests/test_weights.py     or     pytest tests/test_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import weights as wt  # noqa: E402

EPS = 1e-9


def test_base_weights_and_tilt_example():
    base = wt.base_weights(pd.Series({"A": 1000.0, "B": 800.0, "C": 200.0}))
    assert np.isclose(base["A"], 0.5) and np.isclose(base.sum(), 1.0)
    # methodology section 15: 10% x exp(0.30 x 0.8 x 0.8)
    raw = wt.tilt(pd.Series({"X": 0.10}), pd.Series({"X": 0.8}), pd.Series({"X": 0.8}))
    assert np.isclose(raw["X"], 0.10 * np.exp(0.192))
    # no news -> no tilt
    assert wt.tilt(pd.Series({"X": 0.1, "Y": 0.2}), pd.Series({"X": 0.5}))["Y"] == 0.2


def test_fit_to_bounds():
    raw = pd.Series([5.0, 3.0, 1.0, 1.0], index=list("ABCD"))
    lower, upper = pd.Series(0.05, index=raw.index), pd.Series(0.40, index=raw.index)
    w = wt.fit_to_bounds(raw, lower, upper)
    assert np.isclose(w.sum(), 1.0) and (w <= upper + EPS).all() and (w >= lower - EPS).all()
    assert np.isclose(w["A"], 0.40)                                   # capped
    assert np.isclose(w["B"] / w["C"], 3.0)                           # the others keep their proportions
    try:
        wt.fit_to_bounds(raw, lower, pd.Series(0.2, index=raw.index))  # 4 x 20% < 100%
    except ValueError:
        pass
    else:
        raise AssertionError("expected infeasible")


def test_cap_weights():
    base = pd.Series([0.30, 0.20] + [0.0625] * 8, index=list("ABCDEFGHIJ"))
    w = wt.cap_weights(base, 0.12)
    assert np.isclose(w.sum(), 1.0) and w.max() <= 0.12 + EPS
    assert np.isclose(w["A"], 0.12) and np.isclose(w["B"], 0.12)
    assert np.allclose(w.iloc[2:], 0.76 / 8)                          # excess spread pro rata


def test_constrained_weights_review_day_and_limits():
    base = pd.Series([0.20, 0.10] + [0.0875] * 8, index=list("ABCDEFGHIJ"))
    raw = wt.tilt(base, pd.Series({"B": 1.0, "C": -1.0}))
    # review day: cap and tilt band apply at once; the cap wins over A's tilt band (15% - 25%)
    w, relaxed = wt.constrained_weights(raw, base, previous=None, review_day=True)
    assert not relaxed and np.isclose(w.sum(), 1.0) and w["A"] <= 0.12 + EPS
    assert (w <= base * 1.25 + EPS).all() and (w[1:] >= base[1:] * 0.75 - EPS).all()
    # next day, from a portfolio where A drifted to 16%: A may fall by at most 1 pp
    prev = w.copy()
    prev["A"] = 0.16
    prev = prev / prev.sum()
    w2, relaxed2 = wt.constrained_weights(raw, base, previous=prev, review_day=False)
    assert not relaxed2 and np.isclose(w2.sum(), 1.0)
    assert ((w2 - prev).abs() <= 0.01 + EPS).all()
    assert np.isclose(w2["A"], prev["A"] - 0.01)                      # moving towards the cap as fast as allowed


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
