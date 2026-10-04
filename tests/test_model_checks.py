"""
Tests for the helper functions in src/model_checks.py (no model download needed).

Run:  python tests/test_model_checks.py     or     pytest tests/test_model_checks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_checks as mc  # noqa: E402


def test_unk_stats():
    stats = mc.unk_stats([[5, 2, 7], [8, 9], [2, 2]], unk_id=2)
    assert stats["word pieces"] == 7
    assert stats["[UNK] % of pieces"] == round(100 * 3 / 7, 2)
    assert stats["% texts with [UNK]"] == round(100 * 2 / 3, 1)


def test_sentiment_direction():
    classes = [1, 4, 5, 6, 7, 10]
    coef = np.array([[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [1.0, 3.0]])
    # feature 0: +1 on every positive class, -1 on every negative -> 2; feature 1: only class 10 -> 1
    assert np.allclose(mc.sentiment_direction(coef, classes), [2.0, 1.0])


def test_anchor_on_date():
    calendar = pd.DatetimeIndex(["2017-02-03", "2017-02-06", "2017-02-07"])   # Fri, Mon, Tue
    dates = pd.Series(pd.to_datetime(["2017-02-06", "2017-02-04", "2017-02-07", "2017-02-02"]))
    # Mon -> Mon (same day), Sat -> Fri, Tue -> Tue, before the calendar -> -1
    assert mc.anchor_on_date(dates, calendar).tolist() == [1, 0, 2, -1]


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
