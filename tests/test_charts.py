"""
Tests for src/charts.py: the active-signal count and that all four PNGs are written.

Run:  python tests/test_charts.py     or     pytest tests/test_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import charts as ch  # noqa: E402

DAYS = pd.bdate_range("2019-01-02", periods=4)


def test_active_signal_count_only_members():
    base = pd.DataFrame({"X": [0.5] * 4, "Y": [0.5] * 4, "Z": [np.nan] * 4}, index=DAYS)
    s = pd.DataFrame({"X": [0, 0.3, 0.3, 0], "Y": [0, 0, -0.2, 0], "Z": [0.5] * 4}, index=DAYS)
    assert ch.active_signal_count(base, s).tolist() == [0, 1, 2, 0]


def test_make_charts_writes_pngs(tmp=Path(__file__).resolve().parent / "_tmp_charts"):
    navs = pd.DataFrame({"A": [1, 1.1, 1.0, 1.2], "D": [1, 1.05, 1.0, 1.1]}, index=DAYS)
    d = {"weights": {day: pd.Series({"X": 0.5, "Y": 0.5}) for day in DAYS}}
    base = pd.DataFrame(0.5, index=DAYS, columns=["X", "Y"])
    paths = ch.make_charts(navs, d, base, base * 0, out_dir=tmp)
    assert [p.name for p in paths] == ["index_levels.png", "drawdowns.png", "weights_D.png", "active_signals.png"]
    assert all(p.stat().st_size > 0 for p in paths)
    for p in paths:
        p.unlink()
    tmp.rmdir()


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
