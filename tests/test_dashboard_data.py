"""
Tests for src/dashboard_data.py (the dashboard's data helpers).

Run:  python tests/test_dashboard_data.py     or     pytest tests/test_dashboard_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dashboard_data as dd  # noqa: E402

DAYS = pd.to_datetime(["2019-01-02", "2019-01-03"])
WEIGHTS = pd.DataFrame({"date": [DAYS[0], DAYS[0], DAYS[1], DAYS[1], DAYS[1]],
                        "strategy": ["D", "D", "D", "D", "A"], "ticker": ["X", "Y", "X", "Y", "X"],
                        "weight": [0.6, 0.4, 0.55, 0.45, 1.0], "base_weight": [0.5, 0.5, 0.5, 0.5, 1.0]})


def test_weights_wide_and_on_date():
    wide = dd.weights_wide(WEIGHTS, "D")
    assert wide.shape == (2, 2) and wide.loc[DAYS[1], "Y"] == 0.45
    table = dd.weights_on_date(WEIGHTS, "D", "2019-01-05")             # a Saturday -> last day <= date
    assert np.isclose(table.loc["X", "tilt (pp)"], 5) and table.index[0] == "X"


def test_filter_signals():
    engine = pd.DataFrame({"date": DAYS, "ticker": ["X", "Y"], "news_type": ["Earnings", "Other"]})
    assert len(dd.filter_signals(engine, "All", DAYS[0], DAYS[1])) == 2
    assert dd.filter_signals(engine, "Y", DAYS[0], DAYS[1])["ticker"].tolist() == ["Y"]
    assert len(dd.filter_signals(engine, None, DAYS[0], DAYS[1], ["Earnings"])) == 1


def test_severity_deciles_parses_report_block():
    text = "\n".join(["NEWS - full: model:", "   decile  header line",
                      "        1     0   NaN   NaN   835   1.04   0.66",
                      "        2    50  0.76  0.68   834   1.51   0.95", "", "next block"])
    table = dd.severity_deciles(text)
    assert table["mean |r1| %"].tolist() == [1.04, 1.51]


def test_real_severity_report_has_ten_deciles():
    table = dd.severity_deciles(dd.SEVERITY_REPORT.read_text("utf-8"))
    assert list(table.index) == list(range(1, 11))


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
