"""
Tests for src/export_results.py (summary, yearly returns, tilt and the Excel file).

Run:  python tests/test_export_results.py     or     pytest tests/test_export_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import export_results as ex  # noqa: E402

DAYS = pd.to_datetime(["2019-12-30", "2019-12-31", "2020-02-28", "2020-03-31", "2020-06-30"])


def test_yearly_returns_and_march():
    navs = pd.DataFrame({"S": [1.0, 1.1, 1.21, 0.968, 1.0]}, index=DAYS)
    y = ex.yearly_returns(navs)["S"]
    assert np.isclose(y["2019"], 10) and np.isclose(y["2020"], 100 * (1.0 / 1.1 - 1))
    assert np.isclose(y["Mar-2020"], -20)


def test_avg_abs_tilt():
    weights = {DAYS[0]: pd.Series({"X": 0.6, "Y": 0.4}), DAYS[1]: pd.Series({"X": 0.5, "Y": 0.5})}
    base = pd.DataFrame({"X": [0.5, 0.5], "Y": [0.5, 0.5]}, index=DAYS[:2])
    assert np.isclose(ex.avg_abs_tilt(weights, base), (0.1 + 0.0) / 2)


def test_export_writes_all_sheets(tmp=Path(__file__).resolve().parent / "_tmp_export"):
    ex.XLSX_PATH, ex.SUMMARY_CSV = tmp / "r.xlsx", tmp / "s.csv"
    days = pd.bdate_range("2019-01-02", periods=5)
    w = {d: pd.Series({"X": 0.5, "Y": 0.5}) for d in days}
    daily = pd.DataFrame({"nav_after_cost": [1, 1.01, 1.02, 1.0, 1.03], "cost": 0.0, "traded": 0.0}, index=days)
    res = {"A-capped": {"daily": daily, "weights": w}, "D": {"daily": daily, "weights": w}}
    navs = pd.DataFrame({"A-capped": daily["nav_after_cost"], "D": daily["nav_after_cost"],
                         ex.OEX: daily["nav_after_cost"]})
    reviews = pd.DataFrame({"review": [days[0]], "effective": [days[1]], "members": [["X", "Y"]],
                            "added": [[]], "removed": [[]]})
    base = pd.DataFrame(0.5, index=days, columns=["X", "Y"])
    summary = ex.export_all(navs, res, pd.Series(0.0, index=days), base, pd.Series({"X": "IT", "Y": "IT"}), reviews,
                            {0.3: res["D"], 1.0: res["D"]})
    assert list(summary.index) == ["A-capped", "D", ex.OEX]
    with pd.ExcelFile(ex.XLSX_PATH) as book:              # 'with' closes the file again (Windows)
        sheets = book.sheet_names
    assert sheets == ["Summary", "Yearly", "Daily_NAV", "Daily_Weights_D", "Turnover_Costs", "Constituents",
                      "Sensitivity", "Definitions"]
    sens = pd.read_excel(ex.XLSX_PATH, sheet_name="Sensitivity")
    assert sens["alpha"].tolist() == [0.3, 1.0] and np.allclose(sens["CAGR - A-capped (pp)"], 0)
    assert ex.SUMMARY_CSV.exists()
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


def test_save_dashboard_csvs(tmp=Path(__file__).resolve().parent / "_tmp_dash"):
    tmp.mkdir(exist_ok=True)
    ex.WEIGHTS_CSV, ex.SIGNALS_CSV = tmp / "w.csv", tmp / "s.csv"
    days = pd.bdate_range("2019-01-02", periods=2)
    res = {"D": {"weights": {d: pd.Series({"X": 0.7, "Y": 0.3}) for d in days}}}
    base = pd.DataFrame({"X": [0.6, 0.6], "Y": [0.4, 0.4], "Z": [np.nan, np.nan]}, index=days)
    panel = pd.DataFrame({"X": [0.5, 0.0], "Y": [0.0, 0.0]}, index=days)
    sig = {"S": panel, "C": panel, "S_eff": panel, "n_events": (panel != 0).astype(int)}
    ex.save_dashboard_csvs(res, base, sig)
    w = pd.read_csv(ex.WEIGHTS_CSV)
    assert len(w) == 4 and list(w.columns) == ["date", "strategy", "ticker", "weight", "base_weight"]
    assert np.isclose(w.loc[w["ticker"] == "X", "base_weight"], 0.6).all()
    s = pd.read_csv(ex.SIGNALS_CSV)
    assert len(s) == 1 and s["ticker"][0] == "X" and s["S_eff"][0] == 0.5
    for f in tmp.iterdir():
        f.unlink()
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
