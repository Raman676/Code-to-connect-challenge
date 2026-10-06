"""
Tests for src/news_type.py.

Run:  python tests/test_news_type.py     or     pytest tests/test_news_type.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import news_type as nt  # noqa: E402

EXAMPLES = {
    "Credit Event": "Moody's Cuts Ratings Outlook as Company Weighs Chapter 11 Filing",
    "Analyst Rating": "Morgan Stanley Upgrades Apple to Overweight, Raises Price Target to $250",
    "Earnings": "Apple Reports Q3 EPS $2.58 vs $2.10 Estimate",
    "Merger/Acquisition": "Disney to Acquire 21st Century Fox Assets in $52B Deal",
    "Legal/Regulatory": "FTC Opens Antitrust Probe Into Facebook",
    "Management Change": "Uber CEO Travis Kalanick Resigns",
    "Product Launch": "Tesla Unveils Its Electric Semi Truck",
    "Macroeconomic": "Fed Raises Interest Rate by 25 Basis Points",
    "Geopolitical": "Stocks Fall as US Announces New Tariff on Chinese Imports",
    "Other": "Shares of Several Companies Are Moving in Wednesday's Session",
}


def test_one_headline_per_category():
    for expected, text in EXAMPLES.items():
        assert nt.classify(text) == expected, (text, nt.classify(text))


def test_precedence_debt_downgrade_is_credit_event():
    assert nt.classify("S&P announces debt downgrade for the company") == "Credit Event"
    assert nt.classify("Goldman downgrades the company") == "Analyst Rating"
    # analyst rating before earnings: a downgrade mentioning estimates is still a rating action
    assert nt.classify("Analyst downgrades stock after cutting EPS estimates") == "Analyst Rating"


def test_whole_words_only():
    assert nt.classify("Federal court approves plan") == "Other"        # 'fed' inside 'Federal'
    assert nt.classify("Profit warning ahead") == "Earnings"            # 'war' inside 'warning' does not count
    assert nt.classify("Secures new contract") == "Other"               # 'sec' inside 'Secures'
    assert nt.classify("trade   war escalates") == "Geopolitical"       # extra spaces in a phrase
    assert nt.classify(None) == "Other"


def test_add_news_type_and_accuracy_file():
    df = pd.DataFrame({"date": ["2019-01-02"] * 3, "ticker": ["AAPL"] * 3,
                       "text": [EXAMPLES["Earnings"], EXAMPLES["Other"], EXAMPLES["Analyst Rating"]]})
    out = nt.add_news_type(df)
    assert out["news_type"].tolist() == ["Earnings", "Other", "Analyst Rating"]
    path = Path(tempfile.mkdtemp()) / "check.csv"
    sample = nt.write_check_file(out, path, n=3, seed=1)
    assert list(sample.columns) == ["date", "ticker", "text", "news_type", "correct"]
    filled = pd.read_csv(path, dtype=str, keep_default_na=False)
    filled["correct"] = ["y", "Earnings", ""]                           # one right, one wrong unless it was Earnings
    filled.to_csv(path, index=False)
    result = nt.check_accuracy(path)
    assert result["checked"] == 2
    expected = (1 + (filled.loc[1, "news_type"] == "Earnings")) / 2
    assert abs(result["accuracy"] - expected) < 1e-9


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
