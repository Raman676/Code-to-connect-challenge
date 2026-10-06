"""
Tests for src/build_engine_output.py, using small made-up input files.

Run:  python tests/test_build_engine_output.py     or     pytest tests/test_build_engine_output.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_engine_output as be  # noqa: E402


def test_build_combines_sources_with_schema():
    tmp = Path(tempfile.mkdtemp())
    pd.DataFrame({"ticker": ["AAPL", "NKE"], "company": ["Apple Inc.", None]}).dropna().to_csv(tmp / "u.csv", index=False)
    pd.DataFrame({"date": ["2019-01-02"], "ticker": ["AAPL"], "text": ["Apple Reports Q1 EPS Beat"],
                  "sentiment_score": [0.4], "severity": [7]}).to_csv(tmp / "s.csv", index=False)
    pd.DataFrame({"date": ["2018-10-01", "2018-10-01"], "company_name": ["Nike", "Apple"],
                  "ticker": ["NKE", "AAPL"], "source": ["tweets", "news"], "text": ["Nike ad", "x"],
                  "sentiment_score": [-0.3, 0.1], "impact": [5, 2]}).to_csv(tmp / "o.csv", index=False)
    be.UNIVERSE_CSV, be.SIGNALS_CSV, be.SENTIMENT_OUTPUT = tmp / "u.csv", tmp / "s.csv", tmp / "o.csv"
    out = be.build()
    assert list(out.columns) == be.COLUMNS
    assert out["source"].tolist() == ["social", "news"]                  # sorted by date; news rows of the
    nike = out[out["ticker"] == "NKE"].iloc[0]                           # sentiment output are not used
    assert nike["company_name"] == "Nike"                                # fallback name outside the S&P 100
    apple = out[out["ticker"] == "AAPL"].iloc[0]
    assert apple["company_name"] == "Apple Inc." and apple["news_type"] == "Earnings"
    assert np.allclose(out["confidence"], out["sentiment_score"].abs())


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
