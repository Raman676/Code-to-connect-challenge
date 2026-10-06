"""
Tests for src/api.py with FastAPI's TestClient on a small made-up signals file.

Run:  python tests/test_api.py     or     pytest tests/test_api.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import api  # noqa: E402

ROWS = pd.DataFrame({
    "date": ["2019-01-02", "2019-01-03", "2019-02-01", "2018-10-01"],
    "company_name": ["Apple Inc.", "Apple Inc.", "Microsoft", "Nike"],
    "ticker": ["AAPL", "AAPL", "MSFT", "NKE"],
    "source": ["news", "news", "news", "social"],
    "text": ["a", "b", "c", "d"],
    "sentiment_score": [0.5, -0.1, 0.2, -0.4],
    "impact": [8, 4, 6, 2],
    "news_type": ["Earnings", "Analyst Rating", "Earnings", "Other"],
    "confidence": [0.5, 0.1, 0.2, 0.4],
})


def make_client() -> TestClient:
    path = Path(tempfile.mkdtemp()) / "signals.csv"
    ROWS.to_csv(path, index=False)
    return TestClient(api.create_app(path))


def test_health():
    r = make_client().get("/health")
    assert r.status_code == 200 and r.json()["rows"] == 4 and r.json()["status"] == "ok"


def test_signals_filters_and_limit():
    client = make_client()
    rows = client.get("/signals", params={"ticker": "aapl"}).json()
    assert [r["date"] for r in rows] == ["2019-01-03", "2019-01-02"]               # newest first
    rows = client.get("/signals", params={"start": "2019-01-03", "end": "2019-02-01"}).json()
    assert {r["ticker"] for r in rows} == {"AAPL", "MSFT"} and len(rows) == 2
    assert len(client.get("/signals", params={"source": "social"}).json()) == 1
    assert len(client.get("/signals", params={"news_type": "earnings"}).json()) == 2
    assert len(client.get("/signals", params={"limit": 1}).json()) == 1
    assert client.get("/signals", params={"source": "radio"}).status_code == 422   # validated
    assert client.get("/signals", params={"limit": 0}).status_code == 422


def test_summary():
    client = make_client()
    s = client.get("/signals/summary", params={"ticker": "AAPL"}).json()
    assert s["count"] == 2 and abs(s["mean_sentiment"] - 0.2) < 1e-9 and s["mean_impact"] == 6.0
    assert s["news_type_counts"] == {"Earnings": 1, "Analyst Rating": 1}
    assert client.get("/signals/summary").json()["count"] == 4
    assert client.get("/signals/summary", params={"ticker": "ZZZZ"}).status_code == 404


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
