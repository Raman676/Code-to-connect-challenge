"""
Tests for src/prices.py: caching logic, using a fake downloader (no network).

Run:  python tests/test_prices.py     or     pytest tests/test_prices.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prices as px  # noqa: E402


def _fake_prices(tickers):
    dates = pd.bdate_range("2017-01-02", periods=5, name="date")
    return pd.DataFrame({t: np.arange(1.0, 6.0) * (i + 1) for i, t in enumerate(tickers)}, index=dates)


class CountingDownloader:
    def __init__(self):
        self.calls = 0

    def __call__(self, tickers):
        self.calls += 1
        return _fake_prices(tickers)


def test_cache_is_written_then_reused():
    cache = Path(tempfile.mkdtemp()) / "adj_close.csv"
    download = CountingDownloader()
    first = px.load_adj_close(["AAA", "BBB"], cache_path=cache, downloader=download)
    second = px.load_adj_close(["AAA", "BBB"], cache_path=cache, downloader=download)
    assert download.calls == 1                       # second call read the cache
    assert cache.exists() and cache.with_suffix(".info.txt").exists()
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["AAA", "BBB"]
    assert isinstance(first.index, pd.DatetimeIndex) and first.index.name == "date"


def test_refresh_downloads_again():
    cache = Path(tempfile.mkdtemp()) / "adj_close.csv"
    download = CountingDownloader()
    px.load_adj_close(["AAA"], cache_path=cache, downloader=download)
    px.load_adj_close(["AAA"], cache_path=cache, downloader=download, refresh=True)
    assert download.calls == 2


def test_no_cache_and_no_tickers_raises():
    cache = Path(tempfile.mkdtemp()) / "missing.csv"
    try:
        px.load_adj_close(None, cache_path=cache, downloader=CountingDownloader())
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_summarize():
    prices = _fake_prices(["AAA", "BBB"])
    prices.loc[prices.index[:2], "BBB"] = np.nan     # BBB starts trading later
    summary = px.summarize(prices)
    assert summary.loc["AAA", "days"] == 5 and summary.loc["BBB", "days"] == 3
    assert summary.loc["BBB", "first"] == prices.index[2]


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
