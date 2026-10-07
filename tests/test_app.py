"""
Smoke test for app.py: runs the dashboard headless (Streamlit's AppTest) on the real output files
and checks that every tab renders without an error. Needs python src/backtest.py to have been run.

Run:  python tests/test_app.py     or     pytest tests/test_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_renders_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Index", "Weights", "Signals feed", "Engine results", "Sensitivity"]


def test_dashboard_other_strategy_and_all_tickers():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.selectbox[0].select("A-capped").run()
    at.sidebar.selectbox[1].select("All").run()
    assert not at.exception, at.exception


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
