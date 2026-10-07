"""
Tests for src/architecture.py: the diagram is drawn, and its steps are in pipeline order.

Run:  python tests/test_architecture.py     or     pytest tests/test_architecture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import architecture as arch  # noqa: E402


def test_steps_numbered_in_order():
    numbers = [int(step[0].split(".")[0]) for step in arch.STEPS]
    assert numbers == list(range(1, 11))


def test_draw_writes_png(tmp=Path(__file__).resolve().parent / "_tmp_arch.png"):
    path = arch.draw(tmp)
    assert path.exists() and path.stat().st_size > 10_000
    path.unlink()


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
