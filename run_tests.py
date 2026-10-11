"""
Run every test file in tests/ and print one PASS/FAIL line per file, then the total.

Run:  python run_tests.py

Each tests/test_*.py file is run as its own script (python tests/test_x.py), in a separate
process, so one file cannot affect another. A file passes when its script exits with code 0.
This script exits with code 1 if any file fails, so it can be used in CI.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    files = sorted((ROOT / "tests").glob("test_*.py"))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}          # Windows: print non-ASCII text safely
    failed = []
    start = time.time()
    for path in files:
        t0 = time.time()
        run = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        lines = run.stdout.strip().splitlines()
        result = lines[-1] if lines else "no output"        # e.g. "8/8 tests passed"
        status = "PASS" if run.returncode == 0 else "FAIL"
        print(f"{status}  {path.name:<32} {result:<22} ({time.time() - t0:.1f}s)")
        if run.returncode != 0:
            failed.append(path.name)
            print("      " + "\n      ".join((run.stdout + run.stderr).strip().splitlines()[-15:]))
    print(f"\n{len(files) - len(failed)}/{len(files)} test files passed in {time.time() - start:.0f}s")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
