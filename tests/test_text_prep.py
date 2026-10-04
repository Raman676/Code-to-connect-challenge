"""
Tests for src/text_prep.py.

Run:  python tests/test_text_prep.py     or     pytest tests/test_text_prep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_prep import cap_per_company_day, mask_text  # noqa: E402

CASES = {
    # URLs and mentions
    "Check this https://t.co/abc123 out": "Check this [URL] out",
    "RT @Nike: new drop www.nike.com/x": "RT [COMPANY]: new drop [URL]",
    "truncated link https…": "truncated link [URL]",
    # cashtags and tickers
    "$AAPL and $nflx up, $5 tip": "[COMPANY] and [COMPANY] up, $5 tip",
    "AAPL beats; GOOG falls": "[COMPANY] beats; [COMPANY] falls",
    "Proha Plc (NYSE: KO) said": "Proha Plc ([COMPANY]) said",
    "AT&T (T) cuts dividend": "[COMPANY] ([COMPANY]) cuts dividend",
    "The T-shirt costs $5 at TJX": "The T-shirt costs $5 at TJX",
    # names, case rules, possessives
    "Nike's ad and NIKE's stock": "[COMPANY]'s ad and [COMPANY]'s stock",
    "Apple unveils phone": "[COMPANY] unveils phone",
    "I ate an apple pie": "I ate an apple pie",
    "APPLE IS DOWN": "[COMPANY] IS DOWN",
    "my travel visa expired": "my travel visa expired",
    "Bank of America upgrades Wells Fargo": "[COMPANY] upgrades [COMPANY]",
    "McDonald’s and Coca-Cola": "[COMPANY] and [COMPANY]",
    # subsidiaries and CEOs
    "Instagram and WhatsApp outage": "[COMPANY] and [COMPANY] outage",
    "Zuckerberg testifies; Tim Cook smiles": "[COMPANY] testifies; [COMPANY] smiles",
    "a good cook": "a good cook",
    # placeholders are not re-masked
    "[COMPANY] COST cut": "[COMPANY] [COMPANY] cut",
}


def test_mask_text_cases():
    for raw, expected in CASES.items():
        assert mask_text(raw) == expected, (raw, mask_text(raw))


def test_mask_text_handles_missing():
    assert mask_text(None) == "" and mask_text(float("nan")) == ""


def test_cap_per_company_day():
    df = pd.DataFrame({"ticker": ["A"] * 30 + ["B"] * 5, "anchor_date": ["2018-01-02"] * 35, "x": range(35)})
    capped = cap_per_company_day(df, n=20, seed=0)
    assert capped.groupby("ticker").size().to_dict() == {"A": 20, "B": 5}
    assert capped.index.is_monotonic_increasing
    assert capped.equals(cap_per_company_day(df, n=20, seed=0))      # deterministic


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
