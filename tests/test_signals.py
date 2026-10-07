"""
Tests for src/signals.py (security-level sentiment and confidence).

Run:  python tests/test_signals.py     or     pytest tests/test_signals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import signals as sg  # noqa: E402

CAL = pd.bdate_range("2019-01-07", periods=30)          # starts on a Monday


def events_from(rows):
    df = pd.DataFrame(rows, columns=["date", "ticker", "score", "severity", "model_conf"])
    df["date"] = pd.to_datetime(df["date"])
    return sg.build_events(df, CAL)


def test_weekend_news_usable_on_monday():
    pos = sg.availability_position(pd.Series(pd.to_datetime(["2019-01-12", "2019-01-11"])), CAL)  # Sat, Fri
    assert CAL[pos[0]] == pd.Timestamp("2019-01-14") and CAL[pos[1]] == pd.Timestamp("2019-01-14")


def test_same_ticker_same_date_is_one_event():
    ev = events_from([("2019-01-08", "X", 0.4, 4, 0.5), ("2019-01-08", "X", 0.8, 8, 0.7),
                      ("2019-01-09", "X", 0.1, 2, 0.1)])
    assert len(ev) == 2
    first = ev.iloc[0]
    assert np.isclose(first["S"], 0.6) and np.isclose(first["I"], 0.8) and np.isclose(first["model_conf"], 0.6)


def test_decay_and_15_day_window():
    sig = sg.security_signals(events_from([("2019-01-07", "X", 0.5, 10, 1.0)]), CAL, ["X"])
    s = sig["S"]["X"]
    assert (s.iloc[1:16] == 0.5).all() and s.iloc[16] == 0 and s.iloc[0] == 0   # days 1..15 only
    # two events of opposite sign: the newer one gets more weight (half-life 3 trading days)
    sig = sg.security_signals(events_from([("2019-01-07", "X", 1.0, 10, 1.0), ("2019-01-10", "X", -1.0, 10, 1.0)]),
                              CAL, ["X"])
    t = 4                                                    # newer event dt=0, older dt=3 -> weights 1 and 0.5
    assert np.isclose(sig["S"]["X"].iloc[t], (1 * 0.5 - 1 * 1) / 1.5)


def test_confidence_formula():
    sig = sg.security_signals(events_from([("2019-01-07", "X", 0.5, 10, 0.8), ("2019-01-07", "Y", 0.5, 10, 0.8),
                                           ("2019-01-08", "Y", -0.5, 10, 0.4)]), CAL, ["X", "Y"])
    # X on day 1: one event, F=1 -> 0.8 x (1 - e^-0.5) x 1
    assert np.isclose(sig["C"]["X"].iloc[1], 0.8 * (1 - np.exp(-0.5)))
    # Y on day 2: weights 0.5^(1/3) (older) and 1 (newer)
    f = 0.5 ** (1 / 3)
    expected = ((0.8 * f + 0.4) / (f + 1)) * (1 - np.exp(-(f + 1) / 2)) * (abs(f - 1) / (f + 1))
    assert np.isclose(sig["C"]["Y"].iloc[2], expected)


def test_neutral_zone():
    sig = sg.security_signals(events_from([("2019-01-07", "X", 0.1, 5, 1.0), ("2019-01-07", "Y", -0.2, 5, 1.0)]),
                              CAL, ["X", "Y", "Z"])
    assert sig["S"]["X"].iloc[1] == 0.1 and sig["S_eff"]["X"].iloc[1] == 0
    assert sig["S_eff"]["Y"].iloc[1] == -0.2 and (sig["S_eff"]["Z"] == 0).all()


def test_add_scores_variants():
    h = pd.DataFrame({"p_1": [0.1], "p_4": [0.1], "p_5": [0.1], "p_6": [0.2], "p_7": [0.1], "p_10": [0.4],
                      "sentiment_score": [0.3], "tone_score": [-0.5], "tone_pos": [0.1], "tone_neg": [0.6]})
    main = sg.add_scores(h, "main")
    assert np.isclose(main["model_conf"][0], 0.7 - 0.3) and main["score"][0] == 0.3
    tone = sg.add_scores(h, "tone")
    assert np.isclose(tone["model_conf"][0], 0.5) and tone["score"][0] == -0.5


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
