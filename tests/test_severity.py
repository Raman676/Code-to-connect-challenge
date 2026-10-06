"""
Tests for src/severity.py on small made-up data (no real files, no network).

Run:  python tests/test_severity.py     or     pytest tests/test_severity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import severity as sv  # noqa: E402


def test_deciles():
    cuts = sv.decile_cutpoints(np.arange(1, 101))          # 1..100
    assert len(cuts) == 9
    assert sv.to_decile([1, 10, 11, 50, 95, 1000], cuts).tolist() == [1, 1, 2, 5, 10, 10]


def test_count_features_previous_and_same_day():
    items = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "anchor_pos": [11, 12, 12]})
    news_counts = pd.Series({("AAA", 10): 2, ("AAA", 11): 5, ("BBB", 12): 1}).rename_axis(["ticker", "anchor_pos"])
    tweet_counts = pd.Series({("AAA", 11): 7, ("BBB", 11): 3}).rename_axis(["ticker", "anchor_pos"])
    out = sv.add_count_features(items, news_counts, tweet_counts, tweet_days={11})
    # previous day = anchor position - 1
    assert np.allclose(np.expm1(out["log_news_prev"]), [2, 5, 0])
    assert np.allclose(np.expm1(out["log_tweets_prev"]), [0, 7, 3])      # day 10 has no tweet data
    assert out["tweet_data_prev"].tolist() == [0.0, 1.0, 1.0]
    # same day = the item's own company-day
    assert np.allclose(np.expm1(out["log_news_same"]), [5, 0, 1])
    assert out["tweet_data_same"].tolist() == [1.0, 0.0, 0.0]


def test_spearman():
    assert np.isclose(sv.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(sv.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_ridge_uses_numeric_and_text():
    rng = np.random.default_rng(0)
    n = 200
    vol = rng.normal(size=n)
    big = rng.random(n) < 0.5
    df = pd.DataFrame({"log_vol_before": vol, "masked": np.where(big, "shares halted big move", "quiet day update")})
    y = 0.5 * vol + 2.0 * big + rng.normal(scale=0.1, size=n)
    df.loc[df["masked"] == "quiet day update", "masked"] = "quiet day update ok"
    with_text = sv.make_ridge(1.0, ["log_vol_before"], True).fit(df, y)
    without = sv.make_ridge(1.0, ["log_vol_before"], False).fit(df, y)
    assert sv.spearman(with_text.predict(df), y) > sv.spearman(without.predict(df), y) + 0.2


def test_fit_variant_picks_alpha_and_respects_refit():
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"log_vol_before": rng.normal(size=n), "split": ["train"] * 200 + ["val"] * 50 + ["test"] * 50})
    df["y"] = df["log_vol_before"] + rng.normal(scale=0.5, size=n)
    model, alpha, val_s = sv.fit_variant(df, ["log_vol_before"], False, "y", refit_on_val=False)
    assert alpha in sv.ALPHA_GRID and val_s > 0.5
    metrics, deciles = sv.evaluate(model, df[df["split"] == "test"].assign(ticker="X", anchor_pos=range(50)), "y",
                                   sv.decile_cutpoints(np.exp(df["y"]) - sv.EPS))
    assert metrics["rows"] == 50 and metrics["spearman"] > 0.5
    assert sorted(deciles["decile of predictions (test)"].unique()) == list(range(1, 11))


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
