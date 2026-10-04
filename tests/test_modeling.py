"""
Tests for src/modeling.py and the split logic in src/run_model.py (no model download needed).

Run:  python tests/test_modeling.py     or     pytest tests/test_modeling.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import modeling as md  # noqa: E402
import run_model as rm  # noqa: E402


def test_sentiment_score_and_predictions():
    classes = [1, 4, 5, 6, 7, 10]
    proba = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],     # all on 10       -> +1
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.5],     # 1 vs 10 tie     -> 0, sentiment +1
        [0.1, 0.1, 0.4, 0.1, 0.2, 0.1],     # neg 0.7, pos 0.3 -> -0.4, impact 5
    ])
    pred = md.predictions(proba, classes)
    assert np.allclose(pred["sentiment_score"], [1.0, 0.0, -0.4])
    assert pred["impact"].tolist() == [10, 1, 5]
    assert pred["sentiment"].tolist() == [1, 1, -1]


def test_sentiment_of_impact():
    assert md.sentiment_of_impact([10, 6, 4, 7, 5, 1]).tolist() == [1, 1, 1, -1, -1, -1]


def test_majority_baseline_and_metrics():
    y_train = [10, 10, 1, 6]
    model = md.MajorityBaseline().fit([None] * 4, y_train)
    pred = md.predictions(model.predict_proba([None] * 3), model.classes_)
    assert pred["impact"].tolist() == [10, 10, 10]
    m = md.metrics([10, 1, 7], pred)
    assert np.isclose(m["imp_acc"], 1 / 3) and np.isclose(m["sent_acc"], 1 / 3)
    assert 0 < m["imp_f1"] < 1


def test_tfidf_model_learns_and_tune_picks_a_C():
    texts = ["profit beats record growth"] * 20 + ["loss warning layoffs cut"] * 20
    y = [10] * 20 + [1] * 20
    best_C, table = md.tune_C(md.make_tfidf_lr, texts, y, texts, y, grid=[0.1, 1])
    assert best_C in (0.1, 1) and len(table) == 2
    model = md.make_tfidf_lr(best_C).fit(texts, y)
    pred = md.predictions(model.predict_proba(["record profit growth", "layoffs and loss"]), model.classes_)
    assert pred["impact"].tolist() == [10, 1]


def test_tfidf_keeps_placeholders_as_tokens():
    vec = md.make_tfidf_lr(1.0).steps[0][1]
    vec.set_params(min_df=1).fit(["[COMPANY] company beats [URL]"])
    assert {"[company]", "company", "[url]"} <= set(vec.vocabulary_)


class FakeEmbedder:
    calls = 0

    def __call__(self, texts):
        FakeEmbedder.calls += 1
        return np.array([[len(t), t.count("a")] for t in texts], dtype=float)


def test_embed_with_cache_only_embeds_new_texts():
    cache = Path(tempfile.mkdtemp()) / "emb.npz"
    FakeEmbedder.calls = 0
    first = md.embed_with_cache(["aa", "b", "aa"], cache, FakeEmbedder, log=lambda m: None)
    assert first.tolist() == [[2, 2], [1, 0], [2, 2]]
    again = md.embed_with_cache(["b", "aa"], cache, FakeEmbedder, log=lambda m: None)
    assert FakeEmbedder.calls == 1 and again.tolist() == [[1, 0], [2, 2]]
    md.embed_with_cache(["new"], cache, FakeEmbedder, log=lambda m: None)
    assert FakeEmbedder.calls == 2


def test_assign_split_with_gap():
    dates = pd.Series(pd.to_datetime(["2018-01-02", "2018-01-09", "2018-01-10", "2018-01-24",
                                      "2018-01-26", "2018-02-08", "2018-02-09"]))
    split = rm.assign_split(dates, "2018-01-10", "2018-01-26", gap_bdays=10)
    # 10 business days after 10 Jan = 24 Jan; after 26 Jan = 9 Feb
    assert split.tolist() == ["train", "train", "gap", "val", "gap", "gap", "test"]
    no_gap = rm.assign_split(dates, "2018-01-10", "2018-01-26", gap_bdays=0)
    assert no_gap.tolist() == ["train", "train", "val", "val", "test", "test", "test"]


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
