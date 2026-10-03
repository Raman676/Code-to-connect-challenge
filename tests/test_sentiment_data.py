"""
Smoke tests for loading and processing the Financial PhraseBank.

Run:  python tests/test_sentiment_data.py        (plain script, prints a report)
 or:  pytest tests/test_sentiment_data.py         (if pytest is installed)

Also writes the processed dataset to data/processed/fpb_sentiment.csv.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sentiment_data as sd  # noqa: E402

PROCESSED_CSV = sd.PROJECT_ROOT / "data" / "processed" / "fpb_sentiment.csv"
EXPECTED_ROWS = {0.50: 4846, 0.66: 4217, 0.75: 3453, 1.00: 2264}


# ---------------------------------------------------------------------------
# Raw files
# ---------------------------------------------------------------------------

def test_raw_files_parse():
    for level, fname in sd.AGREEMENT_FILES.items():
        df = sd.read_phrasebank_file(sd.FPB_DIR / fname)
        assert len(df) == EXPECTED_ROWS[level], (fname, len(df))
        assert set(df["label"]) == set(sd.LABELS)
        assert df["text_raw"].str.len().min() > 0


def test_agreement_files_are_nested():
    sets = {lvl: set(sd.read_phrasebank_file(sd.FPB_DIR / f)["text_raw"])
            for lvl, f in sd.AGREEMENT_FILES.items()}
    assert sets[1.00] <= sets[0.75] <= sets[0.66] <= sets[0.50]


def test_all_data_csv_matches_50agree():
    csv = sd.read_all_data_csv()
    txt = sd.read_phrasebank_file(sd.FPB_DIR / sd.AGREEMENT_FILES[0.50])
    assert len(csv) == len(txt)
    assert csv["label"].value_counts().to_dict() == txt["label"].value_counts().to_dict()
    # Every pure-ASCII TXT sentence appears verbatim in the CSV; only accented rows differ
    # (the CSV mangled them differently, sometimes replacing bytes with '_').
    ascii_txt = set(txt.loc[txt["text_raw"].map(str.isascii), "text_raw"])
    assert ascii_txt <= set(csv["text_raw"])


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def test_fix_mojibake():
    cases = {
        "Clothing retail chain Sepp+ñl+ñ 's sales": "Clothing retail chain Seppälä 's sales",
        "Finnish Bank of +àland 's": "Finnish Bank of Åland 's",
        "Lule+Ñ municipality": "Luleå municipality",
        "the +ä+ñnekoski mill": "the Äänekoski mill",
        "Vladimir Brand+úo": "Vladimir Brandão",
        "France T+®l+®com": "France Télécom",
        "No mojibake here + there": "No mojibake here + there",
    }
    for raw, expected in cases.items():
        assert sd.fix_mojibake(raw) == expected, (raw, sd.fix_mojibake(raw))


def test_normalize_text():
    cases = {
        "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn , up 5.2 % .":
            "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn, up 5.2%.",
        "Proha Plc ( Euronext :7327 ) announced today ( 19 May ) that":
            "Proha Plc (Euronext :7327) announced today (19 May) that",
        "The Group 's business is balanced .": "The Group's business is balanced.",
        "`` That 's a very high figure , '' Noop said .":
            '"That\'s a very high figure," Noop said.',
    }
    for raw, expected in cases.items():
        assert sd.normalize_text(raw) == expected, (raw, sd.normalize_text(raw))


# ---------------------------------------------------------------------------
# Processed dataset
# ---------------------------------------------------------------------------

def test_load_phrasebank():
    df = sd.load_phrasebank()
    assert df["doc_id"].is_unique
    assert df["text"].is_unique
    assert not df.isna().any().any()
    assert set(df["sentiment_score"]) == {-1.0, 0.0, 1.0}
    assert (df["sentiment_score"] == df["label"].map(sd.LABEL_TO_SCORE)).all()
    assert set(df["agreement"]) == set(sd.AGREEMENT_FILES)
    # 4846 raw - 8 duplicate rows (incl. 2 conflicting pairs dropped entirely) ≈ 4836
    assert 4830 <= len(df) <= 4840, len(df)
    # Only a handful of unrecoverable mojibake chars should remain.
    leftover = df["text"].str.contains(r"\+[\x80-\xff]").sum()
    assert leftover <= 5, leftover

    strict = sd.load_phrasebank(min_agreement=1.0)
    assert (strict["agreement"] == 1.0).all()
    assert 2250 <= len(strict) <= 2264


def test_stratified_split():
    df = sd.stratified_split(sd.load_phrasebank())
    sizes = df["split"].value_counts(normalize=True)
    assert abs(sizes["train"] - 0.70) < 0.01
    assert abs(sizes["val"] - 0.15) < 0.01
    assert abs(sizes["test"] - 0.15) < 0.01
    # Label mix is preserved in every split.
    overall = df["label"].value_counts(normalize=True)
    for _, part in df.groupby("split"):
        dist = part["label"].value_counts(normalize=True)
        assert ((dist - overall).abs() < 0.01).all()
    # No text leaks across splits.
    texts = {s: set(p["text"]) for s, p in df.groupby("split")}
    assert not (texts["train"] & texts["test"]) and not (texts["train"] & texts["val"])
    # Deterministic for a fixed seed.
    assert (sd.stratified_split(sd.load_phrasebank())["split"] == df["split"]).all()


# ---------------------------------------------------------------------------
# Learnability smoke test: a dependency-free Naive Bayes baseline.
# Not the engine's model — just proves the processed data is model-ready and
# shows how class probabilities map onto the -1..1 Sentiment Score.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z]+|[+-]?\d+(?:\.\d+)?%?")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class NaiveBayes:
    def fit(self, texts, labels, alpha: float = 1.0):
        self.alpha = alpha
        self.counts = {c: Counter() for c in sd.LABELS}
        prior = Counter(labels)
        for text, label in zip(texts, labels):
            self.counts[label].update(_tokens(text))
        self.vocab = set().union(*self.counts.values())
        self.totals = {c: sum(cnt.values()) for c, cnt in self.counts.items()}
        self.log_prior = {c: math.log(prior[c] / len(labels)) for c in sd.LABELS}
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        v = len(self.vocab)
        logp = {}
        for c in sd.LABELS:
            denom = self.totals[c] + self.alpha * v
            logp[c] = self.log_prior[c] + sum(
                math.log((self.counts[c][t] + self.alpha) / denom)
                for t in _tokens(text) if t in self.vocab)
        m = max(logp.values())
        exp = {c: math.exp(lp - m) for c, lp in logp.items()}
        z = sum(exp.values())
        return {c: e / z for c, e in exp.items()}

    def sentiment_score(self, text: str) -> float:
        p = self.predict_proba(text)
        return p["positive"] - p["negative"]


def _macro_f1(y_true, y_pred) -> float:
    f1s = []
    for c in sd.LABELS:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def evaluate_baseline(df):
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]
    nb = NaiveBayes().fit(train["text"].tolist(), train["label"].tolist())
    preds = [max(p, key=p.get) for p in map(nb.predict_proba, test["text"])]
    acc = sum(p == t for p, t in zip(preds, test["label"])) / len(test)
    majority = test["label"].value_counts(normalize=True).max()
    return nb, acc, _macro_f1(test["label"].tolist(), preds), majority


def test_baseline_beats_majority_class():
    _, acc, macro_f1, majority = evaluate_baseline(sd.stratified_split(sd.load_phrasebank()))
    assert acc > majority + 0.05, (acc, majority)
    assert macro_f1 > 0.5, macro_f1


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failures = 0
    print("Running data tests")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    df = sd.stratified_split(sd.load_phrasebank())
    print(f"\nProcessed dataset: {len(df)} sentences")
    print("  label mix:   ", df["label"].value_counts().to_dict())
    print("  agreement:   ", df["agreement"].value_counts().sort_index().to_dict())
    print("  splits:      ", df["split"].value_counts().to_dict())
    print("  text length: ", df["text"].str.split().str.len().describe()[["mean", "50%", "max"]]
          .round(1).to_dict(), "(words)")

    print("\nSample rows (raw -> processed):")
    for _, row in df[df["text_raw"] != df["text"]].sample(3, random_state=1).iterrows():
        print(f"  [{row['label']:>8}] {row['text_raw'][:90]}")
        print(f"  {'':>10} {row['text'][:90]}")

    nb, acc, macro_f1, majority = evaluate_baseline(df)
    print(f"\nNaive Bayes baseline (test set): accuracy={acc:.3f}  macro-F1={macro_f1:.3f}  "
          f"(majority-class accuracy={majority:.3f})")
    for text in ["Operating profit rose 35% and the company raised its full-year guidance.",
                 "The company issued a profit warning and announced 500 layoffs.",
                 "The annual general meeting will be held in Helsinki on 15 March."]:
        print(f"  score={nb.sentiment_score(text):+.2f}  {text}")

    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {PROCESSED_CSV.relative_to(sd.PROJECT_ROOT)}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
