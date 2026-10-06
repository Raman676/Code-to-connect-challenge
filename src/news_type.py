"""
Rule-based news-type classifier: 10 categories, keyword rules, first match wins.

Run:  python src/news_type.py               (category counts over all walk-forward headlines,
                                             and a 100-row hand-check file docs/news_type_check.csv)
      python src/news_type.py --accuracy    (accuracy from the filled-in hand-check file)

Rules are checked in this order (case-insensitive, whole words):
  1 Credit Event, 2 Analyst Rating, 3 Earnings, 4 Merger/Acquisition, 5 Legal/Regulatory,
  6 Management Change, 7 Product Launch, 8 Macroeconomic, 9 Geopolitical, 10 Other.
So "debt downgrade" is a Credit Event, while a plain "downgrade" is an Analyst Rating.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_CSV = PROJECT_ROOT / "data" / "processed" / "signals_walkforward.csv"
CHECK_CSV = PROJECT_ROOT / "docs" / "news_type_check.csv"
CHECK_ROWS, CHECK_SEED = 100, 42
OTHER = "Other"

# category -> keywords and phrases (each matched as whole words, in any letter case)
RULES = {
    "Credit Event": ["default", "bankruptcy", "chapter 11", "credit rating", "debt downgrade", "junk",
                     "moody's", "fitch"],
    "Analyst Rating": ["upgrade", "upgrades", "upgraded", "downgrade", "downgrades", "downgraded",
                       "price target", "initiates", "reiterates", "maintains", "overweight", "underweight",
                       "outperform", "underperform", "buy rating", "sell rating", "neutral rating"],
    "Earnings": ["eps", "earnings", "q1", "q2", "q3", "q4", "quarter", "revenue", "sales", "guidance", "beats",
                 "misses", "estimates", "outlook", "profit"],
    "Merger/Acquisition": ["acquire", "acquires", "acquired", "acquisition", "merger", "merge", "buyout",
                           "takeover", "stake"],
    "Legal/Regulatory": ["lawsuit", "sue", "sues", "sued", "sec", "ftc", "doj", "antitrust", "probe",
                         "investigation", "fine", "fined", "settlement", "regulator", "recall"],
    "Management Change": ["ceo", "cfo", "chairman", "resign", "resigns", "steps down", "appoint", "appoints",
                          "appointed", "hires"],
    "Product Launch": ["launch", "launches", "unveil", "unveils", "introduce", "introduces", "release",
                       "releases", "rollout", "new product"],
    "Macroeconomic": ["fed", "interest rate", "inflation", "gdp", "jobs report", "unemployment", "recession",
                      "treasury"],
    "Geopolitical": ["tariff", "trade war", "china", "sanction", "brexit", "war", "election"],
}
CATEGORIES = list(RULES) + [OTHER]


def _pattern(words) -> re.Pattern:
    # Whole words: not preceded or followed by a letter/digit. Spaces in phrases match any whitespace.
    alternatives = sorted((re.escape(w).replace(r"\ ", r"\s+") for w in words), key=len, reverse=True)
    return re.compile(rf"(?<![\w])(?:{'|'.join(alternatives)})(?![\w])", re.IGNORECASE)


_PATTERNS = [(category, _pattern(words)) for category, words in RULES.items()]


def classify(text) -> str:
    """The first category (in rule order) with a matching keyword, else 'Other'."""
    if not isinstance(text, str):
        return OTHER
    for category, pattern in _PATTERNS:
        if pattern.search(text):
            return category
    return OTHER


def add_news_type(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    """Copy of df with a news_type column classified from df[col]."""
    return df.assign(news_type=df[col].map(classify))


def write_check_file(df: pd.DataFrame, path: Path = CHECK_CSV, n: int = CHECK_ROWS, seed: int = CHECK_SEED) -> pd.DataFrame:
    """Random rows for a hand check. Fill 'correct' with 'y' if the type is right, else the right category."""
    sample = df.sample(n, random_state=seed)[["date", "ticker", "text", "news_type"]].assign(correct="")
    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False, encoding="utf-8")
    return sample


def check_accuracy(path: Path = CHECK_CSV) -> dict:
    """Accuracy over the rows filled in by hand ('y'/'yes' = correct, a category name = the right answer)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    filled = df[df["correct"].str.strip() != ""]
    if filled.empty:
        return {"checked": 0, "accuracy": None}
    answer = filled["correct"].str.strip()
    right = answer.str.lower().isin(["y", "yes", "1", "true"]) | (answer.str.lower() == filled["news_type"].str.lower())
    unknown = sorted(set(answer[~right & ~answer.isin(CATEGORIES)]))
    per_type = right.groupby(filled["news_type"]).agg(["size", "mean"]).rename(columns={"size": "checked", "mean": "accuracy"})
    return {"checked": len(filled), "accuracy": float(right.mean()), "per_type": per_type,
            "unrecognised_answers": unknown}


def main() -> int:
    if "--accuracy" in sys.argv[1:]:
        result = check_accuracy()
        if not result["checked"]:
            print(f"No rows filled in yet in {CHECK_CSV.relative_to(PROJECT_ROOT)}")
            return 0
        print(f"Checked {result['checked']} rows; accuracy {result['accuracy']:.3f}")
        print(result["per_type"].round(3).to_string())
        if result["unrecognised_answers"]:
            print(f"Answers that are not 'y' or a category name: {result['unrecognised_answers']}")
        return 0
    df = add_news_type(pd.read_csv(SIGNALS_CSV))
    counts = df["news_type"].value_counts().reindex(CATEGORIES, fill_value=0)
    print(f"News types over {len(df):,} walk-forward headlines (2016 - Jun 2020):")
    for category, n in counts.items():
        print(f"  {category:20} {n:7,}  {100 * n / len(df):5.1f}%")
    write_check_file(df)
    print(f"\nHand-check file: {CHECK_CSV.relative_to(PROJECT_ROOT)} ({CHECK_ROWS} rows, seed {CHECK_SEED}); "
          f"fill 'correct' with y or the right category, then run: python src/news_type.py --accuracy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
