"""
Loading and preprocessing of the Financial PhraseBank (Malo et al., 2014)
for the sentiment component of the AI/NLP Risk Engine.

Source: Kaggle "Sentiment Analysis for Financial News"
(sentiment-analysis-for-financial-news.zip), stored in data/phrasebank/:
  - Sentences_{50,66,75,All}Agree.txt  ("sentence@label")
  - all-data.csv  (same sentences as Sentences_50Agree.txt, "label,sentence")

Data quirks handled here (all verified against the raw files):
  * Files are Latin-1 encoded; all-data.csv uses bare CR line endings.
  * Non-ASCII names are mojibake: UTF-8 bytes were decoded as CP437 and the
    box-drawing lead char replaced by '+', e.g. "Seppälä" -> "Sepp+ñl+ñ".
    This is reversible for the TXT files (see `fix_mojibake`). all-data.csv was
    mangled differently, so the TXT files are treated as canonical.
  * The agreement files are nested (All ⊂ 75 ⊂ 66 ⊂ 50), so each sentence's
    annotator agreement can be recovered and used as a label-confidence weight.
  * 8 duplicate sentences in the 50% file, 2 of which carry conflicting labels.
  * Text is pre-tokenised (" , ", " 's", "( 19 May )"), unlike live news/tweets,
    so `normalize_text` detokenises it to match inference-time text.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FPB_DIR = PROJECT_ROOT / "data" / "phrasebank"
ALL_DATA_CSV = FPB_DIR / "all-data.csv"

LABELS = ("negative", "neutral", "positive")
LABEL_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}

# Agreement level -> file, ordered from strictest to loosest.
AGREEMENT_FILES = {
    1.00: "Sentences_AllAgree.txt",
    0.75: "Sentences_75Agree.txt",
    0.66: "Sentences_66Agree.txt",
    0.50: "Sentences_50Agree.txt",
}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Pairs where the second byte was itself a box-drawing char (or a 3-byte UTF-8
# sequence), so the generic CP437 reversal can't recover them.
_MOJIBAKE_OVERRIDES = {
    "+óEUR TM": "'",   # right single quote (E2 80 99)
    "+®": "é",         # Télécom
    "+Â": "ö",         # Partihallsförbindelsen
    "+â ": "ä",        # Varpaisjärvi
}
_MOJIBAKE_RE = re.compile(r"\+([\x80-\xff])")


def _reverse_cp437(match: re.Match) -> str:
    try:
        return (b"\xc3" + match.group(1).encode("cp437")).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return match.group(0)


def fix_mojibake(text: str) -> str:
    """Repair '+X' mojibake in Financial PhraseBank text (e.g. 'Sepp+ñl+ñ' -> 'Seppälä')."""
    if "+" not in text:
        return text
    for bad, good in _MOJIBAKE_OVERRIDES.items():
        text = text.replace(bad, good)
    return _MOJIBAKE_RE.sub(_reverse_cp437, text)


_DETOKENIZE_RULES = [
    (re.compile(r"``\s*|\s*''"), '"'),
    (re.compile(r'\s+([,.;:!?%)\]}])(?=[\s"]|$)'), r"\1"),  # " ," -> ","
    (re.compile(r"([(\[{])\s+"), r"\1"),                   # "( " -> "("
    (re.compile(r"\s+('s|'re|'ve|'ll|'d|n't)\b"), r"\1"),  # "company 's" -> "company's"
    (re.compile(r"\s{2,}"), " "),
]


def normalize_text(text: str) -> str:
    """Detokenise PhraseBank-style text so it resembles raw news/tweet text."""
    text = text.strip()
    for pattern, repl in _DETOKENIZE_RULES:
        text = pattern.sub(repl, text)
    return text.strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def read_phrasebank_file(path: Path) -> pd.DataFrame:
    """Read one 'sentence@label' file. Splits on the last '@' since sentences may contain '@'."""
    rows = []
    with open(path, encoding="latin-1") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            sentence, sep, label = line.rpartition("@")
            label = label.strip().lower()
            if not sep or label not in LABELS:
                raise ValueError(f"{path.name}:{lineno}: unparseable line: {line[:80]!r}")
            rows.append((sentence.strip(), label))
    return pd.DataFrame(rows, columns=["text_raw", "label"])


def read_all_data_csv(path: Path = ALL_DATA_CSV) -> pd.DataFrame:
    """Read all-data.csv (no header, Latin-1, CR line endings). Kept for cross-checking only."""
    df = pd.read_csv(path, header=None, names=["label", "text_raw"], encoding="latin-1")
    df["text_raw"] = df["text_raw"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()
    return df[["text_raw", "label"]]


def load_phrasebank(fpb_dir: Path = FPB_DIR, min_agreement: float = 0.50) -> pd.DataFrame:
    """
    Load the Financial PhraseBank as a clean, deduplicated DataFrame.

    Columns: doc_id, source, text, text_raw, label, sentiment_score, agreement
      - text:            mojibake-fixed, detokenised text (model input)
      - sentiment_score: -1.0 / 0.0 / +1.0
      - agreement:       strictest annotator-agreement level the sentence appears at
    """
    base = read_phrasebank_file(fpb_dir / AGREEMENT_FILES[0.50])

    # Assign each sentence the strictest agreement file it belongs to.
    base["agreement"] = 0.50
    for level in sorted(AGREEMENT_FILES, reverse=True):
        if level == 0.50:
            continue
        members = set(read_phrasebank_file(fpb_dir / AGREEMENT_FILES[level])["text_raw"])
        mask = base["text_raw"].isin(members) & (base["agreement"] == 0.50)
        base.loc[mask, "agreement"] = level

    base["text"] = base["text_raw"].map(fix_mojibake).map(normalize_text)

    # Drop sentences annotated with conflicting labels, then exact duplicates.
    n_labels = base.groupby("text")["label"].transform("nunique")
    base = base[n_labels == 1].drop_duplicates(subset="text").copy()

    base = base[base["agreement"] >= min_agreement].reset_index(drop=True)
    base["sentiment_score"] = base["label"].map(LABEL_TO_SCORE)
    base["source"] = "news"
    base["doc_id"] = [f"fpb_{i:05d}" for i in range(len(base))]
    return base[["doc_id", "source", "text", "text_raw", "label", "sentiment_score", "agreement"]]


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    stratify_col: str = "label",
) -> pd.DataFrame:
    """Add a 'split' column (train/val/test), stratified by label. Assumes df is deduplicated."""
    out = df.copy()
    out["split"] = "train"
    for _, group in out.groupby(stratify_col):
        shuffled = group.sample(frac=1.0, random_state=seed).index
        n_test = round(len(shuffled) * test_frac)
        n_val = round(len(shuffled) * val_frac)
        out.loc[shuffled[:n_test], "split"] = "test"
        out.loc[shuffled[n_test:n_test + n_val], "split"] = "val"
    return out
