"""
Models and metrics for the 6-class impact target.

Impact classes come from the sign pattern of the 1/2/3/7-day returns (see labels.py):
  positive first-day move: 10 (++++), 6 (+++-), 4 (++--)
  negative first-day move:  7 (--++), 5 (---+), 1 (----)
From predicted class probabilities:
  sentiment_score = P(10) + P(6) + P(4) - P(7) - P(5) - P(1)   (between -1 and 1)
  impact          = most likely class
  sentiment       = sign of sentiment_score (+1 when the score is exactly 0)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IMPACT_CLASSES = [1, 4, 5, 6, 7, 10]
POSITIVE_CLASSES = {10, 6, 4}
NEGATIVE_CLASSES = {7, 5, 1}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
SEED = 0

FINBERT_NAME = "yiyanghkust/finbert-tone"
FINBERT_MAX_TOKENS = 64


# ---------------------------------------------------------------------------
# Turning probabilities into the engine's outputs
# ---------------------------------------------------------------------------

def sentiment_score(proba: np.ndarray, classes) -> np.ndarray:
    """P(positive-first-day classes) - P(negative-first-day classes), per row."""
    classes = list(classes)
    pos = [classes.index(c) for c in classes if c in POSITIVE_CLASSES]
    neg = [classes.index(c) for c in classes if c in NEGATIVE_CLASSES]
    return proba[:, pos].sum(axis=1) - proba[:, neg].sum(axis=1)


def predictions(proba: np.ndarray, classes) -> pd.DataFrame:
    score = sentiment_score(proba, classes)
    return pd.DataFrame({
        "sentiment_score": score,
        "impact": np.asarray(classes)[proba.argmax(axis=1)],
        "sentiment": np.where(score >= 0, 1, -1),
    })


def sentiment_of_impact(impact) -> np.ndarray:
    return np.where(np.isin(impact, list(POSITIVE_CLASSES)), 1, -1)


def metrics(true_impact, pred: pd.DataFrame) -> dict:
    true_impact = np.asarray(true_impact)
    true_sent = sentiment_of_impact(true_impact)
    return {
        "sent_acc": accuracy_score(true_sent, pred["sentiment"]),
        "sent_f1": f1_score(true_sent, pred["sentiment"], average="macro", labels=[-1, 1], zero_division=0),
        "imp_acc": accuracy_score(true_impact, pred["impact"]),
        "imp_f1": f1_score(true_impact, pred["impact"], average="macro", labels=IMPACT_CLASSES, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Models: all expose fit(X, y), predict_proba(X) and classes_
# ---------------------------------------------------------------------------

class MajorityBaseline:
    """Always predicts the most common impact class in the training labels."""
    def fit(self, X, y):
        self.classes_ = np.array(IMPACT_CLASSES)
        self.majority_ = pd.Series(y).value_counts().idxmax()
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, list(self.classes_).index(self.majority_)] = 1.0
        return proba


class RandomBaseline:
    """Predicts a random impact class per row: 'uniform' (each class 1/6) or 'stratified'
    (drawn with the training class frequencies). Sentiment follows from the drawn class."""
    def __init__(self, strategy: str = "uniform", seed: int = SEED):
        self.strategy, self.seed = strategy, seed

    def fit(self, X, y):
        self.classes_ = np.array(IMPACT_CLASSES)
        if self.strategy == "uniform":
            self.freq_ = np.full(len(self.classes_), 1 / len(self.classes_))
        else:
            self.freq_ = pd.Series(y).value_counts(normalize=True).reindex(self.classes_, fill_value=0).to_numpy()
        return self

    def predict_proba(self, X):
        draws = np.random.default_rng(self.seed).choice(len(self.classes_), size=len(X), p=self.freq_)
        proba = np.zeros((len(X), len(self.classes_)))
        proba[np.arange(len(X)), draws] = 1.0
        return proba


def make_tfidf() -> TfidfVectorizer:
    """TF-IDF on words and word pairs; keeps "[company]" / "[url]" as single tokens."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
                           token_pattern=r"\[?\b\w\w+\b\]?")


def make_tfidf_lr(C: float):
    """TF-IDF + class-weighted logistic regression."""
    return make_pipeline(make_tfidf(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED))


def make_embedding_lr(C: float):
    """Standardised embeddings + class-weighted logistic regression."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED))


def tune_C(make_model, X_train, y_train, X_val, y_val, grid=C_GRID):
    """Pick C by impact macro-F1 on the validation set. Returns (best C, table of scores)."""
    rows = []
    for C in grid:
        model = make_model(C).fit(X_train, y_train)
        m = metrics(y_val, predictions(model.predict_proba(X_val), model.classes_))
        rows.append({"C": C, **m})
    table = pd.DataFrame(rows)
    return float(table.loc[table["imp_f1"].idxmax(), "C"]), table


# ---------------------------------------------------------------------------
# FinBERT embeddings (mean of the last hidden layer), cached on disk
# ---------------------------------------------------------------------------

def load_finbert_tokenizer(name: str = FINBERT_NAME):
    """Uncased WordPiece tokenizer built from the repo's vocab.txt.
    (The repo has no tokenizer.json and an old config without model_type, so transformers
    cannot build it automatically.)"""
    from huggingface_hub import hf_hub_download
    from transformers import BertTokenizerFast
    return BertTokenizerFast(vocab=hf_hub_download(name, "vocab.txt"), do_lower_case=True)


class FinbertEmbedder:
    """Mean-pooled last-layer embeddings from FinBERT-tone (no fine-tuning, CPU)."""
    def __init__(self, name: str = FINBERT_NAME, max_tokens: int = FINBERT_MAX_TOKENS, batch_size: int = 64):
        import torch
        from transformers import BertModel
        self.tokenizer = load_finbert_tokenizer(name)
        self.model = BertModel.from_pretrained(name).eval()
        if self.tokenizer.vocab_size != self.model.config.vocab_size:
            raise RuntimeError(f"tokenizer vocab {self.tokenizer.vocab_size} != model vocab {self.model.config.vocab_size}")
        self.torch, self.max_tokens, self.batch_size = torch, max_tokens, batch_size

    def __call__(self, texts) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(list(texts[i:i + self.batch_size]), padding=True, truncation=True,
                                     max_length=self.max_tokens, return_tensors="pt")
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((hidden * mask).sum(1) / mask.sum(1)).numpy())   # mean over real tokens
        return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_with_cache(texts, cache_path: Path, make_embedder, log=print) -> np.ndarray:
    """Embeddings for `texts`; only texts not yet in the cache file are computed."""
    cache = {}
    if cache_path.exists():
        data = np.load(cache_path)
        cache = dict(zip(data["keys"], data["vectors"]))
    keys = [text_key(t) for t in texts]
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
    if missing:
        log(f"  embedding {len(missing):,} new texts (cache has {len(cache):,}) ...")
        vectors = make_embedder()([t for _, t in missing])
        cache.update(zip([k for k, _ in missing], vectors))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.array(list(cache)), vectors=np.vstack(list(cache.values())).astype(np.float32))
    return np.vstack([cache[k] for k in keys])
