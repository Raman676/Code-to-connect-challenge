"""
Sanity checks on the sentiment/impact model (news + tweets), written to docs/model_checks.txt.

Run:  python src/model_checks.py

  1. Tokenizer sanity: run finbert-tone's own tone classifier (neutral/positive/negative) with our
     hand-built tokenizer on the PhraseBank test split; [UNK] rates; lowercasing and [CLS]/[SEP]
     handling. Then: does human-style tone (P(pos) - P(neg)) predict our market labels on news?
  2. Top TF-IDF features for news sentiment and the most confident test headlines.
  4. Same-day vs next-day labels (news): anchor = close ON the item's date, so the 1-day return is
     the next trading day, entirely after the news. Retrain TF-IDF + LR with the same split.
(Check 3, random baselines, is part of docs/model_report.txt.)
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb  # noqa: E402
import modeling as md  # noqa: E402
import prices as px  # noqa: E402
import run_model as rm  # noqa: E402
from inspect_data import Tee, indent, section  # noqa: E402
from text_prep import mask_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "model_checks.txt"
TONE_MAX_TOKENS = 128          # PhraseBank sentences are longer than headlines


# ---------------------------------------------------------------------------
# 1. Tokenizer and finbert-tone's own classifier
# ---------------------------------------------------------------------------

def unk_stats(token_ids: list[list[int]], unk_id: int) -> dict:
    """Share of word pieces that are [UNK], and share of texts with at least one."""
    n_tokens = sum(len(ids) for ids in token_ids)
    n_unk = sum(ids.count(unk_id) for ids in token_ids)
    return {"texts": len(token_ids), "word pieces": n_tokens,
            "[UNK] % of pieces": round(100 * n_unk / max(n_tokens, 1), 2),
            "% texts with [UNK]": round(100 * np.mean([unk_id in ids for ids in token_ids]), 1)}


def tone_probabilities(tokenizer, model, texts, batch_size: int = 64) -> pd.DataFrame:
    """finbert-tone class probabilities, columns named from the model's id2label."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                            max_length=TONE_MAX_TOKENS, return_tensors="pt")
            out.append(torch.softmax(model(**enc).logits, dim=-1).numpy())
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    return pd.DataFrame(np.vstack(out), columns=labels)


def check_tokenizer_and_tone(news_test: pd.DataFrame, texts_by_source: dict):
    from transformers import BertForSequenceClassification
    tokenizer = md.load_finbert_tokenizer()
    model = BertForSequenceClassification.from_pretrained(md.FINBERT_NAME).eval()
    cfg = model.config

    section("1a. TOKENIZER SANITY")
    print(f"Vocab size: tokenizer {tokenizer.vocab_size}, model {cfg.vocab_size}")
    print(f"Model labels (config.json id2label): {cfg.id2label}")
    vocab = tokenizer.get_vocab()
    upper = [t for t in vocab if any(c.isupper() for c in t) and not t.startswith("[")]
    print(f"Lowercasing: vocab has {len(upper)} pieces with capitals ({upper[:6]} ...) -> uncased vocab; "
          f"tokenizer lowercases: {tokenizer.tokenize('PROFIT Warning') == tokenizer.tokenize('profit warning')}")
    ids = tokenizer("profits are flat")["input_ids"]
    print(f"Special tokens in vocab: [PAD]={vocab['[PAD]']}, [UNK]={vocab['[UNK]']}, [CLS]={vocab['[CLS]']}, "
          f"[SEP]={vocab['[SEP]']} (non-standard ids; standard BERT uses 0/100/101/102)")
    print(f"Encoded 'profits are flat': {ids} -> {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"  starts with [CLS]: {ids[0] == vocab['[CLS]']}, ends with [SEP]: {ids[-1] == vocab['[SEP]']}, "
          f"tokenizer pad id {tokenizer.pad_token_id} == model pad id {getattr(cfg, 'pad_token_id', 0)}: "
          f"{tokenizer.pad_token_id == getattr(cfg, 'pad_token_id', 0)}")
    print("\n[UNK] rates on the model inputs (masked texts):")
    rows = []
    for source, texts in texts_by_source.items():
        enc = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        rows.append({"source": source, **unk_stats(enc, tokenizer.unk_token_id)})
        examples = [t for t, e in zip(texts, enc) if tokenizer.unk_token_id in e][:2]
        for ex in examples:
            pieces = tokenizer.convert_ids_to_tokens(tokenizer(ex, add_special_tokens=False)["input_ids"])
            print(f"  [{source}] {ex[:90]!r}\n      -> {' '.join(pieces)[:150]}")
    print(indent(pd.DataFrame(rows).to_string(index=False)))

    section("1b. FINBERT-TONE ON THE PHRASEBANK TEST SPLIT (human tone labels)")
    fpb = pd.read_csv(rm.PHRASEBANK_CSV)
    fpb = fpb[fpb["split"] == "test"].reset_index(drop=True)
    probs = tone_probabilities(tokenizer, model, fpb["text"].tolist())
    pred = probs.idxmax(axis=1)
    labels = ["negative", "neutral", "positive"]
    print(f"Sentences: {len(fpb)}; accuracy {accuracy_score(fpb['label'], pred):.3f}, "
          f"macro-F1 {f1_score(fpb['label'], pred, average='macro'):.3f}")
    cm = pd.DataFrame(confusion_matrix(fpb["label"], pred, labels=labels), index=labels, columns=labels)
    cm.index.name = "actual \\ predicted"
    print(indent(cm.to_string()))

    section("1c. DOES HUMAN-STYLE TONE PREDICT OUR MARKET LABELS? (news test set, original headlines)")
    probs = tone_probabilities(tokenizer, model, news_test["text"].tolist())
    tone = (probs["positive"] - probs["negative"]).to_numpy()
    actual = news_test["sentiment"].to_numpy()
    pred_sent = np.where(tone >= 0, 1, -1)
    tone_label = probs.idxmax(axis=1).to_numpy()
    polar = tone_label != "neutral"
    days = (news_test.assign(tone=tone).groupby(["ticker", "anchor_date"])
            .agg(tone=("tone", "mean"), actual=("sentiment", "first")))
    day_pred = np.where(days["tone"] >= 0, 1, -1)
    print(f"Rows: {len(news_test):,}; finbert-tone labels: "
          f"{ {k: int(v) for k, v in pd.Series(tone_label).value_counts().items()} }")
    print(f"sign(tone) vs market sentiment:   accuracy {accuracy_score(actual, pred_sent):.3f}, "
          f"macro-F1 {f1_score(actual, pred_sent, average='macro'):.3f}")
    print(f"  only rows tone calls pos/neg ({polar.sum():,}): accuracy {accuracy_score(actual[polar], pred_sent[polar]):.3f}")
    print(f"company-day (mean tone, {len(days):,} days): accuracy {accuracy_score(days['actual'], day_pred):.3f}, "
          f"macro-F1 {f1_score(days['actual'], day_pred, average='macro'):.3f}")
    print(f"Spearman(tone, 1-day return r1): {pd.Series(tone).corr(news_test['r1'].reset_index(drop=True), method='spearman'):.3f}")
    print("Reference, TF-IDF + LR trained on market labels (docs/model_report.txt): accuracy 0.609, company-day 0.699")


# ---------------------------------------------------------------------------
# 2. Top TF-IDF features
# ---------------------------------------------------------------------------

def sentiment_direction(coef: np.ndarray, classes) -> np.ndarray:
    """Per feature: mean coefficient over positive-first-day classes minus over negative ones."""
    classes = list(classes)
    pos = [classes.index(c) for c in classes if c in md.POSITIVE_CLASSES]
    neg = [classes.index(c) for c in classes if c in md.NEGATIVE_CLASSES]
    return coef[pos].mean(axis=0) - coef[neg].mean(axis=0)


def fit_news_tfidf(items: pd.DataFrame):
    """Same procedure as run_model: tune C on validation, refit on train + val. Returns (model, C)."""
    train = (items["split"] == "train").to_numpy()
    val = (items["split"] == "val").to_numpy()
    C, _ = md.tune_C(md.make_tfidf_lr, items.loc[train, "masked"], items.loc[train, "impact"],
                     items.loc[val, "masked"], items.loc[val, "impact"])
    fit = train | val
    return md.make_tfidf_lr(C).fit(items.loc[fit, "masked"], items.loc[fit, "impact"]), C


def check_top_features(news: pd.DataFrame):
    model, C = fit_news_tfidf(news)
    vectorizer, lr = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    weights = pd.Series(sentiment_direction(lr.coef_, lr.classes_), index=vectorizer.get_feature_names_out())
    section(f"2. TOP TF-IDF TERMS FOR NEWS SENTIMENT (news/tfidf_lr, C={C}, fit on train+val)")
    print("Weight = mean coefficient of classes 10/6/4 minus mean of classes 7/5/1.")
    top = pd.DataFrame({"most positive": weights.nlargest(30).index, "weight +": weights.nlargest(30).round(3).values,
                        "most negative": weights.nsmallest(30).index, "weight -": weights.nsmallest(30).round(3).values})
    print(indent(top.to_string(index=False)))

    test = news[news["split"] == "test"]
    pred = md.predictions(model.predict_proba(test["masked"]), model.classes_)
    pred.index = test.index
    shown = test.assign(score=pred["sentiment_score"])[["date", "ticker", "score", "sentiment", "impact", "text"]]
    shown = shown.rename(columns={"sentiment": "actual sent", "impact": "actual imp"})
    shown["date"] = shown["date"].dt.date
    shown["text"] = shown["text"].str[:95]
    with pd.option_context("display.width", 250, "display.max_colwidth", 95):
        print("\n10 test headlines predicted most POSITIVE:")
        print(indent(shown.nlargest(10, "score").round(3).to_string(index=False)))
        print("\n10 test headlines predicted most NEGATIVE:")
        print(indent(shown.nsmallest(10, "score").round(3).to_string(index=False)))
        for label, part in (("positive", shown.nlargest(10, "score")), ("negative", shown.nsmallest(10, "score"))):
            hit = (part["actual sent"] == (1 if label == "positive" else -1)).sum()
            print(f"  most {label}: {hit}/10 have that actual market sentiment")


# ---------------------------------------------------------------------------
# 4. Same-day vs next-day labels (news)
# ---------------------------------------------------------------------------

def anchor_on_date(item_dates: pd.Series, calendar: pd.DatetimeIndex) -> np.ndarray:
    """Position of the close ON the item's date (or the last close before it on weekends/holidays)."""
    return calendar.searchsorted(item_dates.to_numpy(), side="right") - 1


def prepare_news(labelled: pd.DataFrame) -> pd.DataFrame:
    df = labelled.copy()
    df["date"] = pd.to_datetime(df["item_time"].str[:10])
    df["masked"] = df["text"].map(mask_text)
    df["split"] = rm.assign_split(df["anchor_date"], **rm.SPLITS["news"])
    return df[df["split"] != "gap"]


def evaluate_tfidf(news: pd.DataFrame) -> dict:
    model, C = fit_news_tfidf(news)
    test = news[news["split"] == "test"]
    pred = md.predictions(model.predict_proba(test["masked"]), model.classes_)
    pred.index = test.index
    m = md.metrics(test["impact"], pred)
    day = rm.company_day_metrics(news, pred)
    majority = test["sentiment"].value_counts(normalize=True).max()
    return {"C": C, "train": int((news["split"] == "train").sum()), "val": int((news["split"] == "val").sum()),
            "test": len(test), "sent_acc": m["sent_acc"], "sent_f1": m["sent_f1"],
            "day_sent_acc": day["day_sent_acc"], "company-days": day["company-days"],
            "imp_acc": m["imp_acc"], "imp_f1": m["imp_f1"], "test majority-sentiment share": majority}


def check_next_day_labels(current_news: pd.DataFrame):
    prices = px.load_adj_close()
    log = lb.StepLog()
    news = lb.load_news(log)
    anchor = anchor_on_date(news["item_date"], prices.index)
    news = news.join(lb.market_features(news["ticker"], anchor, prices))
    relabelled = prepare_news(lb.apply_labels(news, "news", log))

    section("4. SAME-DAY vs NEXT-DAY LABELS (news; TF-IDF + LR, same split dates and procedure)")
    print("current : anchor = last close BEFORE the item's date -> the 1-day return includes the news day")
    print("next-day: anchor = close ON the item's date          -> the 1-day return is entirely after the news")
    print("\nNext-day relabelling, rows at each step:")
    print(indent(log.table("news").to_string(index=False)))
    results = pd.DataFrame([{"labels": "current", **evaluate_tfidf(current_news)},
                            {"labels": "next-day", **evaluate_tfidf(relabelled)}])
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print("\n" + indent(results.to_string(index=False)))
    both = current_news[["item_id", "sentiment"]].merge(relabelled[["item_id", "sentiment"]], on="item_id",
                                                        suffixes=("_current", "_next"))
    print(f"\nItems labelled under both schemes: {len(both):,}; same sentiment in "
          f"{100 * (both['sentiment_current'] == both['sentiment_next']).mean():.1f}%")


def main() -> int:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Model checks - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/model_checks.py")
        items, _ = rm.load_items()
        news = items[items["source"] == "news"].copy()
        texts = {s: items.loc[items["source"] == s, "masked"].tolist() for s in ("news", "tweets")}

        print("  check 1 ...", file=sys.__stdout__, flush=True)
        check_tokenizer_and_tone(news[news["split"] == "test"].reset_index(drop=True), texts)
        print("  check 2 ...", file=sys.__stdout__, flush=True)
        check_top_features(news)
        print("  check 4 ...", file=sys.__stdout__, flush=True)
        check_next_day_labels(news)

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
