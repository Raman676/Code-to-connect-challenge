"""
Train and evaluate the sentiment/impact models, write test-set predictions and a report.

Run:  python src/run_model.py
      -> data/outputs/sentiment_predictions_test.csv, docs/model_report.txt
      (then run src/severity.py, which writes the final data/outputs/sentiment_output_test.csv)
      (FinBERT embeddings are cached in data/interim/finbert_embeddings.npz)

Experiments
  tweets   : trained on tweets only.  Final model = train only (val labels overlap the first
             test days' price moves, since tweets have no gap between splits).
  news     : trained on news only.    Final model = train + val (10-trading-day gaps between splits).
  combined : trained on both sources, scored on each test set (final = tweets train + news train/val).
Models: majority-class baseline, TF-IDF + logistic regression, FinBERT embeddings + logistic regression.
Regularisation C is chosen by impact macro-F1 on validation; the best model per source is the one
with the highest validation impact macro-F1.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modeling as md  # noqa: E402
from inspect_data import Tee, indent, section  # noqa: E402
from text_prep import cap_per_company_day, mask_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FILES = {"tweets": PROJECT_ROOT / "data" / "processed" / "labels_tweets.csv",
               "news": PROJECT_ROOT / "data" / "processed" / "labels_news.csv"}
PHRASEBANK_CSV = PROJECT_ROOT / "data" / "processed" / "fpb_sentiment.csv"
EMBEDDING_CACHE = PROJECT_ROOT / "data" / "interim" / "finbert_embeddings.npz"
# Sentiment predictions only; src/severity.py adds the impact (severity) columns and writes
# the final data/outputs/sentiment_output_test.csv.
OUTPUT_PATH = PROJECT_ROOT / "data" / "outputs" / "sentiment_predictions_test.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "model_report.txt"
TWEET_CAP = 20
SEED = 0
RANDOM_SEEDS = 100        # random baselines are averaged over this many draws

# Chronological splits by anchor date (approved). Items in the gap after a cut-off are dropped.
SPLITS = {
    "tweets": {"val_start": "2018-09-11", "test_start": "2018-09-27", "gap_bdays": 0},
    "news": {"val_start": "2018-11-09", "test_start": "2019-11-11", "gap_bdays": 10},
}
REFIT_ON_VAL = {"tweets": False, "news": True}

MODELS = {   # name -> (factory taking C, feature type, tune C?)
    "majority": (lambda C: md.MajorityBaseline(), "text", False),
    "tfidf_lr": (md.make_tfidf_lr, "text", True),
    "finbert_lr": (md.make_embedding_lr, "emb", True),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def assign_split(anchor_dates: pd.Series, val_start, test_start, gap_bdays: int) -> pd.Series:
    """'train' / 'val' / 'test' by anchor date; 'gap' for items within gap_bdays after a cut-off."""
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    gap = pd.offsets.BDay(gap_bdays)
    split = pd.Series("gap", index=anchor_dates.index)
    split[anchor_dates < val_start] = "train"
    split[(anchor_dates >= val_start + gap) & (anchor_dates < test_start)] = "val"
    split[anchor_dates >= test_start + gap] = "test"
    return split


def load_items() -> tuple[pd.DataFrame, dict]:
    """Both label files, masked, tweets capped, with a split column. Returns (items, row counts)."""
    frames, counts = [], {}
    for source, path in LABEL_FILES.items():
        df = pd.read_csv(path, parse_dates=["anchor_date"])
        counts[source] = {"labelled rows": len(df)}
        if source == "tweets":
            df = cap_per_company_day(df, TWEET_CAP, SEED)
            counts[source][f"after cap ({TWEET_CAP} per company-day)"] = len(df)
            df["date"] = pd.to_datetime(df["item_time"], format="%d/%m/%Y")
        else:
            df["date"] = pd.to_datetime(df["item_time"].str[:10])
        df["masked"] = df["text"].map(mask_text)
        df["split"] = assign_split(df["anchor_date"], **SPLITS[source])
        counts[source]["dropped in gaps"] = int((df["split"] == "gap").sum())
        frames.append(df[df["split"] != "gap"])
    items = pd.concat(frames, ignore_index=True)
    return items, counts


def split_table(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, split), g in items.groupby(["source", "split"], sort=False):
        dist = g["impact"].value_counts(normalize=True)
        rows.append({"source": source, "split": split, "rows": len(g),
                     "company-days": g.groupby(["ticker", "anchor_date"]).ngroups,
                     "anchor dates": f"{g['anchor_date'].min().date()} -> {g['anchor_date'].max().date()}",
                     **{f"imp {c} %": round(100 * dist.get(c, 0), 1) for c in md.IMPACT_CLASSES}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_experiment(name, items, emb, train, val, refit, tests: dict):
    """Tune, fit and score every model. Returns (metric rows, predictions, final models, chosen C)."""
    def X(mask, feature):
        return items.loc[mask, "masked"] if feature == "text" else emb[items.loc[mask, "emb_idx"].to_numpy()]

    def y(mask):
        return items.loc[mask, "impact"].to_numpy()

    rows, preds, models, chosen = [], {}, {}, {}
    for model_name, (make, feature, tunable) in MODELS.items():
        print(f"  [{name}] {model_name} ...", file=sys.__stdout__, flush=True)
        C = md.tune_C(make, X(train, feature), y(train), X(val, feature), y(val))[0] if tunable else None
        model = make(C).fit(X(train, feature), y(train))
        val_pred = md.predictions(model.predict_proba(X(val, feature)), model.classes_)
        rows.append({"experiment": name, "model": model_name, "C": C, "scored on": "val (both)" if name == "combined"
                     else f"val ({name})", **md.metrics(y(val), val_pred)})
        if refit.any():
            model = make(C).fit(X(train | refit, feature), y(train | refit))
        for source, test in tests.items():
            pred = md.predictions(model.predict_proba(X(test, feature)), model.classes_)
            pred.index = items.index[test]
            preds[(name, model_name, source)] = pred
            rows.append({"experiment": name, "model": model_name, "C": C, "scored on": f"test ({source})",
                         **md.metrics(y(test), pred)})
        models[model_name], chosen[model_name] = (model, feature), C

    # Random baselines: metrics averaged over RANDOM_SEEDS draws (row level and company-day level)
    day_rows = []
    for strategy in ("uniform", "stratified"):
        val_m, test_m, day_m = [], {s: [] for s in tests}, {s: [] for s in tests}
        for seed in range(RANDOM_SEEDS):
            model = md.RandomBaseline(strategy, seed).fit(None, y(train | refit))
            val_m.append(md.metrics(y(val), md.predictions(model.predict_proba(y(val)), model.classes_)))
            for source, test in tests.items():
                pred = md.predictions(model.predict_proba(y(test)), model.classes_)
                pred.index = items.index[test]
                test_m[source].append(md.metrics(y(test), pred))
                day_m[source].append(company_day_metrics(items, pred))
        model_name = f"{strategy}_random"
        rows.append({"experiment": name, "model": model_name, "C": None,
                     "scored on": "val (both)" if name == "combined" else f"val ({name})",
                     **pd.DataFrame(val_m).mean().to_dict()})
        for source in tests:
            rows.append({"experiment": name, "model": model_name, "C": None, "scored on": f"test ({source})",
                         **pd.DataFrame(test_m[source]).mean().to_dict()})
            day_rows.append({"experiment": name, "model": model_name, "source": source,
                             **pd.DataFrame(day_m[source]).mean().to_dict()})
    return rows, preds, models, chosen, day_rows


def company_day_metrics(items: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Average the sentiment score per (ticker, anchor date) and compare its sign with the day's label."""
    df = items.loc[pred.index, ["ticker", "anchor_date", "sentiment"]].assign(score=pred["sentiment_score"])
    days = df.groupby(["ticker", "anchor_date"]).agg(score=("score", "mean"), actual=("sentiment", "first"))
    predicted = np.where(days["score"] >= 0, 1, -1)
    from sklearn.metrics import accuracy_score, f1_score
    return {"company-days": len(days), "day_sent_acc": accuracy_score(days["actual"], predicted),
            "day_sent_f1": f1_score(days["actual"], predicted, average="macro", labels=[-1, 1], zero_division=0)}


def phrasebank_check(model, feature, fpb: pd.DataFrame, fpb_emb: np.ndarray) -> tuple[pd.DataFrame, dict]:
    X = fpb["masked"] if feature == "text" else fpb_emb
    pred = md.predictions(model.predict_proba(X), model.classes_)
    fpb = fpb.assign(score=pred["sentiment_score"].to_numpy(), pred_sent=pred["sentiment"].to_numpy())
    by_label = fpb.groupby("label").agg(sentences=("score", "size"), mean_score=("score", "mean"),
                                        share_predicted_positive=("pred_sent", lambda s: (s == 1).mean()))
    polar = fpb[fpb["label"] != "neutral"]
    truth = np.where(polar["label"] == "positive", 1, -1)
    from sklearn.metrics import accuracy_score, f1_score
    summary = {"pos/neg sentences": len(polar), "accuracy": accuracy_score(truth, polar["pred_sent"]),
               "macro_f1": f1_score(truth, polar["pred_sent"], average="macro"),
               "spearman(score, human -1/0/+1)": fpb["score"].corr(fpb["sentiment_score"], method="spearman")}
    return by_label.round(3), summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(df: pd.DataFrame) -> str:
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        return indent(df.to_string(index=False))


def main() -> int:
    import sklearn
    import torch
    import transformers

    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Model report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/run_model.py")
        print(f"scikit-learn {sklearn.__version__}, torch {torch.__version__}, transformers {transformers.__version__}; "
              f"FinBERT: {md.FINBERT_NAME} (mean-pooled last layer, max {md.FINBERT_MAX_TOKENS} tokens, no fine-tuning)")
        print(f"C grid {md.C_GRID}; selection by validation impact macro-F1; seed {SEED}")

        items, counts = load_items()
        fpb = pd.read_csv(PHRASEBANK_CSV)
        fpb = fpb[fpb["split"] == "test"].reset_index(drop=True)
        fpb["masked"] = fpb["text"].map(mask_text)

        # FinBERT embeddings for every text (cached), one row per item
        texts = items["masked"].tolist() + fpb["masked"].tolist()
        emb_all = md.embed_with_cache(texts, EMBEDDING_CACHE, md.FinbertEmbedder,
                                      log=lambda m: print(m, file=sys.__stdout__, flush=True))
        emb, fpb_emb = emb_all[:len(items)], emb_all[len(items):]
        items["emb_idx"] = np.arange(len(items))

        section("DATA AND SPLITS")
        for source, c in counts.items():
            print(f"{source}: {c}")
        print(f"Tweet splits: {SPLITS['tweets']}; news splits: {SPLITS['news']} (gap in business days)")
        print(f"Final fit uses validation rows: {REFIT_ON_VAL}")
        print(fmt(split_table(items)))
        print("\nMasking examples (original -> model input):")
        for _, r in items.groupby("source").sample(3, random_state=SEED).iterrows():
            print(f"  [{r['source']}] {r['text'][:110]}\n  {'':>{len(r['source']) + 2}} {r['masked'][:110]}")

        src, spl = items["source"], items["split"]
        is_ = lambda s, p: ((src == s) & (spl == p)).to_numpy()   # noqa: E731
        none = np.zeros(len(items), dtype=bool)
        experiments = {
            "tweets": dict(train=is_("tweets", "train"), val=is_("tweets", "val"), refit=none,
                           tests={"tweets": is_("tweets", "test")}),
            "news": dict(train=is_("news", "train"), val=is_("news", "val"), refit=is_("news", "val"),
                         tests={"news": is_("news", "test")}),
            "combined": dict(train=(spl == "train").to_numpy(), val=(spl == "val").to_numpy(),
                             refit=is_("news", "val"), tests={"tweets": is_("tweets", "test"), "news": is_("news", "test")}),
        }
        all_rows, all_preds, all_models, chosen_C, random_day_rows = [], {}, {}, {}, []
        for name, exp in experiments.items():
            rows, preds, models, chosen, day_rows = run_experiment(name, items, emb, **exp)
            all_rows += rows
            all_preds.update(preds)
            all_models[name], chosen_C[name] = models, chosen
            random_day_rows += day_rows
        results = pd.DataFrame(all_rows)

        section("VALIDATION METRICS (used to choose C and the best model)")
        print(fmt(results[results["scored on"].str.startswith("val")]))
        print(f"(uniform_random / stratified_random: averages over {RANDOM_SEEDS} random draws)")
        best = {}
        for source in ("tweets", "news"):
            val = results[(results["experiment"] == source) & results["scored on"].str.startswith("val")
                          & results["model"].isin(MODELS)]
            best[source] = val.loc[val["imp_f1"].idxmax(), "model"]
        print(f"\nBest model per source (highest validation impact macro-F1): {best}")

        section("TEST METRICS (sentiment and impact, vs majority baseline)")
        test = results[results["scored on"].str.startswith("test")].copy()
        for source in ("tweets", "news"):
            part = test[test["scored on"] == f"test ({source})"].copy()
            base = part[part["model"] == "majority"].set_index("experiment")
            for m in ("sent_acc", "sent_f1", "imp_acc", "imp_f1"):
                part[f"{m} vs base"] = part[m] - part["experiment"].map(base[m])
            print(f"\n{source.upper()} test set ({int(((src == source) & (spl == 'test')).sum()):,} rows):")
            print(fmt(part.drop(columns=["scored on"])))

        section("COMPANY-DAY LEVEL (test): mean sentiment score per company-day vs that day's label")
        day_rows = [{"experiment": e, "model": m, "source": s, **company_day_metrics(items, p)}
                    for (e, m, s), p in all_preds.items()]
        day_table = pd.DataFrame(day_rows + random_day_rows).sort_values(["source", "experiment"], kind="stable")
        print(fmt(day_table))

        section("PREDICTED vs ACTUAL IMPACT CLASSES (test, % of rows)")
        for source in ("tweets", "news"):
            mask = (src == source) & (spl == "test")
            dist = {"actual": items.loc[mask, "impact"].value_counts(normalize=True)}
            for (e, m, s), p in all_preds.items():
                if s == source:
                    dist[f"{e}/{m}"] = p["impact"].value_counts(normalize=True)
            table = (100 * pd.DataFrame(dist).reindex(md.IMPACT_CLASSES).fillna(0)).round(1)
            table.index.name = "impact"
            print(f"\n{source.upper()}:")
            print(indent(table.reset_index().to_string(index=False)))

        section("CONFUSION MATRIX of the best model per source (test; rows = actual, columns = predicted)")
        for source in ("tweets", "news"):
            pred = all_preds[(source, best[source], source)]
            actual = items.loc[pred.index, "impact"]
            cm = pd.crosstab(actual, pred["impact"]).reindex(index=md.IMPACT_CLASSES, columns=md.IMPACT_CLASSES, fill_value=0)
            print(f"\n{source.upper()} ({best[source]}):")
            print(indent(cm.to_string()))

        section("PHRASEBANK CHECK (human-labelled tone, test split) scored by the best news model")
        pb_name = best["news"] if best["news"] != "majority" else "tfidf_lr"
        model, feature = all_models["news"][pb_name]
        by_label, summary = phrasebank_check(model, feature, fpb, fpb_emb)
        print(f"Model: news/{pb_name}")
        print(indent(by_label.reset_index().to_string(index=False)))
        print(f"\nPositive vs negative sentences: { {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()} }")

        # Output: best per-source model's predictions on each test set
        out = []
        for source in ("tweets", "news"):
            pred = all_preds[(source, best[source], source)]
            rows = items.loc[pred.index]
            out.append(pd.DataFrame({
                "item_id": rows["item_id"], "date": rows["date"].dt.strftime("%Y-%m-%d"),
                "company_name": rows["company"], "ticker": rows["ticker"], "source": source, "text": rows["text"],
                "sentiment_score": pred["sentiment_score"].round(4), "actual_sentiment": rows["sentiment"]}))
        out = pd.concat(out, ignore_index=True)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

        section("LIMITATIONS")
        for line in [
            "Labels come from the market reaction, not from reading the text: every item about a company on one day "
            "shares that day's label, so the text often carries little information about it.",
            "Tweets have no gap between splits: a story spanning a cut-off (e.g. Nike's Kaepernick ad, early Sept 2018) "
            "can appear in train and validation.",
            "The tweet test period (27 Sep - 30 Oct 2018) was a market sell-off: impact 1 is far more common than in "
            "training, so results there partly measure a regime change.",
            "Product names (iPhone, Windows, Prime, Air Jordan, Xbox games ...) are not masked and still identify companies; "
            "masking of names/tickers is rule-based and can miss variants or misspellings.",
            "All @mentions are masked as [COMPANY], including mentions of ordinary users.",
            "The 10-day news gap uses business days (exchange holidays not removed).",
            "FinBERT is used frozen (no fine-tuning); the PhraseBank has a neutral class that our 6-class model cannot "
            "predict, so neutral sentences are only described.",
        ]:
            print(f"- {line}")
        print(f"\nOutput: {OUTPUT_PATH.relative_to(PROJECT_ROOT)} ({len(out):,} rows; best per-source models {best})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
