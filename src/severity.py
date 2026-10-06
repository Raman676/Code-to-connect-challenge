"""
Severity model: how BIG the stock's move is around a tweet / news item (impact 1-10).
Direction stays in sentiment_score (src/run_model.py).

Run:  python src/severity.py        (after src/run_model.py)
      -> docs/severity_report.txt, data/outputs/sentiment_output_test.csv

Label    |r1| = absolute 1-day return from the anchor close (same anchor as the sentiment labels:
         last close before the item's date). Model target: log(|r1| + 0.0001).
Severity actual severity = decile of |r1| on the training set's |r1| cut-points (1 = smallest moves);
         predicted severity (the 'impact' output) = decile of the prediction among the final model's
         TRAINING predictions, so all ten scores are used (same rule as src/walk_forward.py).
Items    all deduplicated pool items with valid prices (before the volatility and pattern filters);
         tweets capped at 20 per company-day.
Features known before the item: TF-IDF of the masked text, log trailing 10-day volatility, and news
         and tweet counts for the company on the PREVIOUS trading day (+ a flag for whether tweet
         data exists that day; tweets only cover 121 dates in 2017-2018).
Model    ridge regression, alpha chosen by validation Spearman. Baseline: trailing volatility alone.
Comparisons (reported, not used): same-day counts (include later items -> leak), and the
         next-day label (anchor = close ON the item's date, so the return is entirely after it).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb  # noqa: E402
import modeling as md  # noqa: E402
import prices as px  # noqa: E402
import run_model as rm  # noqa: E402
from inspect_data import Tee, indent, section  # noqa: E402
from text_prep import cap_per_company_day, mask_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "severity_report.txt"
SENTIMENT_PREDICTIONS = rm.OUTPUT_PATH                      # written by run_model.py
FINAL_OUTPUT = PROJECT_ROOT / "data" / "outputs" / "sentiment_output_test.csv"
MODEL_CHECKS = PROJECT_ROOT / "docs" / "model_checks.txt"

EPS = 1e-4
ALPHA_GRID = [1, 3, 10, 30, 100, 300, 1000]
PREV_DAY = ["log_vol_before", "log_news_prev", "log_tweets_prev", "tweet_data_prev"]
SAME_DAY = ["log_vol_before", "log_news_same", "log_tweets_same", "tweet_data_same"]

VARIANTS = {   # name -> (numeric features, use text?, target column)
    "baseline: trailing volatility": (["log_vol_before"], False, "y"),
    "volatility + prev-day counts": (PREV_DAY, False, "y"),
    "full: volatility + prev-day counts + text": (PREV_DAY, True, "y"),
}
COMPARISONS = {
    "full, SAME-day counts (leaks later items)": (SAME_DAY, True, "y"),
    "baseline, NEXT-day label": (["log_vol_before"], False, "y_next"),
    "full, NEXT-day label": (PREV_DAY, True, "y_next"),
}
MAIN_MODEL = "full: volatility + prev-day counts + text"


# ---------------------------------------------------------------------------
# Items and features
# ---------------------------------------------------------------------------

def build_items(log: lb.StepLog) -> dict:
    """All deduplicated pool items with |r1| and trailing volatility, per source."""
    prices = px.load_adj_close()
    calendar = prices.index
    frames = {}
    for source, loader in (("tweets", lb.load_tweets), ("news", lb.load_news)):
        df = loader(log)
        pos = lb.anchor_date_only(df["item_date"], calendar)
        df = df.join(lb.market_features(df["ticker"], pos, prices)[["anchor_date", "r1", "vol_before"]])
        df["anchor_pos"] = pos
        next_pos = calendar.searchsorted(df["item_date"].to_numpy(), side="right") - 1   # close ON the date
        df["r1_next"] = lb.market_features(df["ticker"], next_pos, prices)["r1"]
        df = df[(df["anchor_pos"] >= 0) & df[["r1", "vol_before"]].notna().all(axis=1)].copy()
        log.add(source, "valid prices (r1 and trailing volatility)", len(df))
        frames[source] = df
    return frames


def add_count_features(df: pd.DataFrame, news_counts: pd.Series, tweet_counts: pd.Series, tweet_days) -> pd.DataFrame:
    """
    News/tweet counts per company for the previous trading day ('prev', known before the item)
    and for the item's own company-day ('same', includes later items: comparison only).
    Counts are indexed by (ticker, anchor position); tweet counts only mean something on days
    with tweet data, so they come with a 0/1 flag.
    """
    df = df.copy()
    for when, offset in (("prev", 1), ("same", 0)):
        day = df["anchor_pos"].to_numpy() - offset
        key = pd.MultiIndex.from_arrays([df["ticker"].to_numpy(), day])
        has_tweets = np.isin(day, list(tweet_days))
        df[f"log_news_{when}"] = np.log1p(news_counts.reindex(key).fillna(0).to_numpy())
        df[f"log_tweets_{when}"] = np.log1p(np.where(has_tweets, tweet_counts.reindex(key).fillna(0).to_numpy(), 0))
        df[f"tweet_data_{when}"] = has_tweets.astype(float)
    return df


def prepare(frames: dict, log: lb.StepLog) -> tuple[dict, dict]:
    """Features, targets, cap and splits. Returns (modelling rows per source, all valid items per source)."""
    news_counts = frames["news"].groupby(["ticker", "anchor_pos"]).size()
    tweet_counts = frames["tweets"].groupby(["ticker", "anchor_pos"]).size()
    tweet_days = set(frames["tweets"]["anchor_pos"])
    rows, all_items = {}, {}
    for source, df in frames.items():
        df = add_count_features(df, news_counts, tweet_counts, tweet_days)
        df["log_vol_before"] = np.log(df["vol_before"] + EPS)
        df["abs_r1"] = df["r1"].abs()
        df["y"] = np.log(df["abs_r1"] + EPS)
        df["y_next"] = np.log(df["r1_next"].abs() + EPS)
        df["masked"] = df["text"].map(mask_text)
        all_items[source] = df
        if source == "tweets":
            df = cap_per_company_day(df, rm.TWEET_CAP, rm.SEED)
            log.add(source, f"cap {rm.TWEET_CAP} per company-day", len(df))
        df = df.assign(split=rm.assign_split(df["anchor_date"], **rm.SPLITS[source]))
        df = df[df["split"] != "gap"]
        log.add(source, "after split gaps", len(df))
        rows[source] = df
    return rows, all_items


# ---------------------------------------------------------------------------
# Model and metrics
# ---------------------------------------------------------------------------

def make_ridge(alpha: float, numeric, use_text: bool):
    parts = [("num", StandardScaler(), list(numeric))]
    if use_text:
        parts.append(("text", md.make_tfidf(), "masked"))
    return make_pipeline(ColumnTransformer(parts), Ridge(alpha=alpha))


def spearman(a, b) -> float:
    return float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)), method="spearman"))


def decile_cutpoints(values) -> np.ndarray:
    return np.quantile(np.asarray(values), np.linspace(0.1, 0.9, 9))


def to_decile(values, cutpoints) -> np.ndarray:
    """1..10: how many training cut-points the value exceeds, plus one."""
    return np.searchsorted(cutpoints, np.asarray(values), side="right") + 1


def fit_variant(df: pd.DataFrame, numeric, use_text: bool, target: str, refit_on_val: bool):
    """Tune alpha on validation (Spearman), then fit the final model. Returns (model, alpha, val Spearman)."""
    data = df[df[target].notna()]
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]
    scores = {a: spearman(make_ridge(a, numeric, use_text).fit(train, train[target]).predict(val), val[target])
              for a in ALPHA_GRID}
    alpha = max(scores, key=scores.get)
    final = pd.concat([train, val]) if refit_on_val else train
    return make_ridge(alpha, numeric, use_text).fit(final, final[target]), alpha, scores[alpha]


def evaluate(model, test: pd.DataFrame, target: str, cutpoints) -> tuple[dict, pd.DataFrame]:
    """Spearman (rows and company-days) and mean actual |r1| per predicted decile."""
    test = test[test[target].notna()]
    pred = model.predict(test)
    actual = np.exp(test[target]) - EPS
    days = (test.assign(pred=pred, actual=actual).groupby(["ticker", "anchor_pos"])
            .agg(pred=("pred", "mean"), actual=("actual", "first")))
    deciles = pd.DataFrame({
        "severity (training cut-points)": to_decile(np.exp(pred) - EPS, cutpoints),
        "decile of predictions (test)": pd.qcut(pd.Series(pred).rank(method="first"), 10, labels=range(1, 11)).astype(int),
        "actual": actual.to_numpy()})
    return {"rows": len(test), "spearman": spearman(pred, actual),
            "company-days": len(days), "day_spearman": spearman(days["pred"], days["actual"])}, deciles


def decile_table(deciles: pd.DataFrame, column: str) -> pd.DataFrame:
    table = deciles.groupby(column)["actual"].agg(rows="size", mean_abs_r1="mean", median_abs_r1="median")
    table = table.reindex(range(1, 11)).fillna({"rows": 0})
    table["rows"] = table["rows"].astype(int)
    table[["mean_abs_r1", "median_abs_r1"]] = (100 * table[["mean_abs_r1", "median_abs_r1"]]).round(2)
    return table.rename(columns={"mean_abs_r1": "mean |r1| %", "median_abs_r1": "median |r1| %"})


# ---------------------------------------------------------------------------
# Report sections copied from docs/model_checks.txt
# ---------------------------------------------------------------------------

def copy_check_section(start: str, stop: str) -> str:
    lines = MODEL_CHECKS.read_text(encoding="utf-8").splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next(k for k in range(i + 2, len(lines)) if lines[k].startswith(stop))   # i+1 is the header's own "===="
    return "\n".join(lines[i:j]).rstrip("= \n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    buffer = io.StringIO()
    log = lb.StepLog()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Severity report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/severity.py")
        print(f"Target log(|r1| + {EPS}); ridge alpha grid {ALPHA_GRID} chosen by validation Spearman; "
              f"final fit uses validation rows: {rm.REFIT_ON_VAL}")

        print("  building items ...", file=sys.__stdout__, flush=True)
        rows, all_items = prepare(build_items(log), log)

        section("ITEMS AND SPLITS")
        for source in ("tweets", "news"):
            print(f"{source.upper()}:")
            print(indent(log.table(source).to_string(index=False)))
        split_rows = []
        for source, df in rows.items():
            for split, g in df.groupby("split", sort=False):
                split_rows.append({"source": source, "split": split, "rows": len(g),
                                   "company-days": g.groupby(["ticker", "anchor_pos"]).ngroups,
                                   "anchor dates": f"{g['anchor_date'].min().date()} -> {g['anchor_date'].max().date()}",
                                   "median |r1| %": round(100 * g["abs_r1"].median(), 2),
                                   "% with tweet data prev day": round(100 * g["tweet_data_prev"].mean(), 1)})
        print("\n" + indent(pd.DataFrame(split_rows).to_string(index=False)))

        results, final_models, cutpoints, tables, pred_cutpoints = [], {}, {}, {}, {}
        for source, df in rows.items():
            train_abs = df.loc[df["split"] == "train", "abs_r1"]
            cutpoints[source] = decile_cutpoints(train_abs)
            for name, (numeric, use_text, target) in {**VARIANTS, **COMPARISONS}.items():
                print(f"  [{source}] {name} ...", file=sys.__stdout__, flush=True)
                model, alpha, val_s = fit_variant(df, numeric, use_text, target, rm.REFIT_ON_VAL[source])
                cuts = cutpoints[source] if target == "y" else decile_cutpoints(np.exp(df.loc[df["split"] == "train", target].dropna()) - EPS)
                m, deciles = evaluate(model, df[df["split"] == "test"], target, cuts)
                results.append({"source": source, "model": name, "comparison": name in COMPARISONS,
                                "alpha": alpha, "val spearman": val_s, **m})
                tables[(source, name)] = deciles
                if name == MAIN_MODEL:
                    final_models[source] = model
                    fit_rows = df[df["split"].isin(["train", "val"] if rm.REFIT_ON_VAL[source] else ["train"])]
                    pred_cutpoints[source] = decile_cutpoints(model.predict(fit_rows))
        results = pd.DataFrame(results)

        section("TRAINING DECILE CUT-POINTS OF |r1| (%)")
        for source, cuts in cutpoints.items():
            print(f"  {source}: {np.round(100 * cuts, 2).tolist()}")

        section("TEST METRICS: Spearman of predicted vs actual |r1|")
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            for source in ("tweets", "news"):
                print(f"\n{source.upper()}:")
                print(indent(results[results["source"] == source].drop(columns="source").to_string(index=False)))

        section("MEAN ACTUAL |r1| BY PREDICTED DECILE (test)")
        print("'severity' = predicted |r1| placed on the TRAINING cut-points (the 1-10 impact score written to the output);")
        print("'decile of predictions' = the test predictions split into ten equal groups (shows the ranking quality).")
        for source in ("tweets", "news"):
            for name in ("baseline: trailing volatility", MAIN_MODEL):
                d = tables[(source, name)]
                print(f"\n{source.upper()} - {name}:")
                both = decile_table(d, "severity (training cut-points)").join(
                    decile_table(d, "decile of predictions (test)"), lsuffix=" [severity]", rsuffix=" [pred decile]")
                both.index.name = "decile"
                print(indent(both.reset_index().to_string(index=False)))

        # Final output: sentiment predictions + severity from the main model
        sent = pd.read_csv(SENTIMENT_PREDICTIONS)
        out = []
        for source in ("tweets", "news"):
            s = sent[sent["source"] == source]
            items = all_items[source].set_index("item_id").loc[s["item_id"]]
            pred = final_models[source].predict(items)
            out.append(s.assign(impact=to_decile(pred, pred_cutpoints[source]),
                                actual_impact=to_decile(items["abs_r1"].to_numpy(), cutpoints[source])))
        out = pd.concat(out)[["date", "company_name", "ticker", "source", "text", "sentiment_score", "impact",
                              "actual_sentiment", "actual_impact"]]
        FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8")
        section("OUTPUT")
        print(f"{FINAL_OUTPUT.relative_to(PROJECT_ROOT)}: {len(out):,} rows (sentiment test rows from run_model.py)")
        dist = pd.crosstab(out["source"], out["impact"]).reindex(columns=range(1, 11), fill_value=0)
        print("Predicted severity (impact) distribution:")
        print(indent(dist.to_string()))
        print(f"Spearman(impact, actual_impact): "
              f"{ {s: round(spearman(g['impact'], g['actual_impact']), 3) for s, g in out.groupby('source')} }")

        section("FROM docs/model_checks.txt (computed in the previous step, unchanged)")
        print(copy_check_section("1b. FINBERT-TONE", "===="))
        print()
        print(copy_check_section("2. TOP TF-IDF TERMS", "10 test headlines predicted most POSITIVE"))

        section("LIMITATIONS")
        for line in [
            "The main label uses the same anchor as the sentiment labels, so |r1| includes the news day: text that "
            "describes the move ('shares trading lower', 'halted') can 'predict' its size. See the NEXT-day comparison.",
            "Same-day counts are reported only as a comparison: they include items published after the item and are "
            "partly caused by the move itself.",
            "Tweet counts exist only on 121 dates (2017-2018); elsewhere the flag is 0 and the count carries no information.",
            "Ridge predictions are pulled towards the average, so predicted severity on the training cut-points rarely "
            "reaches 1 or 10; the 'decile of predictions' table shows the ranking quality instead.",
            "News type (to be added) is not yet a feature.",
        ]:
            print(f"- {line}")

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
