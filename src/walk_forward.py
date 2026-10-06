"""
Walk-forward signals: out-of-sample sentiment and severity for every S&P 100 headline, 2016 - Jun 2020.

Run:  python src/walk_forward.py
      -> data/processed/signals_walkforward.csv, docs/walkforward_report.txt
      python src/walk_forward.py --filter-existing
      (only apply drop_reused_ticker_headlines to an existing signals file)
      (finbert-tone probabilities cached in data/interim/finbert_tone.npz)

For each test year Y (2016..2020):
  training window: news items whose label window ends before Y starts
                   (anchor at least GAP_DAYS + 1 trading days before Y's first trading day)
  tuning:          the last ~12 months of that window are validation (with their own gap);
                   sentiment C by impact macro-F1, severity alpha by Spearman
  final models:    refit on the whole training window
  scoring:         EVERY deduplicated S&P 100 headline dated in Y (no label filters, no price needed)

Sentiment: 6-class sign-pattern target (labels.py rules), TF-IDF + logistic regression;
           sentiment_score = P(10,6,4) - P(7,5,1).
Severity:  ridge on log|r1| with TF-IDF, trailing 10-day volatility and that company's headline
           count on the previous trading day; severity 1-10 = decile of the prediction among that
           year's TRAINING predictions.
Variant:   finbert-tone's own classifier, tone_score = P(positive) - P(negative) (no training).
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
import index_data as idx  # noqa: E402
import labels as lb  # noqa: E402
import modeling as md  # noqa: E402
import severity as sv  # noqa: E402
from inspect_data import NEWS_FILES, Tee, indent, read_chunks, section  # noqa: E402
from text_prep import mask_text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNALS_CSV = PROJECT_ROOT / "data" / "processed" / "signals_walkforward.csv"
REPORT_PATH = PROJECT_ROOT / "docs" / "walkforward_report.txt"
TONE_CACHE = PROJECT_ROOT / "data" / "interim" / "finbert_tone.npz"

TEST_YEARS = [2016, 2017, 2018, 2019, 2020]
GAP_DAYS = 10                 # label windows reach 10 trading days past the anchor
VAL_DAYS = 252                # ~12 months of trading days used for tuning
SEVERITY_NUMERIC = ["log_vol_before", "log_news_prev"]
CLASS_COLS = [f"p_{c}" for c in md.IMPACT_CLASSES]
# Tickers reused by a different company: headlines before this date are about the old company.
# SNDK: SanDisk (acquired by Western Digital in 2016) vs today's Sandisk (spun off in 2025).
REUSED_TICKERS = {"SNDK": "2025-01-01"}


def drop_reused_ticker_headlines(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Remove headlines filed under a ticker before its current company existed (see REUSED_TICKERS)."""
    dates = pd.to_datetime(df[date_col])
    old = pd.Series(False, index=df.index)
    for ticker, since in REUSED_TICKERS.items():
        old |= (df["ticker"] == ticker) & (dates < pd.Timestamp(since))
    return df[~old]


# ---------------------------------------------------------------------------
# Items, prices, features and labels
# ---------------------------------------------------------------------------

def symbol_map(universe: pd.DataFrame) -> dict:
    """News symbol -> index ticker. Alphabet's two share classes become one company (GOOGL)."""
    mapping = {}
    for ticker, symbols in zip(universe["ticker"], universe["news_symbols"]):
        for sym in symbols:
            mapping[sym] = "GOOGL" if ticker == "GOOG" else ticker
    return mapping


def load_news_items(universe: pd.DataFrame, log: lb.StepLog) -> pd.DataFrame:
    mapping = symbol_map(universe)
    frames, n_raw = [], 0
    for chunk in read_chunks(NEWS_FILES["analyst_ratings_processed"]):
        n_raw += len(chunk)
        chunk = chunk.rename(columns={"Unnamed: 0": "row_id", "title": "text", "date": "item_time", "stock": "symbol"})
        chunk["ticker"] = chunk["symbol"].map(mapping)
        frames.append(chunk[chunk["ticker"].notna() & chunk["text"].notna()])
    log.add("news", "raw rows (analyst_ratings_processed.csv)", n_raw)
    df = pd.concat(frames, ignore_index=True)
    df["row_id"] = df["row_id"].astype(int)
    df["local_time"] = pd.to_datetime(df["item_time"].str[:19], format="%Y-%m-%d %H:%M:%S")
    df["item_date"] = df["local_time"].dt.normalize()
    log.add("news", "S&P 100 tickers (old symbols mapped, GOOG -> GOOGL)", len(df))
    df = drop_reused_ticker_headlines(df, "item_date")
    log.add("news", "drop headlines of reused tickers (old SanDisk under SNDK)", len(df))
    df = lb.drop_duplicate_texts(df, "local_time")
    log.add("news", "drop duplicate headline+ticker (keep earliest)", len(df))
    df["item_id"] = "nw_" + df["row_id"].astype(str)
    return df.reset_index(drop=True)


def load_ticker_prices(universe: pd.DataFrame) -> pd.DataFrame:
    """Adjusted closes with index tickers as columns (BRK-B -> BRK.B; GOOG dropped, GOOGL kept)."""
    _, adj = idx.load_prices()
    to_ticker = dict(zip(universe["yf_symbol"], universe["ticker"]))
    adj = adj.rename(columns=to_ticker)
    return adj.drop(columns=[c for c in ["GOOG"] if c in adj.columns])


def tradable_returns(item_dates: pd.Series, tickers: pd.Series, prices: pd.DataFrame) -> np.ndarray:
    """Return earned when trading at the close of the first trading day AFTER the item's date and
    holding one day (close t+1 -> close t+2), matching the backtest's timing."""
    calendar = prices.index
    trade = calendar.searchsorted(item_dates.to_numpy(), side="right")
    col = prices.columns.get_indexer(tickers)
    p = prices.to_numpy()
    out = np.full(len(trade), np.nan)
    ok = (col >= 0) & (trade + 1 < len(calendar))
    out[ok] = p[trade[ok] + 1, col[ok]] / p[trade[ok], col[ok]] - 1
    return out


def add_features_and_labels(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    calendar = prices.index
    df = df.copy()
    df["anchor_pos"] = lb.anchor_date_only(df["item_date"], calendar)
    df = df.join(lb.market_features(df["ticker"], df["anchor_pos"].to_numpy(), prices))
    # Sentiment label: same rules as labels.py (price window, volatility filter, pattern table)
    returns = [f"r{k}" for k in lb.HORIZONS]
    full_window = df[returns + ["vol_before", "vol_after"]].notna().all(axis=1)
    pattern = lb.sign_pattern(df)
    keep = full_window & (df["vol_after"] > df["vol_before"]) & pattern.isin(lb.IMPACT_TABLE)
    df["impact_class"] = pattern.map(lb.IMPACT_TABLE).where(keep)
    df["sentiment_label"] = np.where(df["r1"] > 0, 1, -1)
    df.loc[df["impact_class"].isna(), "sentiment_label"] = 0          # 0 = unlabelled
    # Severity label and features (known before the item)
    df["abs_r1"] = df["r1"].abs()
    df["y_sev"] = np.log(df["abs_r1"] + sv.EPS)
    df["log_vol_before"] = np.log(df["vol_before"] + sv.EPS)
    df["log_news_prev"] = np.log1p(previous_day_counts(df))
    df["tradable_r"] = tradable_returns(df["item_date"], df["ticker"], prices)
    df["masked"] = df["text"].map(mask_text)
    return df


def previous_day_counts(df: pd.DataFrame) -> np.ndarray:
    """That company's headlines anchored to the previous trading day (published before the item)."""
    counts = df.groupby(["ticker", "anchor_pos"]).size()
    key = pd.MultiIndex.from_arrays([df["ticker"].to_numpy(), df["anchor_pos"].to_numpy() - 1])
    return counts.reindex(key).fillna(0).to_numpy()


# ---------------------------------------------------------------------------
# Yearly windows
# ---------------------------------------------------------------------------

def year_windows(anchor_pos: pd.Series, item_dates: pd.Series, calendar: pd.DatetimeIndex, year: int) -> dict:
    """Boolean masks: fit (training minus validation), val, train (= fit + gap + val), test (items dated in year)."""
    first_test = calendar.searchsorted(pd.Timestamp(f"{year}-01-01"))
    last_train = first_test - GAP_DAYS - 1                    # anchor + 10 < first test day
    val_start = last_train - VAL_DAYS + 1
    last_fit = val_start - GAP_DAYS - 1
    a = anchor_pos.to_numpy()
    in_history = a >= 0
    return {
        "train": in_history & (a <= last_train),
        "fit": in_history & (a <= last_fit),
        "val": (a >= val_start) & (a <= last_train),
        "test": (item_dates.dt.year == year).to_numpy(),
        "first_test_day": calendar[first_test],
        "last_train_anchor": calendar[last_train],
    }


def run_year(df: pd.DataFrame, w: dict) -> tuple[pd.DataFrame, dict]:
    """Fit and score one test year. Returns (scored rows, model info for the report)."""
    labelled = df["impact_class"].notna().to_numpy()
    sev_ok = df[["y_sev", "log_vol_before"]].notna().all(axis=1).to_numpy()

    # Sentiment: tune C on fit/val, refit on the whole training window
    fit, val, train = w["fit"] & labelled, w["val"] & labelled, w["train"] & labelled
    C, _ = md.tune_C(md.make_tfidf_lr, df.loc[fit, "masked"], df.loc[fit, "impact_class"].astype(int),
                     df.loc[val, "masked"], df.loc[val, "impact_class"].astype(int))
    sent_model = md.make_tfidf_lr(C).fit(df.loc[train, "masked"], df.loc[train, "impact_class"].astype(int))

    # Severity: tune alpha by Spearman, refit, cut-points from training predictions
    fit_s, val_s, train_s = w["fit"] & sev_ok, w["val"] & sev_ok, w["train"] & sev_ok
    scores = {a: sv.spearman(sv.make_ridge(a, SEVERITY_NUMERIC, True).fit(df[fit_s], df.loc[fit_s, "y_sev"])
                             .predict(df[val_s]), df.loc[val_s, "y_sev"]) for a in sv.ALPHA_GRID}
    alpha = max(scores, key=scores.get)
    sev_model = sv.make_ridge(alpha, SEVERITY_NUMERIC, True).fit(df[train_s], df.loc[train_s, "y_sev"])
    cutpoints = sv.decile_cutpoints(sev_model.predict(df[train_s]))

    # Score every headline dated in the test year
    test = df[w["test"]].copy()
    vol_median = df.loc[train_s, "log_vol_before"].median()
    test["vol_imputed"] = test["log_vol_before"].isna()            # e.g. too little price history
    test["log_vol_before"] = test["log_vol_before"].fillna(vol_median)
    proba = sent_model.predict_proba(test["masked"])
    pred = md.predictions(proba, sent_model.classes_)
    for i, c in enumerate(sent_model.classes_):
        test[f"p_{c}"] = proba[:, i]
    test["sentiment_score"] = pred["sentiment_score"].to_numpy()
    test["pred_log_abs_r1"] = sev_model.predict(test)
    test["severity"] = sv.to_decile(test["pred_log_abs_r1"], cutpoints)
    info = {"C": C, "alpha": alpha, "sent_train_rows": int(train.sum()), "sev_train_rows": int(train_s.sum()),
            "val_alpha_spearman": scores[alpha], "first_test_day": w["first_test_day"].date(),
            "last_train_anchor": w["last_train_anchor"].date()}
    return test, info


# ---------------------------------------------------------------------------
# finbert-tone variant
# ---------------------------------------------------------------------------

class ToneScorer:
    """finbert-tone class probabilities in the order (neutral, positive, negative)."""
    def __init__(self):
        from transformers import BertForSequenceClassification
        import model_checks as mc
        self.mc = mc
        self.tokenizer = md.load_finbert_tokenizer()
        self.model = BertForSequenceClassification.from_pretrained(md.FINBERT_NAME).eval()

    def __call__(self, texts) -> np.ndarray:
        probs = self.mc.tone_probabilities(self.tokenizer, self.model, list(texts))
        return probs[["neutral", "positive", "negative"]].to_numpy(dtype=np.float32)


def add_tone(scored: pd.DataFrame) -> pd.DataFrame:
    probs = md.embed_with_cache(scored["text"].tolist(), TONE_CACHE, ToneScorer,
                                log=lambda m: print(m, file=sys.__stdout__, flush=True))
    scored = scored.copy()
    scored["tone_neutral"], scored["tone_pos"], scored["tone_neg"] = probs[:, 0], probs[:, 1], probs[:, 2]
    scored["tone_score"] = scored["tone_pos"] - scored["tone_neg"]
    return scored


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def sentiment_checks(rows: pd.DataFrame, score_col: str) -> dict:
    """Accuracy vs that year's market labels (labelled rows) and correlation with the tradable return."""
    lab = rows[rows["sentiment_label"] != 0]
    pred = np.where(lab[score_col] >= 0, 1, -1)
    days = (lab.assign(s=lab[score_col]).groupby(["ticker", "anchor_pos"])
            .agg(s=("s", "mean"), actual=("sentiment_label", "first")))
    trad = rows[rows["tradable_r"].notna()]
    q = pd.qcut(trad[score_col].rank(method="first"), 5, labels=False)
    return {"labelled": len(lab), "acc": (pred == lab["sentiment_label"]).mean(),
            "day_acc": (np.where(days["s"] >= 0, 1, -1) == days["actual"]).mean(),
            "spearman_tradable": sv.spearman(trad[score_col], trad["tradable_r"]),
            "top-bottom quintile tradable (bp)": 1e4 * (trad["tradable_r"][q == 4].mean() - trad["tradable_r"][q == 0].mean())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filter_existing_output() -> int:
    """Apply drop_reused_ticker_headlines to an existing signals file without re-running the models."""
    df = pd.read_csv(SIGNALS_CSV)
    kept = drop_reused_ticker_headlines(df)
    kept.to_csv(SIGNALS_CSV, index=False, float_format="%.5f", encoding="utf-8")
    print(f"{SIGNALS_CSV.relative_to(PROJECT_ROOT)}: {len(df):,} -> {len(kept):,} rows "
          f"(dropped {len(df) - len(kept)} headlines of reused tickers {list(REUSED_TICKERS)})")
    return 0


def main() -> int:
    if "--filter-existing" in sys.argv[1:]:
        return filter_existing_output()
    buffer = io.StringIO()
    log = lb.StepLog()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Walk-forward report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/walk_forward.py")
        print(f"Test years {TEST_YEARS}; gap {GAP_DAYS} trading days; validation {VAL_DAYS} trading days; "
              f"sentiment C grid {md.C_GRID}; severity alpha grid {sv.ALPHA_GRID}")

        universe = idx.load_universe()
        prices = load_ticker_prices(universe)
        print("  loading news and building labels ...", file=sys.__stdout__, flush=True)
        df = add_features_and_labels(load_news_items(universe, log), prices)

        section("ITEMS")
        print(indent(log.table("news").to_string(index=False)))
        by_year = df.groupby(df["item_date"].dt.year).agg(
            headlines=("item_id", "size"), companies=("ticker", "nunique"),
            sentiment_labelled=("sentiment_label", lambda s: int((s != 0).sum())),
            severity_labelled=("y_sev", lambda s: int(s.notna().sum())))
        by_year.index.name = "year"
        print("\n" + indent(by_year.reset_index().to_string(index=False)))

        scored, infos = [], []
        for year in TEST_YEARS:
            print(f"  test year {year} ...", file=sys.__stdout__, flush=True)
            w = year_windows(df["anchor_pos"], df["item_date"], prices.index, year)
            rows, info = run_year(df, w)
            rows["test_year"] = year
            scored.append(rows)
            infos.append({"test_year": year, **info, "scored": len(rows)})
        scored = pd.concat(scored, ignore_index=True)
        print("  finbert-tone ...", file=sys.__stdout__, flush=True)
        scored = add_tone(scored)

        section("MODELS PER TEST YEAR")
        print(indent(pd.DataFrame(infos).to_string(index=False)))

        section("SENTIMENT vs THAT YEAR'S MARKET LABELS, and vs the TRADABLE next-day return")
        print("acc / day_acc: sign of the score vs the same-day market label (labelled rows only; chance = 0.5).")
        print("spearman_tradable: score vs the return from the close after the item's date to the next close.")
        rows = []
        for year, g in scored.groupby("test_year"):
            for name, col in (("tfidf_lr", "sentiment_score"), ("finbert-tone", "tone_score")):
                rows.append({"test_year": year, "model": name, **sentiment_checks(g, col)})
        for name, col in (("tfidf_lr", "sentiment_score"), ("finbert-tone", "tone_score")):
            rows.append({"test_year": "all", "model": name, **sentiment_checks(scored, col)})
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            print(indent(pd.DataFrame(rows).to_string(index=False)))

        section("SEVERITY vs ACTUAL |r1| (rows with prices)")
        sev = []
        for year, g in list(scored.groupby("test_year")) + [("all", scored)]:
            g = g[g["abs_r1"].notna()]
            days = g.groupby(["ticker", "anchor_pos"]).agg(p=("pred_log_abs_r1", "mean"), a=("abs_r1", "first"))
            sev.append({"test_year": year, "rows": len(g), "spearman": sv.spearman(g["pred_log_abs_r1"], g["abs_r1"]),
                        "company-day spearman": sv.spearman(days["p"], days["a"]),
                        "mean |r1| % in severity 1": 100 * g.loc[g["severity"] == 1, "abs_r1"].mean(),
                        "mean |r1| % in severity 10": 100 * g.loc[g["severity"] == 10, "abs_r1"].mean()})
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            print(indent(pd.DataFrame(sev).to_string(index=False)))
        print(f"\nSeverity distribution (all test years): "
              f"{ {int(k): int(v) for k, v in scored['severity'].value_counts().sort_index().items()} }")
        print(f"Rows with imputed volatility (too little price history): {int(scored['vol_imputed'].sum())}")

        section("COVERAGE: scored headlines per company (2016 - Jun 2020)")
        per_company = scored.groupby("ticker").size().sort_values(ascending=False)
        print(f"{len(per_company)} companies; median {per_company.median():.0f} headlines; "
              f"top 10 {per_company.head(10).to_dict()}")
        print(f"Fewest: {per_company.tail(15).to_dict()}")

        out_cols = ["item_id", "item_date", "ticker", "text", "test_year", *CLASS_COLS, "sentiment_score",
                    "pred_log_abs_r1", "severity", "vol_imputed", "tone_neutral", "tone_pos", "tone_neg", "tone_score"]
        out = scored[out_cols].rename(columns={"item_date": "date"})
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        SIGNALS_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(SIGNALS_CSV, index=False, float_format="%.5f", encoding="utf-8")
        print(f"\nOutput: {SIGNALS_CSV.relative_to(PROJECT_ROOT)} ({len(out):,} rows, "
              f"{SIGNALS_CSV.stat().st_size / 1e6:.1f} MB)")

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
