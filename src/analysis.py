"""
Does news sentiment predict returns beyond FFMC and price? (methodology section 25, remaining metrics)

Run:  python src/analysis.py        -> docs/analysis_report.txt   (after python src/backtest.py)

1. Daily IC        each day, Spearman rank correlation across the index members between S_eff (known at
                   that close) and the NEXT day's return. Mean, t-stat (mean / (std / sqrt(days))), hit rate
                   (share of days with IC > 0). Days where every member's S_eff is equal are skipped.
2. Fama-MacBeth    each day, regress next-day return on [1, S_eff, past 1-day return, past 5-day return]
                   across members (least squares). Average each coefficient over days; t = mean / (std / sqrt(days)).
3. News events     active return (strategy - A) earned on the day after the top 1% of days by news intensity
                   sum(|S| x I) over members, and over the COVID window 2020-02-19 -> 2020-04-30.
Market impact is not modelled.
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
import backtest as bt  # noqa: E402
import signals as sg  # noqa: E402
import universe as uv  # noqa: E402
from inspect_data import Tee, indent, section  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "analysis_report.txt"
COVID = ("2020-02-19", "2020-04-30")
TOP_SHARE = 0.01


def t_stat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else np.nan


def daily_ic(s_eff: pd.DataFrame, next_r: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Spearman correlation across members per day; NaN on days with no variation in S_eff."""
    out = {}
    for day in s_eff.index:
        m = members.loc[day]
        s, r = s_eff.loc[day, m], next_r.loc[day, m]
        ok = r.notna()
        s, r = s[ok], r[ok]
        if s.nunique() < 2 or len(s) < 3:
            continue
        out[day] = s.rank().corr(r.rank())               # Spearman = Pearson on ranks
    return pd.Series(out, dtype=float)


def fama_macbeth(y: pd.DataFrame, factors: dict[str, pd.DataFrame], members: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional regressions; returns mean coefficient, t-stat and number of days per factor."""
    names = list(factors)
    coefs = {}
    for day in y.index:
        m = members.loc[day]
        X = pd.DataFrame({n: f.loc[day, m] for n, f in factors.items()})
        target = y.loc[day, m]
        ok = X.notna().all(axis=1) & target.notna()
        X, target = X[ok], target[ok]
        if len(X) <= len(names) + 1 or (X.nunique() < 2).any():    # need variation in every factor
            continue
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        beta, *_ = np.linalg.lstsq(A, target.to_numpy(), rcond=None)
        coefs[day] = beta
    table = pd.DataFrame.from_dict(coefs, orient="index", columns=["intercept"] + names)
    return pd.DataFrame({"mean coef": table.mean(), "t-stat": table.apply(t_stat), "days": table.count()})


def top_news_days(s: pd.DataFrame, importance: pd.DataFrame, members: pd.DataFrame,
                  share: float = TOP_SHARE) -> pd.DatetimeIndex:
    """The days with the highest news intensity sum(|S| x I) over the members."""
    intensity = (s.abs() * importance).where(members, 0.0).sum(axis=1)
    n = max(1, int(round(share * len(intensity))))
    return intensity.nlargest(n).index.sort_values()


def active_on_days(navs: pd.DataFrame, days: pd.DatetimeIndex, ref: str = "A") -> pd.DataFrame:
    """Return earned from each day's close to the next (weights set on that day), minus the reference's."""
    r = navs.pct_change().shift(-1)                      # row t = return from t to t+1
    active = r.sub(r[ref], axis=0).loc[days]
    return pd.DataFrame({"mean active bp per day": 1e4 * active.mean(),
                         "sum active bp": 1e4 * active.sum(),
                         "days beating A": (active > 0).sum()})


def window_returns(navs: pd.DataFrame, start: str, end: str, ref: str = "A") -> pd.DataFrame:
    w = navs.loc[start:end]
    total = 100 * (w.iloc[-1] / w.iloc[0] - 1)
    return pd.DataFrame({"return %": total, "vs A (pp)": total - total[ref]})


def main() -> int:
    data = uv.load_market_data()
    ffmc_table = uv.ffmc(data)
    calendar = data.close.index
    reviews = uv.run_reviews(ffmc_table, uv.review_dates(calendar))
    days = calendar[(calendar >= reviews["effective"].iloc[0]) & (calendar <= pd.Timestamp(uv.LAST_DAY))]
    base = bt.daily_base_weights(ffmc_table, reviews, days)
    members = base.notna()
    tickers = base.columns
    sig = sg.build_signals(calendar, tickers)
    s_eff, importance = sig["S_eff"].loc[days, tickers], sig["I"].loc[days, tickers]

    prices = data.adj_close[tickers]
    next_r = (prices.shift(-1) / prices - 1).loc[days]          # return from t to t+1 (tradable)
    past_1 = (prices / prices.shift(1) - 1).loc[days]
    past_5 = (prices / prices.shift(5) - 1).loc[days]
    navs = pd.read_csv(bt.NAV_CSV, index_col=0, parse_dates=True).drop(columns="^OEX (price only)")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Analysis report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/analysis.py")
        print(f"Period {days[0].date()} -> {days[-1].date()}; members = the Top-20 index constituents each day; "
              "S_eff = our sentiment model, known at day t's close (news dated t-1 or earlier).")

        section("1. DAILY IC: Spearman(S_eff at t, return t -> t+1) across members")
        ic = daily_ic(s_eff, next_r, members)
        print(indent(f"days with variation in S_eff: {len(ic)} of {len(days)}\n"
                     f"mean IC {ic.mean():.4f}   t-stat {t_stat(ic):.2f}   hit rate (IC > 0) {(ic > 0).mean():.1%}\n"
                     f"by year (mean IC): " + ", ".join(f"{y}: {v:.4f}" for y, v in ic.groupby(ic.index.year).mean().items())))

        section("2. FAMA-MACBETH: next-day return on S_eff, past 1-day and past 5-day return")
        fm = fama_macbeth(next_r, {"S_eff": s_eff, "past 1-day return": past_1, "past 5-day return": past_5}, members)
        with pd.option_context("display.float_format", "{:,.5f}".format):
            print(indent(fm.to_string()))
        print(indent("Coefficients are in return units (0.001 = 0.1%). |t| > 2 would be significant at about 5%."))

        section(f"3a. TOP {TOP_SHARE:.0%} NEWS DAYS by sum(|S| x I): return earned next day, vs Strategy A")
        usable = days[:-1]                                   # the last day has no next-day return
        top = top_news_days(sig["S"].loc[usable, tickers], importance.loc[usable], members.loc[usable])
        print(indent(f"{len(top)} days: " + ", ".join(str(d.date()) for d in top)))
        with pd.option_context("display.float_format", "{:,.2f}".format):
            print(indent(active_on_days(navs, top).to_string()))

        section(f"3b. COVID WINDOW {COVID[0]} -> {COVID[1]}")
        with pd.option_context("display.float_format", "{:,.2f}".format):
            print(indent(window_returns(navs, *COVID).to_string()))

        section("NOTES")
        print(indent("Market impact is not modelled (10 bp per unit traded only).\n"
                     "IC and Fama-MacBeth use only days where S_eff differs across members; most members have no\n"
                     "news on most days (S_eff = 0), so the tests have less power than the day count suggests."))

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
