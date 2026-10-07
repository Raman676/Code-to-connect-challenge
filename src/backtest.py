"""
Daily index backtest, Jan 2016 - Jun 2020 (methodology sections 24-25).

Run:  python src/backtest.py      -> docs/backtest_report.txt, data/outputs/backtest_nav.csv

Timing  At each trading day's close the portfolio (whose weights have drifted with prices) is traded
        to the strategy's target; 10 bps is paid on every unit of weight traded; the new weights then
        earn the next day's return (adjusted closes, so dividends are included).
        Review membership is traded at the close of the trading day after the review.
        The backtest starts fully invested at the first effective day (no cost for the initial build).
Strategies
  A          FFMC Top 20; FFMC weights at each review, then the weights drift with prices.
  A-capped   as A, with the 12% cap applied at each review.
  B          daily FFMC weights tilted by S_eff (C = 1), no risk controls.
  C          daily FFMC weights tilted by S_eff x C, no risk controls.
  D          C + tilt band +-25%, 12% cap and 1 pp daily change limit.
  D-tone     D with finbert-tone scores instead of our sentiment model.
Signals come from src/signals.py, which applies the only lag (news dated d -> weights at d+1's close).
Also writes data/outputs/backtest_results.xlsx, backtest_summary.csv and docs/figures/*.png
(src/export_results.py, src/charts.py).
Benchmarks  ^OEX (S&P 100 PRICE index: no dividends, so about 2%/year easier to beat) and Strategy A.
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
import signals as sg  # noqa: E402
import universe as uv  # noqa: E402
import weights as wt  # noqa: E402
from inspect_data import Tee, indent, section  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "docs" / "backtest_report.txt"
NAV_CSV = PROJECT_ROOT / "data" / "outputs" / "backtest_nav.csv"
COST_BPS = 10
TRADING_DAYS = 252
SENSITIVITY_ALPHAS = (0.3, 1.0, 3.0)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def simulate(adj_close: pd.DataFrame, days: pd.DatetimeIndex, target_fn, cost_bps: float = COST_BPS) -> dict:
    """
    Run one strategy. target_fn(day, drifted_weights) returns the target weights for that close, or
    None to keep the drifted weights (no trade). Returns daily NAV, returns, costs, turnover and weights.
    """
    returns = adj_close.pct_change()
    nav, held = 1.0, None
    rows, weights_log, stale = [], {}, 0
    for i, day in enumerate(days[:-1]):
        target = target_fn(day, held)
        if target is None:
            target = held
        target = target[target > 0]
        traded = 0.0 if held is None else float((target.sub(held, fill_value=0.0)).abs().sum())
        cost = traded * cost_bps / 1e4
        nav *= 1 - cost
        next_r = returns.loc[days[i + 1], target.index]
        stale += int(next_r.isna().sum())
        next_r = next_r.fillna(0.0)                            # no price that day: treat as unchanged
        port_r = float((target * next_r).sum())
        held = target * (1 + next_r) / (1 + port_r)            # weights drift with prices
        rows.append({"date": day, "nav_after_cost": nav, "cost": cost, "traded": traded})
        nav *= 1 + port_r
        weights_log[day] = target
    rows.append({"date": days[-1], "nav_after_cost": nav, "cost": 0.0, "traded": 0.0})
    daily = pd.DataFrame(rows).set_index("date")      # nav_after_cost = NAV at each close, after that day's costs
    return {"daily": daily, "weights": weights_log, "stale_prices": stale}


def review_target_fn(data: uv.MarketData, ffmc_table: pd.DataFrame, reviews: pd.DataFrame, capped: bool):
    """Strategy A / A-capped: new FFMC weights on each effective day, otherwise hold (drift)."""
    by_effective = dict(zip(reviews["effective"], reviews["members"]))

    def target(day, held):
        if day not in by_effective:
            return None
        base = wt.base_weights(ffmc_table.loc[day, by_effective[day]])
        return wt.cap_weights(base) if capped else base
    return target


def daily_base_weights(ffmc_table: pd.DataFrame, reviews: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """FFMC weights of the current members, recomputed every day (section 23). NaN = not a member."""
    effective = reviews.set_index("effective")["members"]
    ffmc_filled = ffmc_table.ffill()                       # a missing price keeps the last known FFMC
    rows = {}
    for day in days:
        members = effective[effective.index <= day].iloc[-1]
        rows[day] = wt.base_weights(ffmc_filled.loc[day, members])
    return pd.DataFrame(rows).T


def sentiment_target_fn(base_table: pd.DataFrame, s_eff: pd.DataFrame, conf: pd.DataFrame,
                        effective_days, mode: str, relaxed_days: list | None = None, alpha: float = wt.ALPHA):
    """
    Strategies B, C and D: new target weights every day.
    s_eff / conf rows are used for the same day with NO shift: signals.py already applied the one lag
    (news dated d -> row d+1), and simulate() earns the next day's return on these weights.
      B  normalise(base x exp(alpha x S_eff))            C = 1, no risk controls
      C  normalise(base x exp(alpha x S_eff x C))        no risk controls
      D  C's raw weights + tilt band, 12% cap, 1 pp daily change limit (weights.constrained_weights)
    alpha is the tilt strength (0.30 in the methodology; other values for the sensitivity test).
    """
    effective_days = set(effective_days)

    def target(day, held):
        base = base_table.loc[day].dropna()
        s = s_eff.loc[day]
        if mode == "B":
            return wt.normalise(wt.tilt(base, s, 1.0, alpha))
        raw = wt.tilt(base, s, conf.loc[day], alpha)
        if mode == "C":
            return wt.normalise(raw)
        w, relaxed = wt.constrained_weights(raw, base, held, review_day=day in effective_days)
        if relaxed and relaxed_days is not None:
            relaxed_days.append(day)
        return w
    return target


# ---------------------------------------------------------------------------
# Metrics (section 25)
# ---------------------------------------------------------------------------

def daily_rf(tbill_pct: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return from the 13-week T-bill yield (annualised %)."""
    y = tbill_pct.reindex(days).ffill() / 100
    return (1 + y) ** (1 / TRADING_DAYS) - 1


def performance(nav: pd.Series, rf: pd.Series, bench_r: pd.Series | None = None) -> dict:
    r = nav.pct_change().dropna()
    excess = r - rf.reindex(r.index).fillna(0.0)
    years = len(r) / TRADING_DAYS
    downside = excess[excess < 0]
    out = {"total return %": 100 * (nav.iloc[-1] / nav.iloc[0] - 1),
           "CAGR %": 100 * ((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
           "volatility %": 100 * r.std() * np.sqrt(TRADING_DAYS),
           "Sharpe": excess.mean() / r.std() * np.sqrt(TRADING_DAYS),
           "Sortino": excess.mean() / np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS),
           "max drawdown %": 100 * (nav / nav.cummax() - 1).min()}
    if bench_r is not None:
        b = bench_r.reindex(r.index).fillna(0.0)
        active = r - b
        te = active.std() * np.sqrt(TRADING_DAYS)
        b_ex = b - rf.reindex(r.index).fillna(0.0)
        beta = np.cov(excess, b_ex)[0, 1] / b_ex.var()
        out.update({"tracking error %": 100 * te,
                    "information ratio": active.mean() * TRADING_DAYS / te if te > 0 else np.nan,
                    "beta": beta, "alpha % (annual)": 100 * (excess.mean() - beta * b_ex.mean()) * TRADING_DAYS})
    return out


def portfolio_stats(result: dict, sectors: pd.Series, reviews: pd.DataFrame) -> dict:
    daily, weights = result["daily"], result["weights"]
    years = len(daily) / TRADING_DAYS
    w = pd.DataFrame(weights).T.fillna(0.0)
    sector_w = w.T.groupby(sectors.reindex(w.columns).values).sum().T
    return {"turnover % per year (one-way)": 100 * daily["traded"].sum() / 2 / years,
            "cost bps per year": 1e4 * daily["cost"].sum() / years,
            "constituent changes": int(reviews["added"].map(len).sum()),
            "max single weight %": 100 * w.max().max(),
            "max weight holder": w.max().idxmax(),
            "avg largest sector weight %": 100 * sector_w.max(axis=1).mean(),
            "avg sector HHI": (sector_w ** 2).sum(axis=1).mean()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import charts as ch                # imported here: both modules import backtest themselves
    import export_results as ex
    data = uv.load_market_data()
    ffmc_table = uv.ffmc(data)
    calendar = data.close.index
    reviews = uv.run_reviews(ffmc_table, uv.review_dates(calendar))
    start = reviews["effective"].iloc[0]
    days = calendar[(calendar >= start) & (calendar <= pd.Timestamp(uv.LAST_DAY))]
    bench = pd.read_csv(idx.BENCHMARK_CSV, index_col="date", parse_dates=["date"])
    rf = daily_rf(bench["tbill_13w_pct"], days)
    oex = bench["sp100"].reindex(days).ffill()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, buffer)):
        print(f"Backtest report - generated {dt.datetime.now():%Y-%m-%d %H:%M} by src/backtest.py")
        print(f"Period {days[0].date()} -> {days[-1].date()} ({len(days)} trading days); costs {COST_BPS} bps "
              f"per unit traded; universe {ffmc_table.shape[1]} companies (current S&P 100, Alphabet once)")
        print("Assumptions: free-float factors are today's values, held constant (BRK.B 0.85 is our estimate); "
              "current S&P 100 members used for 2016-2020 (survivorship bias); ^OEX is a price index (no dividends);\n"
              "META and BKNG share counts back-filled with their first known count before their share history "
              "starts; LIN, RTX and DELL unranked until their share history starts.")

        section("CONSTITUENTS AT EACH REVIEW (exact-20 rule with the 20/25 buffer)")
        for _, r in reviews.iterrows():
            added = ", ".join(r["added"])
            removed = ", ".join(f"{t} (rank {k})" for t, k in r["removed_ranks"].items())
            changes = (f"  + {added}" if added else "") + (f"  - {removed}" if removed else "")
            members = " ".join(f"{t}:{r['ranks'][t]}" for t in r["members"])
            print(f"{r['review'].date()} (traded {r['effective'].date()}, {r['ranked_companies']} ranked){changes}")
            print(f"    {members}")
        print(f"\nConstituent changes over the period: {int(reviews['added'].map(len).sum())} additions, "
              f"{int(reviews['removed'].map(len).sum())} deletions over {len(reviews) - 1} reviews")

        results = {name: simulate(data.adj_close, days, review_target_fn(data, ffmc_table, reviews, capped))
                   for name, capped in (("A", False), ("A-capped", True))}
        base_table = daily_base_weights(ffmc_table, reviews, days)
        tickers = ffmc_table.columns
        sig = {v: sg.build_signals(calendar, tickers, variant=v) for v in ("main", "tone")}
        relaxed = {}
        for name, mode, v in (("B", "B", "main"), ("C", "C", "main"), ("D", "D", "main"), ("D-tone", "D", "tone")):
            relaxed[name] = []
            fn = sentiment_target_fn(base_table, sig[v]["S_eff"], sig[v]["C"], reviews["effective"], mode, relaxed[name])
            results[name] = simulate(data.adj_close, days, fn)
        navs = pd.DataFrame({name: res["daily"]["nav_after_cost"] for name, res in results.items()})
        # sensitivity: Strategy D with stronger tilts (everything else unchanged)
        sensitivity = {wt.ALPHA: results["D"]}
        for alpha in SENSITIVITY_ALPHAS:
            if alpha != wt.ALPHA:
                fn = sentiment_target_fn(base_table, sig["main"]["S_eff"], sig["main"]["C"], reviews["effective"],
                                         "D", alpha=alpha)
                sensitivity[alpha] = simulate(data.adj_close, days, fn)
        navs["^OEX (price only)"] = oex / oex.iloc[0]

        section("KEY METRICS (vs ^OEX; and A-capped vs A)")
        rows = []
        oex_r = navs["^OEX (price only)"].pct_change()
        for name in navs.columns:
            row = {"strategy": name, **performance(navs[name], rf, oex_r if name != "^OEX (price only)" else None)}
            if name in results:
                row.update(portfolio_stats(results[name], data.sectors, reviews))
            rows.append(row)
        table = pd.DataFrame(rows).set_index("strategy").T
        with pd.option_context("display.width", 200, "display.float_format", "{:,.3f}".format):
            print(indent(table.to_string()))
        vs_a = performance(navs["A-capped"], rf, navs["A"].pct_change())
        print(f"\nA-capped vs A: tracking error {vs_a['tracking error %']:.2f}%, "
              f"information ratio {vs_a['information ratio']:.2f}")
        for name, res in results.items():
            print(f"{name}: member-days without a price (return set to 0): {res['stale_prices']}")
        for name, d in relaxed.items():
            if name.startswith("D"):
                print(f"{name}: days where the 1 pp change limit had to be relaxed: {len(d)}")

        NAV_CSV.parent.mkdir(parents=True, exist_ok=True)
        navs.to_csv(NAV_CSV, float_format="%.6f")
        print(f"\nDaily NAVs saved to {NAV_CSV.relative_to(PROJECT_ROOT)}")

        summary = ex.export_all(navs, results, rf, base_table, data.sectors, reviews, sensitivity)
        section("SUMMARY vs A-capped (also in data/outputs/backtest_results.xlsx)")
        with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
            print(indent(summary.T.to_string()))
        section("SENSITIVITY: Strategy D with different alpha (also in the Excel file)")
        with pd.option_context("display.width", 250, "display.float_format", "{:,.3f}".format):
            print(indent(ex.sensitivity_table(sensitivity, results["A-capped"], rf, base_table).to_string()))
        ch.make_charts(navs, results["D"], base_table, sig["main"]["S_eff"])
        print(f"\nCharts saved to {ch.FIGURE_DIR.relative_to(PROJECT_ROOT)}")

    REPORT_PATH.write_text(buffer.getvalue(), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
