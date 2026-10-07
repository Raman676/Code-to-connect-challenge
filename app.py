"""
Streamlit dashboard: the risk engine's signals and the sentiment-tilted index.

Run:  streamlit run app.py        (first run python src/backtest.py so the CSVs exist)

Sidebar: strategy, date range, ticker.
Tabs:    1 Index  2 Weights  3 Signals feed  4 Engine results  5 Sensitivity
Reads files only (src/dashboard_data.py); no models run inside the app.
"""
import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Sentiment-tilted index", layout="wide")


@st.cache_data
def load():
    """Read every file once; Streamlit keeps the result between clicks."""
    return {"nav": dd.load_nav(), "summary": dd.load_summary(), "weights": dd.load_weights(),
            "signals": dd.load_daily_signals(), "engine": dd.load_engine_signals(),
            "sensitivity": dd.load_sensitivity(), "severity": dd.severity_deciles(dd.SEVERITY_REPORT.read_text("utf-8"))}


data = load()
nav = data["nav"]
strategies = [s for s in nav.columns if not s.startswith("^")]

# ---- sidebar --------------------------------------------------------------------------------------------
st.sidebar.title("Controls")
strategy = st.sidebar.selectbox("Strategy", strategies, index=strategies.index("D"))
first, last = nav.index.min().date(), nav.index.max().date()
picked = st.sidebar.date_input("Date range", (first, last), min_value=first, max_value=last)
start, end = picked if len(picked) == 2 else (picked[0], last)     # while only a start is picked
tickers = ["All"] + sorted(data["engine"]["ticker"].unique())
ticker = st.sidebar.selectbox("Ticker", tickers, index=tickers.index("AAPL"))
st.sidebar.caption("News is date-only: a headline dated d moves the weights at the close of d+1.")

st.title("AI/NLP Risk Engine and sentiment-tilted Top-20 index")
tab_index, tab_weights, tab_feed, tab_engine, tab_sens = st.tabs(
    ["Index", "Weights", "Signals feed", "Engine results", "Sensitivity"])

# ---- 1 Index ----------------------------------------------------------------------------------------------
with tab_index:
    window = nav.loc[str(start):str(end)]
    window = 100 * window / window.iloc[0]                 # rebase to 100 at the chosen start date
    st.plotly_chart(px.line(window, labels={"value": "index level (base 100)", "index": ""}),
                    width="stretch")
    st.subheader("Metrics, full period (vs A-capped)")
    st.dataframe(data["summary"].T.style.format("{:.3f}", na_rep="-"), width="stretch")

# ---- 2 Weights --------------------------------------------------------------------------------------------
with tab_weights:
    wide = dd.weights_wide(data["weights"], strategy).loc[str(start):str(end)] * 100
    top = wide.mean().nlargest(10).index
    stacked = wide[top].assign(Others=wide.drop(columns=top).sum(axis=1))
    st.plotly_chart(px.area(stacked, labels={"value": "weight %", "index": ""},
                            title=f"Strategy {strategy} weights (top 10 + others)"), width="stretch")
    day = st.date_input("Weights on date", end, min_value=start, max_value=end)
    st.dataframe(dd.weights_on_date(data["weights"], strategy, day).style.format("{:.2f}"),
                 width="stretch")

# ---- 3 Signals feed ---------------------------------------------------------------------------------------
with tab_feed:
    types = st.multiselect("News type", sorted(data["engine"]["news_type"].unique()))
    rows = dd.filter_signals(data["engine"], ticker, start, end, types)
    st.caption(f"{len(rows):,} items")
    st.dataframe(rows[["date", "company_name", "source", "text", "sentiment_score", "impact", "news_type"]],
                 width="stretch", hide_index=True)
    if ticker != "All":
        s = data["signals"]
        s = s[(s["ticker"] == ticker) & (s["date"] >= str(start)) & (s["date"] <= str(end))]
        st.plotly_chart(px.bar(s, x="date", y="S", title=f"{ticker}: daily stock sentiment S "
                               "(|S| < 0.15 is treated as 0)"), width="stretch")

# ---- 4 Engine results -------------------------------------------------------------------------------------
with tab_engine:
    for name, value, note in dd.ENGINE_NUMBERS:
        st.metric(name, value, help=note)
    st.plotly_chart(px.bar(data["severity"], y="mean |r1| %",
                           title="Severity: mean actual |1-day return| per predicted decile (news test)"),
                    width="stretch")
    st.caption("Sentiment matches the same-day move but not the next day's: the news is priced in quickly.")

# ---- 5 Sensitivity ----------------------------------------------------------------------------------------
with tab_sens:
    sens = data["sensitivity"]
    st.subheader("Strategy D with tilt strength alpha = 0.3 (base), 1.0 and 3.0")
    st.dataframe(sens.style.format("{:.3f}"), width="stretch")
    st.plotly_chart(px.bar(sens.reset_index().astype({"alpha": str}), x="alpha",
                           y=["CAGR - A-capped (pp)", "total costs %"], barmode="group",
                           title="Stronger tilts raise costs, not returns"), width="stretch")
