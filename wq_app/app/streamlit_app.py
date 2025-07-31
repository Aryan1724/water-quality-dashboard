
import os, json, math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Optional OpenAI (only if you set OPENAI_API_KEY)
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

from utils import (
    load_long_df, annual_median, yoy_change, exceedance_sample,
    monthly_quarterly, corr_and_leadlag, anomaly_flags, segment_stations,
    forecast_preview, scope_filter, THRESH,
    fetch_rainfall_monthly, monthly_quality, correlate_quality_rain, build_predictor_rain
)

st.set_page_config(page_title="Water Quality Dashboard — Easy + Chat (fixed)", layout="wide")
st.title("Water Quality Dashboard — Easy Mode (with Chat)")
st.caption("Plain-language labels, more explanation under charts, clean date axes, and a simple data chat.")

# ---------- Helpers ----------
def explain(md_text):
    # Avoid f-strings here so that any '{' or '}' in md_text won't break parsing.
    st.markdown(":information_source: **What this means**  \n" + str(md_text))

def note(md_text):
    st.markdown(":bulb: **Tip** — " + str(md_text))

def pretty_date_axis(ax, dates):
    if len(dates) == 0:
        return
    dmin, dmax = min(dates), max(dates)
    span_days = (dmax - dmin).days if hasattr(dmax, '__sub__') else 365
    if span_days > 3*365:
        locator = mdates.MonthLocator(interval=3)
    elif span_days > 18*30:
        locator = mdates.MonthLocator(interval=2)
    else:
        locator = mdates.MonthLocator(interval=1)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha('right')
    ax.margins(x=0)
    ax.figure.tight_layout()

def tiny_glossary():
    with st.expander("Glossary (tap to open)"):
        st.markdown(
            """
- **Median**: the middle value (less affected by extreme spikes).
- **Exceedance**: % of samples **above** the safe limit (higher = worse).
- **YoY change**: percent change vs the previous year.
- **Lag**: if rain at month **t** relates most to quality at **t+1**, lag = 1.
- **Ridge model**: a simple regression; it avoids overfitting by keeping weights small.
- **Bands**: a rough uncertainty range around a forecast (not guarantees).
            """
        )

# ---------- Sidebar ----------
import os

default_path = os.environ.get("WQ_EXCEL_PATH", str(Path.cwd().parent / "wq_project" / "data" / "DENZA 31 POINTS DATA 2020-2024 PHD.xlsx"))
excel_path = st.sidebar.text_input("Path to Excel file", value=default_path)
scope = st.sidebar.selectbox("Scope (choose 'NO_STP' to hide inlet/STP stations)", options=["ALL","NO_STP"], index=0)
param = st.sidebar.selectbox("Parameter", options=["Fecal_Coliform","Turbidity"], index=0)

@st.cache_data(show_spinner=True)
def _load_df(path):
    return load_long_df(Path(path))

if not Path(excel_path).exists():
    st.error(f"Excel not found: {excel_path}. Edit the path in the left sidebar.")
    st.stop()

df_all = _load_df(excel_path)
df = scope_filter(df_all, scope)

st.sidebar.markdown("---")
key_in = st.sidebar.text_input(
  "🔑 OpenAI  Key (optional)",
  type="password",
  help="enter"
)
if key_in:
    os.environ["OPENAI_API_KEY"] = key_in

# Precompute summary tables used in chat
ann_tbl = annual_median(df)
exc_tbl = exceedance_sample(df)
mon_tbl, qtr_tbl = monthly_quarterly(df)
corr_overall, leadlag_tbl = corr_and_leadlag(df)
an_tbl = anomaly_flags(df)
seg_tbl = segment_stations(df)

def simple_trend_text(p):
    sub = ann_tbl[ann_tbl['parameter']==p].sort_values('year')
    if len(sub) < 2:
        return "Not enough years to judge a trend."
    x = sub['year'].values
    y = sub['annual_median'].values
    m, b = np.polyfit(x, y, 1)
    direction = "upward (worse)" if m > 0 else "downward (better)"
    return f"{p.replace('_',' ')} shows a **{direction}** trend over the years. Slope ≈ {m:.2f} units per year."

def top_stations_text(p, n=5):
    sub = seg_tbl[seg_tbl['parameter']==p].dropna(subset=['exceed_pct']).copy()
    sub = sub.sort_values(['tier','exceed_pct','median'], ascending=[True, False, False])
    worst = sub.sort_values(['exceed_pct','median'], ascending=[False, False]).head(n)
    lines = [f"- {r.station}: exceed {r.exceed_pct:.1f}% | median {r.median:.1f}" for r in worst.itertuples(index=False)]
    return "\n".join(lines) if len(lines)>0 else "No station summary available."

def latest_exceed_text(p):
    if exc_tbl.empty: return "No exceedance info."
    years = sorted(exc_tbl['year'].unique())
    last = years[-1]
    sub = exc_tbl[(exc_tbl['parameter']==p) & (exc_tbl['year']==last)]
    if sub.empty: return "No exceedance info."
    val = float(sub['exceedance_rate_samples_pct'].iloc[0])
    return f"In **{last}**, ~**{val:.1f}%** of samples were above the limit for {p.replace('_',' ')}."

def anomalies_text(p):
    flagged = an_tbl[(an_tbl['parameter']==p) & (an_tbl['anomaly'])]
    return f"Flagged **{len(flagged)}** unusual months for {p.replace('_',' ')} (|robust‑z| ≥ 3.5)."

def build_chat_context():
    parts = []
    for p in ['Fecal_Coliform','Turbidity']:
        parts.append(f"### {p}\n- {simple_trend_text(p)}\n- {latest_exceed_text(p)}\n- {anomalies_text(p)}\n- Top stations (by exceedance/median):\n{top_stations_text(p)}")
    try:
        best = leadlag_tbl.iloc[leadlag_tbl['corr'].abs().argmax()]
        parts.append(f"### FC vs Turbidity\n- Overall correlation: {corr_overall:.3f}\n- Best lag: {int(best['lag_months'])} months (corr {best['corr']:.3f})")
    except Exception:
        pass
    if 'rain_df' in st.session_state:
        rain = st.session_state['rain_df']
        for p in ['Fecal_Coliform','Turbidity']:
            try:
                lagdf = correlate_quality_rain(df, p, rain, lags=(0,1,2))
                best = lagdf.iloc[lagdf['corr'].abs().argmax()]
                parts.append(f"### {p} vs Rain\n- Best lag: {int(best['lag_months'])} months (corr {float(best['corr']):.3f})")
            except Exception:
                pass
    return "\n\n".join(parts)

# ----------------- UI Tabs -----------------
tabs = st.tabs([
    "Overview","Trends","Seasonality","Exceedance","Correlations",
    "Anomalies","Segmentation","Forecast","Data Explorer","Drivers & ML","Chat"
])

with tabs[0]:
    st.subheader("Overview")
    tiny_glossary()
    st.write(f"Rows: **{len(df)}**  |  Stations: **{df['station'].nunique()}**  |  Years: **{sorted(df['year'].unique().tolist())}**")
    st.write("Thresholds (safe limits):", THRESH)
    st.dataframe(df.head(20))
    explain("- This sample shows the cleaned table used for analysis.\n- Use **Scope** to include or exclude likely STP/inlet sites.\n- Use **Parameter** to switch between Fecal Coliform and Turbidity.")

with tabs[1]:
    st.subheader("Yearly Trend (Annual Medians)")
    fig, ax = plt.subplots()
    for p in ann_tbl['parameter'].unique():
        sub = ann_tbl[ann_tbl['parameter']==p]
        ax.plot(sub['year'], sub['annual_median'], marker='o', label=p)
    ax.set_xlabel("Year"); ax.set_ylabel("Annual median"); ax.set_title(f"{scope}: Annual Medians"); ax.legend()
    st.pyplot(fig)
    explain("**Why this matters**\n- Tracks direction over years. Rising = **worse**, falling = **better**.\n**How to read**\n- Dots are yearly **medians** (robust to spikes).\n**Quick take**\n- " + simple_trend_text(param))
    st.subheader("Year over Year (YoY) Change in the Annual Median")
    yoy = yoy_change(ann_tbl); st.dataframe(yoy)
    explain("**Why**: YoY shows year‑to‑year swings.\n**Read**: Positive % = deterioration; negative % = improvement.")

with tabs[2]:
    st.subheader("Monthly & Quarterly Patterns")
    for p in mon_tbl['parameter'].unique():
        fig, ax = plt.subplots()
        for y in sorted(mon_tbl['year'].unique()):
            sub = mon_tbl[(mon_tbl['parameter']==p) & (mon_tbl['year']==y)]
            ax.plot(sub['month'], sub['monthly_median'], marker='o', label=str(y))
        ax.set_xlabel("Month (1=Jan)"); ax.set_ylabel("Monthly median"); ax.set_title(f"{p}: Monthly Medians by Year"); ax.legend()
        st.pyplot(fig)
    explain("**Why**: Repeating peaks/dips hint at climate/flow effects.\n**Read**: Compare same month across years.\n**Look for**: consistent highs in monsoon months or lows in dry months.")
    st.write("Quarterly medians by year"); st.dataframe(qtr_tbl)
    explain("Quarterly view smooths noise into Q1–Q4 for easier comparison.")

with tabs[3]:
    st.subheader("Share of Samples Above the Safe Limit (Exceedance)")
    fig, ax = plt.subplots()
    for p in exc_tbl['parameter'].unique():
        sub = exc_tbl[exc_tbl['parameter']==p]
        ax.plot(sub['year'], sub['exceedance_rate_samples_pct'], marker='o', label=p)
    ax.set_xlabel("Year"); ax.set_ylabel("% of samples above limit"); ax.set_title(f"{scope}: Exceedance Rate"); ax.legend()
    st.pyplot(fig)
    explain("**Why**: Direct view of compliance. Lower is better.\n**Quick take**: " + latest_exceed_text(param))

with tabs[4]:
    st.subheader("Do the Metrics Move Together? (FC vs Turbidity)")
    st.write(f"Overall correlation: **{corr_overall:.3f}** (1=together, -1=opposite, 0=unrelated)")
    fig, ax = plt.subplots()
    ax.bar(leadlag_tbl['lag_months'], leadlag_tbl['corr'])
    ax.set_xlabel("Lag months (positive = FC leads)"); ax.set_ylabel("Correlation")
    ax.set_title("Cross-correlation: FC vs Turbidity")
    st.pyplot(fig)
    try:
        best = leadlag_tbl.iloc[leadlag_tbl['corr'].abs().argmax()]
        best_txt = f"Best lag: {int(best['lag_months'])} months (corr {best['corr']:.3f})"
    except Exception:
        best_txt = "No clear lag."
    explain("**Why**: Helps tell if changes in one metric are followed by changes in the other.\n**Read**: Taller bars mean stronger relationship at that lag.\n**Quick take**: " + best_txt)

with tabs[5]:
    st.subheader("Unusual Points (Anomalies)")
    fig, ax = plt.subplots()
    an_tbl['robust_z_log'].dropna().hist(bins=50, ax=ax)
    ax.set_title("How unusual are the points? (Robust z on log scale)")
    st.pyplot(fig)
    st.dataframe(an_tbl[an_tbl['anomaly']].sort_values('robust_z_log', key=lambda s: s.abs(), ascending=False).head(100))
    explain("**Why**: Flags surprising spikes/dips worth investigating.\n**How**: Uses a robust score on log scale; we flag |z| ≥ 3.5.\n**Next**: Check station logs, rainfall, tourism, or operations around those months.")

with tabs[6]:
    st.subheader("Station Tiers (Low / Moderate / High Risk)")
    st.dataframe(seg_tbl.sort_values(['parameter','tier','exceed_pct'], ascending=[True, True, False]))
    explain("**Why**: Focus attention.\n**Read**: Tier uses % above limit and typical level.\n**Top concerns for** " + param.replace('_',' ') + ":\n" + top_stations_text(param))

with tabs[7]:
    st.subheader("Simple Forecast for 2025 (trend + month pattern)")
    for p in ["Fecal_Coliform","Turbidity"]:
        fc = forecast_preview(df, p, year_next=2025)
        fig, ax = plt.subplots()
        ax.plot(fc['month'], fc['forecast'], marker='o')
        ax.fill_between(fc['month'], fc['lo80'], fc['hi80'], alpha=0.2, label="~80% band")
        ax.fill_between(fc['month'], fc['lo95'], fc['hi95'], alpha=0.1, label="~95% band")
        ax.set_xlabel("Month"); ax.set_ylabel("Forecast"); ax.set_title(f"{p}: 2025 Forecast")
        ax.legend()
        st.pyplot(fig)
    explain("**Note**: This quick projection extends the trend + month pattern only. For planning, combine with drivers like rainfall, flows and tourism.")

with tabs[8]:
    st.subheader("Explore the Raw Long Table")
    st.dataframe(df)
    explain("Use this to spot‑check stations and months.")

with tabs[9]:
    st.subheader("External Drivers — Rainfall & a Small ML Model")
    st.caption("We pull monthly rainfall from Open‑Meteo for the location you enter.")

    c1, c2, c3, c4 = st.columns(4)
    lat = c1.number_input("Latitude", value=15.49, format="%.4f")
    lon = c2.number_input("Longitude", value=73.82, format="%.4f")
    start_date = c3.text_input("Start date (YYYY-MM-DD)", value="2020-01-01")
    end_date = c4.text_input("End date (YYYY-MM-DD)", value="2024-12-31")
    param2 = st.selectbox("Quality parameter to relate with rainfall", options=["Fecal_Coliform","Turbidity"], index=0, key="param2")

    if st.button("Fetch Rainfall"):
        try:
            rain = fetch_rainfall_monthly(lat, lon, start_date, end_date)
            st.session_state["rain_df"] = rain
            st.success(f"Fetched months: {len(rain)}")
            fig, ax = plt.subplots()
            ax.plot(rain["date"], rain["rain_mm"], marker='o')
            ax.set_title("Monthly Rainfall (mm)")
            ax.set_xlabel("Date"); ax.set_ylabel("mm")
            pretty_date_axis(ax, list(rain["date"]))
            st.pyplot(fig)
            explain("Shows **total rain per month** for the selected location and dates. Use this together with the overlay to check if rains line up with quality.")
            st.dataframe(rain.head(24))
        except Exception as e:
            st.error(f"Failed to fetch rainfall: {e}")

    if "rain_df" in st.session_state:
        rain = st.session_state["rain_df"]
        st.markdown("---")
        st.subheader("Do quality and rainfall move together?")
        if st.button("Compute correlations (lags 0, 1, 2)"):
            corrdf = correlate_quality_rain(df, param2, rain, lags=(0,1,2))
            fig, ax = plt.subplots()
            ax.bar(corrdf["lag_months"], corrdf["corr"])
            ax.set_xlabel("Lag months (rain leads by L)")
            ax.set_ylabel("Correlation")
            ax.set_title(f"{param2} vs Rainfall — lag correlation")
            st.pyplot(fig)
            try:
                best = corrdf.iloc[corrdf["corr"].abs().argmax()]
                best_txt = f"Best lag: {int(best['lag_months'])} months (corr {float(best['corr']):.3f})"
            except Exception:
                best_txt = "No clear lag."
            explain("**Why**: Finds the delay between rain and quality changes.\n**Read**: The tallest bar is the strongest alignment.\n**Quick take**: " + best_txt)

            qmon = monthly_quality(df, param2).merge(rain, on=["year","month"], how="inner")
            qmon["date"] = pd.to_datetime(qmon["year"].astype(str)+"-"+qmon["month"].astype(str)+"-01")
            qmon = qmon.sort_values(["year","month"])
            fig, ax1 = plt.subplots()
            ax1.plot(qmon["date"], qmon["q_median"], marker='o', label=f"{param2} (median)")
            ax1.set_xlabel("Date"); ax1.set_ylabel(f"{param2} (median)")
            ax2 = ax1.twinx()
            ax2.plot(qmon["date"], qmon["rain_mm"], marker='x', label="Rainfall (mm)")
            ax2.set_ylabel("Rainfall (mm)")
            ax1.set_title(f"{param2} vs Rainfall — overlay")
            pretty_date_axis(ax1, list(qmon["date"]))
            st.pyplot(fig)
            explain("**How to read**: Where the two lines rise/fall together, rain may be a driver. Where they diverge, look for other causes (flows, tourism, events, STP issues).")

        st.markdown("---")
        st.subheader("Small predictive model (Ridge)")
        if st.button("Train model + preview 2025"):
            res = build_predictor_rain(df, param2, rain)
            if isinstance(res, dict) and "error" in res:
                st.error(res["error"])
            else:
                st.write(f"Fit quality — RMSE (log1p): **{res['rmse']:.4f}**, R²: **{res['r2']:.3f}**, α: **{res['alpha']:.2f}**")
                explain("**What it uses**: rain this month and previous 2 months + month pattern + trend.\n**How to use**: Good for a quick sense of direction, not for exact numbers.")
                st.dataframe(res["fitted"].tail(24))

                fig, ax = plt.subplots()
                sub = res["fitted"].tail(36).copy()
                sub["date"] = pd.to_datetime(sub["date"])
                ax.plot(sub["date"], sub["q_median"], marker='o', label="Actual")
                ax.plot(sub["date"], sub["pred"], marker='x', label="Predicted")
                ax.set_title("In‑sample fit (last ~3 years)")
                ax.set_xlabel("Date"); ax.set_ylabel("Monthly median")
                ax.legend()
                pretty_date_axis(ax, list(sub["date"]))
                st.pyplot(fig)
                explain("**Read**: Closer lines = better explanation. Big gaps = other drivers at work.")

                st.write("2025 preview")
                fig, ax = plt.subplots()
                fcst = res["forecast"].copy()
                fcst["date"] = pd.to_datetime(fcst["date"])
                ax.plot(fcst["date"], fcst["forecast_q"], marker='o')
                ax.set_title("2025 monthly preview (uses rainfall climatology)")
                ax.set_xlabel("Date"); ax.set_ylabel("Predicted monthly median")
                pretty_date_axis(ax, list(fcst["date"]))
                st.pyplot(fig)
                explain("This is a **what‑if** based on typical rain. Update with real 2025 rain when you have it.")

with tabs[10]:
    st.subheader("Chat with Your Data (Simple, grounded)")
    st.caption("Ask in plain English. The bot answers using the tables/metrics above. It won’t make up numbers.")

    q = st.text_input("Ask a question (e.g., 'Which year was worst for fecal coliform?')")
    if st.button("Ask"):
        ctx = build_chat_context()
        answer = None

        if _OPENAI_OK and os.environ.get("OPENAI_API_KEY"):
            try:
                client = OpenAI()
                system = (
                    "You are a helpful assistant that explains a water-quality dataset in simple words. "
                    "Use ONLY the information in the provided CONTEXT. If the question asks for numbers, "
                    "quote numbers that are present in the context. If something is not in the context, say "
                    "'I don't have that in the data shown here.' Keep answers short and friendly."
                )
                messages = [
                    {"role":"system","content":system},
                    {"role":"assistant","content":"CONTEXT\n" + ctx},
                    {"role":"user","content":q}
                ]
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content.strip()
            except Exception as e:
                st.warning(f"OpenAI call failed, using rule-based answer. ({e})")

        if answer is None:
            ql = (q or "").lower()
            if "trend" in ql or "better" in ql or "worse" in ql:
                answer = simple_trend_text(param)
            elif "exceed" in ql or "above" in ql or "limit" in ql:
                answer = latest_exceed_text(param)
            elif "station" in ql or "location" in ql or "worst" in ql or "risk" in ql:
                answer = "Top concern stations (by exceedance/median):\n" + top_stations_text(param)
            elif "anomal" in ql or "spike" in ql or "dip" in ql:
                answer = anomalies_text(param)
            elif "rain" in ql:
                if 'rain_df' in st.session_state:
                    try:
                        lagdf = correlate_quality_rain(df, param, st.session_state['rain_df'], lags=(0,1,2))
                        best = lagdf.iloc[lagdf['corr'].abs().argmax()]
                        answer = f"Best rain lag for {param.replace('_',' ')} is {int(best['lag_months'])} months (corr {float(best['corr']):.3f})."
                    except Exception:
                        answer = "I couldn't compute the rain link yet. Try 'Fetch Rainfall' first."
                else:
                    answer = "Fetch rainfall first in the Drivers & ML tab, then ask again."
            else:
                answer = ("I can answer about **trends**, **seasonality**, **exceedance**, **anomalies**, "
                          "**best/worst stations**, and **rainfall links**. Try: "
                          "'Which stations are high‑risk?' or 'Did fecal coliform improve after 2022?'")

        st.markdown("**Answer:** " + answer)
        with st.expander("Context used"):
            st.markdown(ctx)
