from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Inflation Forecast", page_icon="📈", layout="wide")

DATA_PATH = Path("data/cpi_data.csv")
FORECAST_PATH = Path("data/yoy_forecast.csv")
MODEL_COMPARISON_PATH = Path("data/model_comparison.csv")
ROLLING_BACKTEST_PATH = Path("data/rolling_backtest.csv")

PRIMARY_COLOR = "#2563eb"
ACCENT_COLOR = "#f97316"
FED_TARGET = 2.0

st.markdown(
    f"""
    <style>
    .hero {{
        background: linear-gradient(135deg, {PRIMARY_COLOR} 0%, #1e3a8a 100%);
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p {{ margin: 0.3rem 0 0 0; opacity: 0.9; }}
    .updated-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }}
    div[data-testid="stMetric"] {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data["mom"] = data["cpi"].pct_change() * 100
    data["yoy"] = data["cpi"].pct_change(12) * 100
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    return data, forecast


@st.cache_data
def load_backtest():
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    rolling_backtest = pd.read_csv(ROLLING_BACKTEST_PATH)
    return model_comparison, rolling_backtest


data, forecast = load_data()
model_comparison, rolling_backtest = load_backtest()
latest = data.iloc[-1]
previous = data.iloc[-2]

st.markdown(
    f"""
    <div class="hero">
        <h1>📈 AI Inflation Forecast</h1>
        <p>Tracking U.S. CPI data and forecasting inflation with ARIMA models.</p>
        <span class="updated-badge">Latest data: {latest['date'].strftime('%B %Y')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Display Settings")
history_months = st.sidebar.slider(
    "History window (months)", min_value=12, max_value=len(data), value=min(60, len(data)), step=6
)
forecast_months = st.sidebar.slider(
    "Forecast window (months)", min_value=1, max_value=len(forecast), value=len(forecast), step=1
)
st.sidebar.caption("Adjust the sliders to change how much history and forecast is shown in the charts.")

csv_bytes = data.tail(history_months).to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "Download CPI data (CSV)", data=csv_bytes, file_name="cpi_data.csv", mime="text/csv"
)

tab_overview, tab_history, tab_forecast, tab_backtest = st.tabs(
    ["🏠 Overview", "📊 Historical Data", "🔮 Forecast", "🧪 Model Backtest"]
)

with tab_overview:
    yoy_change = latest['yoy'] - previous['yoy']
    if yoy_change > 0.05:
        trend_phrase = f"up {yoy_change:.2f} pp from last month"
    elif yoy_change < -0.05:
        trend_phrase = f"down {abs(yoy_change):.2f} pp from last month"
    else:
        trend_phrase = "roughly unchanged from last month"

    if latest['yoy'] > FED_TARGET:
        target_phrase = f"above the Federal Reserve's {FED_TARGET:.0f}% target"
    else:
        target_phrase = f"at or below the Federal Reserve's {FED_TARGET:.0f}% target"

    st.info(
        f"**Key takeaway:** Annual inflation (YoY) is currently **{latest['yoy']:.2f}%**, "
        f"{trend_phrase}, and is {target_phrase}."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest CPI", f"{latest['cpi']:.3f}", f"{latest['cpi'] - previous['cpi']:.3f}", delta_color="off")
    col2.metric("Monthly Inflation (MoM)", f"{latest['mom']:.2f}%", f"{latest['mom'] - previous['mom']:.2f} pp", delta_color="inverse")
    col3.metric("Annual Inflation (YoY)", f"{latest['yoy']:.2f}%", f"{latest['yoy'] - previous['yoy']:.2f} pp", delta_color="inverse")

    st.subheader("YoY Inflation Forecast (with confidence interval)")
    st.caption("The shaded band shows the model's confidence interval — narrower means the model is more certain.")

    historical_yoy = data.dropna(subset=["yoy"])[["date", "yoy"]].tail(history_months)
    forecast_display = forecast.head(forecast_months)

    color_scale = alt.Scale(
        domain=["Actual", "Forecast", "Fed Target (2%)"],
        range=[PRIMARY_COLOR, ACCENT_COLOR, "#64748b"],
    )

    actual_df = historical_yoy.rename(columns={"yoy": "value"}).assign(series="Actual")
    forecast_df = forecast_display.rename(columns={"mean": "value"}).assign(series="Forecast")
    target_df = pd.DataFrame({"series": ["Fed Target (2%)"], "value": [FED_TARGET]})

    actual_line = alt.Chart(actual_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="YoY Inflation (%)"),
        color=alt.Color("series:N", scale=color_scale, title="Series"),
        tooltip=["date:T", alt.Tooltip("value:Q", format=".2f")],
    )
    forecast_line = alt.Chart(forecast_df).mark_line(strokeWidth=2.5, strokeDash=[5, 3]).encode(
        x="date:T", y="value:Q",
        color=alt.Color("series:N", scale=color_scale),
        tooltip=["date:T", alt.Tooltip("value:Q", format=".2f")],
    )
    forecast_band = alt.Chart(forecast_display).mark_area(opacity=0.18, color=ACCENT_COLOR).encode(
        x="date:T", y="mean_ci_lower:Q", y2="mean_ci_upper:Q"
    )
    target_line = alt.Chart(target_df).mark_rule(strokeDash=[4, 4]).encode(
        y="value:Q",
        color=alt.Color("series:N", scale=color_scale),
    )

    st.altair_chart(
        (forecast_band + actual_line + forecast_line + target_line).interactive().properties(height=380),
        use_container_width=True,
    )

    st.divider()
    with st.expander("ℹ️ About this project / Methodology & Disclaimer"):
        st.markdown(
            "**Data source:** U.S. CPI (All Items) data is sourced from the "
            "Federal Reserve Economic Data (FRED) database, published by the "
            "Federal Reserve Bank of St. Louis, and updated automatically once a day."
        )
        st.markdown(
            "**Method:** Forecasts are generated with an ARIMA (AutoRegressive "
            "Integrated Moving Average) time series model. The best parameter "
            "combination is chosen using rolling backtests (see the Model Backtest "
            "tab). The shaded band on the forecast chart is the model's confidence "
            "interval — it widens for months further in the future because "
            "uncertainty naturally grows the further out a statistical model tries "
            "to predict."
        )
        st.markdown(
            "**Disclaimer:** This dashboard is a personal research and "
            "demonstration project. Forecasts are statistical estimates only, "
            "are not guaranteed to be accurate, and should not be treated as "
            "financial, investment, or economic policy advice."
        )

with tab_history:
    st.subheader("Historical CPI")
    st.line_chart(data.set_index("date")["cpi"].tail(history_months), color=PRIMARY_COLOR)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("YoY Inflation")
        yoy_hist_df = (
            data.dropna(subset=["yoy"])[["date", "yoy"]]
            .tail(history_months)
            .rename(columns={"yoy": "value"})
            .assign(series="Actual")
        )
        target_hist_df = pd.DataFrame({"series": ["Fed Target (2%)"], "value": [FED_TARGET]})
        hist_color_scale = alt.Scale(domain=["Actual", "Fed Target (2%)"], range=[PRIMARY_COLOR, "#64748b"])
        yoy_line = alt.Chart(yoy_hist_df).mark_line().encode(
            x="date:T",
            y=alt.Y("value:Q", title="YoY Inflation (%)"),
            color=alt.Color("series:N", scale=hist_color_scale, title="Series"),
            tooltip=["date:T", alt.Tooltip("value:Q", format=".2f")],
        )
        target_line_hist = alt.Chart(target_hist_df).mark_rule(strokeDash=[4, 4]).encode(
            y="value:Q",
            color=alt.Color("series:N", scale=hist_color_scale),
        )
        st.altair_chart((yoy_line + target_line_hist).properties(height=300), use_container_width=True)
    with col_b:
        st.subheader("MoM Inflation")
        st.line_chart(data.set_index("date")["mom"].tail(history_months), color=ACCENT_COLOR)

    with st.expander("View raw data table"):
        st.dataframe(data.tail(history_months), use_container_width=True)

with tab_forecast:
    st.subheader("Forecast Detail")
    st.caption("mean = model forecast, ci_lower / ci_upper = confidence interval bounds.")
    st.dataframe(
        forecast_display.rename(
            columns={"mean": "Forecast", "mean_ci_lower": "CI Lower", "mean_ci_upper": "CI Upper"}
        )[["date", "Forecast", "CI Lower", "CI Upper"]],
        use_container_width=True,
    )

with tab_backtest:
    st.subheader("Rolling Backtest Results (sorted by RMSE)")
    st.caption(
        "MAE / RMSE are forecast error metrics — lower means the model was more accurate historically. "
        "'Naive baseline' simply repeats last month's value, used as a reference point."
    )
    st.dataframe(rolling_backtest.sort_values("rmse"), use_container_width=True)

    with st.expander("View full ARIMA parameter comparison"):
        st.dataframe(model_comparison.sort_values("rmse"), use_container_width=True)
