from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/cpi_data.csv")
FORECAST_PATH = Path("data/yoy_forecast.csv")

st.title("AI Inflation Forecast")
st.write("Track U.S. CPI data and forecast inflation.")

data = pd.read_csv(DATA_PATH, parse_dates=["date"])
forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
data["mom"] = data["cpi"].pct_change() * 100
data["yoy"] = data["cpi"].pct_change(12) * 100

latest = data.iloc[-1]
col1, col2, col3 = st.columns(3)

col1.metric("Latest CPI", f"{latest['cpi']:.3f}")
col2.metric("Monthly Inflation", f"{latest['mom']:.2f}%")
col3.metric("Annual Inflation", f"{latest['yoy']:.2f}%")

st.caption(
    f"Latest observation: {latest['date'].strftime('%B %Y')}"
)

st.subheader("Historical CPI")

st.line_chart(
    data.set_index("date")["cpi"]
)

st.subheader("Year-over-Year Inflation")

st.line_chart(
    data.set_index("date")["yoy"]
)

st.subheader("Month-over-Month Inflation")

st.line_chart(
    data.set_index("date")["mom"]
)

st.subheader("YoY Inflation Forecast")

historical_yoy = (
    data.dropna(subset=["yoy"])
    .set_index("date")[["yoy"]]
    .tail(60)
    .rename(columns={"yoy": "Actual YoY"})
)

forecast_yoy = (
    forecast.set_index("date")[["mean"]]
    .rename(columns={"mean": "ARIMA Forecast"})
)

forecast_chart = historical_yoy.join(
    forecast_yoy,
    how="outer"
)

st.line_chart(forecast_chart)