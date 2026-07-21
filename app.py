from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/cpi_data.csv")

st.title("AI Inflation Forecast")
st.write("Track U.S. CPI data and forecast inflation.")

data = pd.read_csv(DATA_PATH, parse_dates=["date"])
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