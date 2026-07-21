from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path("data/cpi_data.csv")

st.title("AI Inflation Forecast")
st.write("Track U.S. CPI data and forecast inflation.")

data = pd.read_csv(DATA_PATH, parse_dates=["date"])

latest = data.iloc[-1]

st.metric(
    label="Latest CPI",
    value=f"{latest['cpi']:.3f}",
)

st.caption(
    f"Latest observation: {latest['date'].strftime('%B %Y')}"
)

st.subheader("Historical CPI")

st.line_chart(
    data.set_index("date")["cpi"]
)