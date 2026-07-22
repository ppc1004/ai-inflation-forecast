from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cpi_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "yoy_forecast.csv"


def run_arima_forecast():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Calculate year-over-year inflation
    data["yoy"] = data["cpi"].pct_change(12) * 100
    yoy = data.dropna(subset=["yoy"]).set_index("date")["yoy"]
    yoy = yoy.asfreq("MS")

    # ARIMA model
    model = ARIMA(yoy, order=(1, 0, 1))
    fitted_model = model.fit()

    # Forecast next 12 months
    forecast = fitted_model.get_forecast(steps=12)

    forecast_df = forecast.summary_frame()
    forecast_df.index.name = "date"
    forecast_df.to_csv(OUTPUT_PATH)

    print(f"Forecast saved to {OUTPUT_PATH}")
    print(forecast_df)


if __name__ == "__main__":
    run_arima_forecast()