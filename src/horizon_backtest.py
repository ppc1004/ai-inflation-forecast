"""How accurate is the forecast 1, 2, ... 12 months ahead?

The rolling backtest (rolling_backtest.py) scores one-month-ahead forecasts.
This script repeats the exercise for every horizon up to 12 months and also
checks calibration: how often the actual value landed inside the model's
95% interval. A well-calibrated model should be close to 95%.

For each forecast origin in the evaluation window, the model is fit only on
data up to that month, forecasts 12 months ahead, and every forecast whose
target month has already been published is scored.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_yoy


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_DIR / "cpi_data.csv"
OUTPUT_PATH = DATA_DIR / "horizon_backtest.csv"

ORDER = (1, 0, 2)      # keep in sync with arima_forecast.py
HORIZON = 12           # months ahead
N_ORIGINS = 48         # forecast origins in the evaluation window


def score_origin(yoy, position, forecast_df):
    """Compare one origin's forecasts with the actual values that exist."""
    rows = []
    last_observed = yoy.iloc[position - 1]

    for step in range(HORIZON):
        target = position + step
        if target >= len(yoy):
            break

        actual = yoy.iloc[target]
        row = forecast_df.iloc[step]

        rows.append(
            {
                "origin": yoy.index[position - 1],
                "horizon": step + 1,
                "actual": actual,
                "forecast": row["mean"],
                "lower": row["mean_ci_lower"],
                "upper": row["mean_ci_upper"],
                "naive": last_observed,  # "no change" benchmark
            }
        )

    return rows


def summarise(records):
    """Aggregate per-forecast records into one row per horizon."""
    df = pd.DataFrame(records)

    model_error = df["actual"] - df["forecast"]
    naive_error = df["actual"] - df["naive"]
    inside = (df["actual"] >= df["lower"]) & (df["actual"] <= df["upper"])

    df = df.assign(
        model_abs=model_error.abs(),
        model_sq=model_error**2,
        naive_abs=naive_error.abs(),
        naive_sq=naive_error**2,
        inside=inside,
    )

    grouped = df.groupby("horizon")

    summary = pd.DataFrame(
        {
            "n": grouped.size(),
            "model_mae": grouped["model_abs"].mean(),
            "model_rmse": np.sqrt(grouped["model_sq"].mean()),
            "naive_mae": grouped["naive_abs"].mean(),
            "naive_rmse": np.sqrt(grouped["naive_sq"].mean()),
            "coverage_95": grouped["inside"].mean() * 100,
        }
    )

    return summary.reset_index().round(4)


def horizon_backtest():
    yoy = load_yoy(DATA_PATH)

    if len(yoy) <= N_ORIGINS + 24:
        raise ValueError("Not enough data for the horizon backtest.")

    warnings.filterwarnings("ignore")

    records = []

    for position in range(len(yoy) - N_ORIGINS, len(yoy)):
        train = yoy.iloc[:position]

        fitted_model = ARIMA(train, order=ORDER).fit()
        forecast_df = fitted_model.get_forecast(steps=HORIZON).summary_frame(alpha=0.05)

        records.extend(score_origin(yoy, position, forecast_df))

    summary = summarise(records)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    horizon_backtest()
