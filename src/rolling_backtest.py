from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_yoy

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cpi_data.csv"

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "rolling_backtest.csv"

ERRORS_PATH = Path(__file__).resolve().parents[1] / "data" / "backtest_errors.csv"

BACKTEST_MONTHS = 24

CANDIDATE_ORDERS = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (1, 0, 2),
    (2, 0, 2),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
]


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> tuple[float, float]:
    error = np.array(actual) - np.array(predicted)

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    return mae, rmse


def make_error_rows(
    model_name: str,
    dates: pd.DatetimeIndex,
    actual: Sequence[float],
    predicted: Sequence[float],
) -> list[dict]:
    # One row per backtest month, so the website can plot error over time.
    rows = []

    for date, actual_value, predicted_value in zip(dates, actual, predicted, strict=True):
        error = actual_value - predicted_value  # positive = forecast too low

        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "model": model_name,
                "actual": actual_value,
                "forecast": predicted_value,
                "error": error,
                "abs_error": abs(error),
            }
        )

    return rows


def rolling_backtest() -> None:
    yoy = load_yoy(DATA_PATH)

    if len(yoy) <= BACKTEST_MONTHS:
        raise ValueError("Not enough data for rolling backtesting.")

    test_start = len(yoy) - BACKTEST_MONTHS
    actual_values = yoy.iloc[test_start:].to_numpy()

    test_dates = yoy.index[test_start:]

    results = []
    error_rows = []

    # Naive baseline:
    # Predict that next month's inflation equals this month's inflation.
    naive_predictions = yoy.iloc[test_start - 1 : len(yoy) - 1].to_numpy()

    naive_mae, naive_rmse = calculate_metrics(
        actual_values,
        naive_predictions,
    )

    results.append(
        {
            "model": "Naive baseline",
            "mae": naive_mae,
            "rmse": naive_rmse,
        }
    )

    error_rows.extend(
        make_error_rows(
            "Naive baseline",
            test_dates,
            actual_values,
            naive_predictions,
        )
    )

    warnings.filterwarnings("ignore")

    for order in CANDIDATE_ORDERS:
        predictions = []

        try:
            for position in range(test_start, len(yoy)):
                train = yoy.iloc[:position]

                model = ARIMA(train, order=order)
                fitted_model = model.fit()

                prediction = fitted_model.forecast(steps=1)
                predictions.append(float(prediction.iloc[0]))

            mae, rmse = calculate_metrics(
                actual_values,
                predictions,
            )

            results.append(
                {
                    "model": f"ARIMA{order}",
                    "mae": mae,
                    "rmse": rmse,
                }
            )

            error_rows.extend(
                make_error_rows(
                    f"ARIMA{order}",
                    test_dates,
                    actual_values,
                    predictions,
                )
            )

            print(f"ARIMA{order}: MAE={mae:.3f}, RMSE={rmse:.3f}")

        except Exception as error:
            print(f"ARIMA{order} failed: {error}")

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)

    results_df.to_csv(OUTPUT_PATH, index=False)

    pd.DataFrame(error_rows).to_csv(ERRORS_PATH, index=False)
    print(f"Monthly backtest errors saved to {ERRORS_PATH}")

    print("\nRolling backtest results:")
    print(results_df)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    rolling_backtest()
