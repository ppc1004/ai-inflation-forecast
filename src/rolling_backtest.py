from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cpi_data.csv"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "rolling_backtest.csv"
)

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


def calculate_metrics(actual, predicted):
    error = np.array(actual) - np.array(predicted)

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    return mae, rmse


def rolling_backtest():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])

    data["yoy"] = data["cpi"].pct_change(12) * 100

    yoy = (
        data.dropna(subset=["yoy"])
        .set_index("date")["yoy"]
        .asfreq("MS")
        .dropna()
    )

    if len(yoy) <= BACKTEST_MONTHS:
        raise ValueError("Not enough data for rolling backtesting.")

    test_start = len(yoy) - BACKTEST_MONTHS
    actual_values = yoy.iloc[test_start:].to_numpy()

    results = []

    # Naive baseline:
    # Predict that next month's inflation equals this month's inflation.
    naive_predictions = yoy.iloc[
        test_start - 1 : len(yoy) - 1
    ].to_numpy()

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

            print(
                f"ARIMA{order}: "
                f"MAE={mae:.3f}, RMSE={rmse:.3f}"
            )

        except Exception as error:
            print(f"ARIMA{order} failed: {error}")

    results_df = (
        pd.DataFrame(results)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nRolling backtest results:")
    print(results_df)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    rolling_backtest()