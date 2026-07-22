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
    / "model_comparison.csv"
)


def compare_models():
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Calculate year-over-year inflation
    data["yoy"] = data["cpi"].pct_change(12) * 100

    yoy = (
        data.dropna(subset=["yoy"])
        .set_index("date")["yoy"]
        .asfreq("MS")    
        .dropna()
    )
    # Keep the final 12 months for testing
    train = yoy.iloc[:-12]
    test = yoy.iloc[-12:]

    candidate_orders = [
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

    results = []

    warnings.filterwarnings("ignore")

    for order in candidate_orders:
        try:
            model = ARIMA(train, order=order)
            fitted_model = model.fit()

            prediction = fitted_model.forecast(steps=len(test))

            error = test.to_numpy() - prediction.to_numpy()

            mae = np.mean(np.abs(error))
            rmse = np.sqrt(np.mean(error**2))

            results.append(
                {
                    "order": str(order),
                    "mae": mae,
                    "rmse": rmse,
                    "aic": fitted_model.aic,
                }
            )

            print(
                f"ARIMA{order}: "
                f"MAE={mae:.3f}, RMSE={rmse:.3f}"
            )

        except Exception as error:
            print(f"ARIMA{order} failed: {error}")

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nBest models:")
    print(results_df.head())

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    compare_models()