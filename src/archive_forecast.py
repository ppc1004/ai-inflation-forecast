"""Keep a permanent record of every published forecast.

Each run appends the current 12-month forecast (data/yoy_forecast.csv) to
data/forecast_archive.csv, labelled with its vintage: the last month of CPI
data the model had seen. Re-running for the same vintage replaces that
vintage's rows instead of duplicating them.

Once new CPI prints arrive, the website compares these archived forecasts
with what actually happened: a true out-of-sample track record.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

CPI_PATH = DATA_DIR / "cpi_data.csv"
FORECAST_PATH = DATA_DIR / "yoy_forecast.csv"
ARCHIVE_PATH = DATA_DIR / "forecast_archive.csv"


def archive_forecast():
    cpi = pd.read_csv(CPI_PATH, parse_dates=["date"])
    vintage = cpi["date"].max()

    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    snapshot = pd.DataFrame(
        {
            "vintage": vintage.strftime("%Y-%m-%d"),
            "target_date": forecast["date"].dt.strftime("%Y-%m-%d"),
            "horizon": range(1, len(forecast) + 1),
            "forecast": forecast["mean"].round(4),
            "lower_95": forecast["mean_ci_lower"].round(4),
            "upper_95": forecast["mean_ci_upper"].round(4),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    )

    if ARCHIVE_PATH.exists():
        archive = pd.read_csv(ARCHIVE_PATH, dtype={"vintage": str, "target_date": str})
        archive = archive[archive["vintage"] != snapshot["vintage"].iloc[0]]
        archive = pd.concat([archive, snapshot], ignore_index=True)
    else:
        archive = snapshot

    archive = archive.sort_values(["vintage", "horizon"]).reset_index(drop=True)
    archive.to_csv(ARCHIVE_PATH, index=False)

    print(
        f"Archived {len(snapshot)} forecasts for vintage {vintage:%Y-%m} "
        f"({archive['vintage'].nunique()} vintages total)"
    )


if __name__ == "__main__":
    archive_forecast()
