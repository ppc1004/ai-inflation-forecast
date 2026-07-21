from pathlib import Path

import pandas as pd


FRED_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "cpi_data.csv"
)


def download_cpi():
    data = pd.read_csv(FRED_CSV_URL)

    data = data.rename(
        columns={
            "observation_date": "date",
            "CPIAUCSL": "cpi",
        }
    )

    data["date"] = pd.to_datetime(data["date"])
    data["cpi"] = pd.to_numeric(data["cpi"], errors="coerce")

    data = data.dropna().sort_values("date")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(data)} rows to {OUTPUT_PATH}")
    print(data.tail())


if __name__ == "__main__":
    download_cpi()