"""Download supporting FRED series used by the website.

Writes one monthly, wide CSV (data/extra_series.csv) with a column per series.
Daily series (the 10-year breakeven rate) are averaged by month.
If a single series fails to download, its previous values are kept so one
bad request never wipes a chart off the site.
"""

from pathlib import Path

import pandas as pd


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "extra_series.csv"
)

# column name on the website -> FRED series id
SERIES = {
    # CPI components (index levels, seasonally adjusted)
    "core": "CPILFESL",          # all items less food and energy
    "food": "CPIUFDSL",
    "energy": "CPIENGSL",
    "shelter": "CUSR0000SAH1",
    # BLS average prices, U.S. city average (dollars)
    "eggs": "APU0000708111",     # grade A large, per dozen
    "gasoline": "APU000074714",  # unleaded regular, per gallon
    "milk": "APU0000709112",     # whole, per gallon
    "coffee": "APU0000717311",   # ground roast, per pound
    "bread": "APU0000702111",    # white pan, per pound
    # wages, policy and expectations
    "wages": "AHETPI",           # avg hourly earnings, production & nonsupervisory
    "fed_funds": "FEDFUNDS",     # effective federal funds rate, %
    "exp_consumers_1y": "MICH",  # U. Michigan expected inflation, next 12 months, %
    "exp_market_10y": "T10YIE",  # 10-year breakeven inflation rate, %
}


def fetch_series(series_id):
    data = pd.read_csv(FRED_CSV_URL.format(series_id))
    data.columns = ["date", "value"]
    data["date"] = pd.to_datetime(data["date"])
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    series = data.dropna().set_index("date")["value"]

    # Monthly series pass through unchanged; daily series become monthly averages.
    return series.resample("MS").mean()


def download_extra():
    previous = None
    if OUTPUT_PATH.exists():
        previous = pd.read_csv(OUTPUT_PATH, parse_dates=["date"]).set_index("date")

    columns = {}

    for name, series_id in SERIES.items():
        try:
            columns[name] = fetch_series(series_id)
            print(f"{series_id:>14} -> {name}: {columns[name].dropna().index.max():%Y-%m}")
        except Exception as error:
            print(f"{series_id} failed: {error}")
            if previous is not None and name in previous:
                columns[name] = previous[name]
                print(f"  kept previous values for {name}")

    if not columns:
        raise SystemExit("No series could be downloaded.")

    output = pd.DataFrame(columns).sort_index()
    output = output[output.index >= "1947-01-01"]
    output.index.name = "date"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.round(4).to_csv(OUTPUT_PATH, date_format="%Y-%m-%d")

    print(f"Saved {len(output)} rows x {len(output.columns)} series to {OUTPUT_PATH}")


if __name__ == "__main__":
    download_extra()
