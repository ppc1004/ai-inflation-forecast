"""Shared data preparation for every model script."""

import pandas as pd


def load_yoy(cpi_path):
    """Return monthly year-over-year CPI inflation (%) as a Series with MS frequency.

    BLS did not publish October 2025 CPI (government shutdown). Missing months
    inside the series are filled by linear interpolation so that pct_change(12)
    always compares a month with the same calendar month one year earlier.
    """
    data = (
        pd.read_csv(cpi_path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("MS")  # one row per calendar month; a missing month becomes NaN
    )

    data["cpi"] = data["cpi"].interpolate(limit_area="inside")

    yoy = data["cpi"].pct_change(12, fill_method=None) * 100

    return yoy.dropna().asfreq("MS")
