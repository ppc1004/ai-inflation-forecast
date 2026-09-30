"""Shared data preparation for every model script."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def _monthly(path: str | Path) -> pd.DataFrame:
    """Read a date-indexed CSV onto a complete monthly (MS) index."""
    return (
        pd.read_csv(path, parse_dates=["date"])
        .set_index("date")
        .sort_index()
        .asfreq("MS")  # one row per calendar month; a missing month becomes NaN
    )


def load_yoy(cpi_path: str | Path) -> pd.Series:
    """Return monthly year-over-year CPI inflation (%) as a Series with MS frequency.

    BLS did not publish October 2025 CPI (government shutdown). Missing months
    inside the series are filled by linear interpolation so that pct_change(12)
    always compares a month with the same calendar month one year earlier.
    """
    cpi = _monthly(cpi_path)["cpi"].interpolate(limit_area="inside")
    yoy = cpi.pct_change(12, fill_method=None) * 100
    return yoy.dropna().asfreq("MS")


def load_extra_yoy(extra_path: str | Path, columns: Iterable[str]) -> pd.DataFrame:
    """Year-over-year % change of selected columns in data/extra_series.csv."""
    data = _monthly(extra_path)
    present = [c for c in columns if c in data.columns]
    levels = data[present].interpolate(limit_area="inside")
    return levels.pct_change(12, fill_method=None) * 100
