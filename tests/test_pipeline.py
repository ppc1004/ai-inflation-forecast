"""Tests for the data pipeline. They use small synthetic inputs and never hit the network."""

import pandas as pd
import pytest

import archive_forecast
import download_extra
from data_utils import load_yoy


def write_cpi(path, start="2023-01-01", months=30, skip=None, growth=0.003):
    dates = pd.date_range(start, periods=months, freq="MS")
    cpi = [100 * (1 + growth) ** i for i in range(months)]
    df = pd.DataFrame({"date": dates, "cpi": cpi})
    if skip is not None:
        df = df[df["date"] != pd.Timestamp(skip)]
    df.to_csv(path, index=False)


def test_load_yoy_is_twelve_month_change(tmp_path):
    path = tmp_path / "cpi.csv"
    write_cpi(path)

    yoy = load_yoy(path)

    assert yoy.index[0] == pd.Timestamp("2024-01-01")
    assert yoy.iloc[-1] == pytest.approx(((1.003**12) - 1) * 100)


def test_load_yoy_fills_a_missing_month(tmp_path):
    # Like October 2025: one month is absent from the published data.
    path = tmp_path / "cpi.csv"
    write_cpi(path, skip="2024-06-01")

    yoy = load_yoy(path)

    assert yoy.index.freqstr == "MS"
    assert yoy.isna().sum() == 0
    assert pd.Timestamp("2024-06-01") in yoy.index
    # Constant growth: every YoY value, including the ones that compare
    # against the interpolated month, must stay close to the true rate.
    true_rate = ((1.003**12) - 1) * 100
    assert (yoy - true_rate).abs().max() < 0.01


def test_archive_replaces_same_vintage(tmp_path, monkeypatch):
    write_cpi(tmp_path / "cpi.csv")
    pd.DataFrame(
        {
            "date": pd.date_range("2025-07-01", periods=12, freq="MS"),
            "mean": 3.0,
            "mean_se": 0.4,
            "mean_ci_lower": 2.2,
            "mean_ci_upper": 3.8,
        }
    ).to_csv(tmp_path / "fc.csv", index=False)

    monkeypatch.setattr(archive_forecast, "CPI_PATH", tmp_path / "cpi.csv")
    monkeypatch.setattr(archive_forecast, "FORECAST_PATH", tmp_path / "fc.csv")
    monkeypatch.setattr(archive_forecast, "ARCHIVE_PATH", tmp_path / "archive.csv")

    archive_forecast.archive_forecast()
    archive_forecast.archive_forecast()

    archive = pd.read_csv(tmp_path / "archive.csv")
    assert len(archive) == 12
    assert archive["vintage"].nunique() == 1
    assert archive["horizon"].tolist() == list(range(1, 13))


def test_download_extra_keeps_previous_values_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "extra.csv"
    pd.DataFrame({"date": ["2026-01-01"], "core": [300.0], "food": [310.0]}).to_csv(out, index=False)

    def fake_fetch(series_id):
        if series_id == "CPIUFDSL":
            raise ConnectionError("FRED unavailable")
        return pd.Series([1.0], index=pd.to_datetime(["2026-01-01"]))

    monkeypatch.setattr(download_extra, "OUTPUT_PATH", out)
    monkeypatch.setattr(download_extra, "fetch_series", fake_fetch)

    download_extra.download_extra()

    result = pd.read_csv(out)
    assert result.loc[0, "food"] == 310.0      # kept from the previous file
    assert result.loc[0, "core"] == 1.0        # refreshed
