import pytest

pytest.importorskip("statsmodels")

from horizon_backtest import summarise


def test_summarise_errors_and_coverage():
    records = [
        {"horizon": 1, "actual": 3.0, "forecast": 2.5, "lower": 2.0, "upper": 3.5, "naive": 2.0},
        {"horizon": 1, "actual": 4.0, "forecast": 3.5, "lower": 3.0, "upper": 3.8, "naive": 3.0},
        {"horizon": 2, "actual": 2.0, "forecast": 2.0, "lower": 1.0, "upper": 3.0, "naive": 2.5},
    ]

    summary = summarise(records).set_index("horizon")

    assert summary.loc[1, "n"] == 2
    assert summary.loc[1, "model_mae"] == pytest.approx(0.5)
    assert summary.loc[1, "naive_mae"] == pytest.approx(1.0)
    assert summary.loc[1, "coverage_95"] == pytest.approx(50.0)
    assert summary.loc[2, "model_mae"] == pytest.approx(0.0)
    assert summary.loc[2, "coverage_95"] == pytest.approx(100.0)
