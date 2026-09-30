import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")
pytest.importorskip("sklearn")

from model_benchmark import build_features, diebold_mariano, summarise


def test_dm_matches_hand_calculation():
    rng = np.random.default_rng(1)
    naive = rng.normal(0, 1.0, 40)
    model = rng.normal(0, 0.6, 40)

    stat, p = diebold_mariano(model, naive)

    d = model**2 - naive**2
    n = len(d)
    raw = d.mean() / np.sqrt(((d - d.mean()) ** 2).mean() / n)
    expected = raw * np.sqrt((n + 1 - 2 + 0) / n)
    assert stat == pytest.approx(expected)
    assert stat < 0  # the model has smaller errors
    assert 0 <= p <= 1


def test_dm_identical_errors_is_undefined():
    e = np.array([0.1, -0.2, 0.3, 0.0])
    stat, p = diebold_mariano(e, e)
    assert np.isnan(stat) and np.isnan(p)


def test_features_only_use_past_information():
    idx = pd.date_range("2000-01-01", periods=30, freq="MS")
    yoy = pd.Series(np.arange(30, dtype=float), index=idx)
    mom = pd.Series(np.arange(30, dtype=float) / 10, index=idx)

    features = build_features(yoy, None, mom)

    row = features.loc["2001-06-01"]
    assert row["yoy_lag1"] == yoy.loc["2001-05-01"]
    assert row["mom_lag1"] == mom.loc["2001-05-01"]
    assert row["target_change"] == 1.0


def test_summarise_reports_improvement_vs_naive():
    dates = [f"2025-{m:02d}-01" for m in range(1, 7)]
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "model": "naive", "error": [0.4, -0.5, 0.3, -0.6, 0.5, -0.4][i]})
        rows.append({"date": d, "model": "arima", "error": [0.2, -0.2, 0.1, -0.3, 0.2, -0.1][i]})

    summary = summarise(pd.DataFrame(rows)).set_index("model")

    assert summary.loc["naive", "mae_vs_naive_pct"] == 0
    assert summary.loc["arima", "mae_vs_naive_pct"] < 0
    assert summary.loc["arima", "dm_stat"] < 0
