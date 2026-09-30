"""Head-to-head benchmark of four forecasting models, with significance tests.

Every model produces one-month-ahead forecasts of year-over-year CPI inflation
over the same evaluation window. At each forecast origin a model is fit only
on data that was available at the time (rolling origin, expanding window).

Models
------
naive    "No change": next month's inflation equals this month's.
arima    ARIMA(1, 0, 2) on YoY inflation alone (the production model).
arimax   The same ARIMA plus last month's energy and core inflation as
         exogenous regressors (both are known when the forecast is made).
gbm      Gradient-boosted trees predicting next month's change in YoY
         inflation from lags of headline, core, energy and monthly CPI.

Each model is compared with the naive benchmark using the Diebold-Mariano
test (squared-error loss, Harvey-Leybourne-Newbold small-sample correction),
which asks whether the difference in accuracy is larger than chance.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

from data_utils import load_extra_yoy, load_yoy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CPI_PATH = DATA_DIR / "cpi_data.csv"
EXTRA_PATH = DATA_DIR / "extra_series.csv"
SUMMARY_PATH = DATA_DIR / "model_benchmark.csv"
ERRORS_PATH = DATA_DIR / "model_benchmark_errors.csv"

ORDER = (1, 0, 2)  # keep in sync with arima_forecast.py
WINDOW = 48  # evaluation months (same window as horizon_backtest.py)
GBM_START = "1960-01-01"

LABELS = {
    "naive": "Naive (no change)",
    "arima": "ARIMA(1,0,2)",
    "arimax": "ARIMAX (+ energy, core)",
    "gbm": "Gradient boosting",
}


# ---------------------------------------------------------------- statistics
def diebold_mariano(
    errors_model: np.ndarray, errors_benchmark: np.ndarray, horizon: int = 1
) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

    Uses squared-error loss. Returns (statistic, two-sided p-value). A negative
    statistic means the model's errors are smaller than the benchmark's.
    """
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_benchmark, dtype=float)
    d = e1**2 - e2**2
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")

    d_mean = d.mean()
    centered = d - d_mean
    # long-run variance with autocovariances up to horizon - 1
    long_run_var = centered @ centered / n
    for lag in range(1, horizon):
        long_run_var += 2 * (centered[lag:] @ centered[:-lag]) / n
    if long_run_var <= 0:
        return float("nan"), float("nan")

    dm = d_mean / np.sqrt(long_run_var / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = dm * correction
    p_value = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    return float(dm_hln), float(p_value)


# ---------------------------------------------------------------- features
def build_features(yoy: pd.Series, extra: pd.DataFrame | None, cpi_mom: pd.Series) -> pd.DataFrame:
    """Predictors for month t, using only information available at the end of month t-1."""
    frame = pd.DataFrame({"yoy": yoy})
    frame["target_change"] = frame["yoy"].diff()

    frame["yoy_lag1"] = frame["yoy"].shift(1)
    for lag in (1, 2, 3, 6, 12):
        frame[f"change_lag{lag}"] = frame["target_change"].shift(lag)
    for lag in (1, 2, 3):
        frame[f"mom_lag{lag}"] = cpi_mom.reindex(frame.index).shift(lag)

    if extra is not None:
        for column in ("energy", "core"):
            if column in extra:
                series = extra[column].reindex(frame.index)
                frame[f"{column}_lag1"] = series.shift(1)
                frame[f"{column}_change_lag1"] = series.diff().shift(1)

    frame["month"] = frame.index.month
    return frame


def gbm_forecast(features: pd.DataFrame, position: int) -> float:
    columns = [c for c in features.columns if c not in ("yoy", "target_change")]
    train = features.iloc[:position].loc[GBM_START:].dropna()
    row = features.iloc[[position]][columns]
    if row.isna().any(axis=None) or len(train) < 120:
        return float("nan")

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=0
    )
    model.fit(train[columns], train["target_change"])
    change = float(model.predict(row)[0])
    return float(features["yoy"].iloc[position - 1] + change)


# ---------------------------------------------------------------- benchmark
def run_benchmark() -> None:
    yoy = load_yoy(CPI_PATH)
    cpi = pd.read_csv(CPI_PATH, parse_dates=["date"]).set_index("date").asfreq("MS")
    cpi_mom = cpi["cpi"].interpolate(limit_area="inside").pct_change(fill_method=None) * 100

    extra = load_extra_yoy(EXTRA_PATH, ["energy", "core"]) if EXTRA_PATH.exists() else None
    exog = None
    if extra is not None and {"energy", "core"} <= set(extra.columns):
        exog = extra[["energy", "core"]].reindex(yoy.index).shift(1)

    features = build_features(yoy, extra, cpi_mom)

    warnings.filterwarnings("ignore")
    rows = []

    for position in range(len(yoy) - WINDOW, len(yoy)):
        date = yoy.index[position]
        actual = float(yoy.iloc[position])
        train = yoy.iloc[:position]
        forecasts = {"naive": float(train.iloc[-1])}

        try:
            fitted = ARIMA(train, order=ORDER).fit()
            forecasts["arima"] = float(fitted.forecast(steps=1).iloc[0])
        except Exception as error:
            print(f"{date:%Y-%m} arima failed: {error}")

        if exog is not None:
            exog_train = exog.iloc[:position]
            valid = exog_train.notna().all(axis=1)
            first_valid = valid.idxmax() if valid.any() else None
            next_exog = exog.iloc[[position]]
            if first_valid is not None and not next_exog.isna().any(axis=None):
                try:
                    fitted_x = ARIMA(
                        train.loc[first_valid:], exog=exog_train.loc[first_valid:], order=ORDER
                    ).fit()
                    forecasts["arimax"] = float(fitted_x.forecast(steps=1, exog=next_exog).iloc[0])
                except Exception as error:
                    print(f"{date:%Y-%m} arimax failed: {error}")

        forecasts["gbm"] = gbm_forecast(features, position)

        for model, forecast in forecasts.items():
            if np.isfinite(forecast):
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "model": model,
                        "actual": actual,
                        "forecast": forecast,
                        "error": actual - forecast,
                    }
                )

    errors = pd.DataFrame(rows)
    summary = summarise(errors)

    errors.round(4).to_csv(ERRORS_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """One row per model: MAE, RMSE and Diebold-Mariano tests against naive and against ARIMA."""
    wide = errors.pivot(index="date", columns="model", values="error")
    naive = wide["naive"]
    out = []
    for model in [m for m in LABELS if m in wide.columns]:
        paired = pd.concat([wide[model], naive], axis=1, keys=["m", "n"]).dropna()
        e = paired["m"].to_numpy()
        mae = float(np.mean(np.abs(e)))
        rmse = float(np.sqrt(np.mean(e**2)))
        naive_mae = float(np.mean(np.abs(paired["n"])))
        if model == "naive":
            dm, p = float("nan"), float("nan")
        else:
            dm, p = diebold_mariano(e, paired["n"].to_numpy())
        p_vs_arima = float("nan")
        if model not in ("naive", "arima") and "arima" in wide.columns:
            both = pd.concat([wide[model], wide["arima"]], axis=1, keys=["m", "a"]).dropna()
            _, p_vs_arima = diebold_mariano(both["m"].to_numpy(), both["a"].to_numpy())
        out.append(
            {
                "model": model,
                "label": LABELS[model],
                "n": len(paired),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mae_vs_naive_pct": round((mae / naive_mae - 1) * 100, 2),
                "dm_stat": round(dm, 3) if np.isfinite(dm) else None,
                "dm_pvalue": round(p, 4) if np.isfinite(p) else None,
                "dm_pvalue_vs_arima": round(p_vs_arima, 4) if np.isfinite(p_vs_arima) else None,
            }
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    run_benchmark()
