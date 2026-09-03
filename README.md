# AI Inflation Forecast

A small dashboard that tracks U.S. CPI (Consumer Price Index) data and forecasts near-term inflation with a classical ARIMA time-series model.

**Live demo:** https://ppc1004.github.io/ai-inflation-forecast/

> Personal research & demonstration project — not financial, investment, or economic policy advice, and not affiliated with the Federal Reserve, the Bureau of Labor Statistics, or FRED.

## What it does

- Pulls monthly U.S. CPI (All Items) data from [FRED](https://fred.stlouisfed.org/series/CPIAUCSL) (Federal Reserve Economic Data) automatically, once a day, via GitHub Actions.
- Computes month-over-month and year-over-year inflation from the raw CPI series.
- Forecasts the next 12 months of YoY inflation with an ARIMA model (order chosen by rolling backtest, not by machine learning — see the FAQ on the live site for why this is a statistical model rather than "AI" in the ML sense).
- Shows a rolling backtest comparing several ARIMA parameterizations against a naive baseline, so the forecast's track record is visible rather than just asserted.

## How it's built

| Piece | What it is |
|---|---|
| `src/download_cpi.py` | Downloads the latest CPI series from FRED into `data/cpi_data.csv` |
| `src/arima_forecast.py` | Fits the chosen ARIMA model and writes a 12-month forecast to `data/yoy_forecast.csv` |
| `src/model_comparison.py` | Compares candidate ARIMA orders on a held-out window (`data/model_comparison.csv`) |
| `src/rolling_backtest.py` | Runs a rolling-origin backtest of each candidate order vs. a naive baseline (`data/rolling_backtest.csv`) |
| `.github/workflows/update-data.yml` | Runs the pipeline above daily; only regenerates the forecast/backtest files when a new CPI release actually changes the data |
| `app.py` | The interactive dashboard, built with [Streamlit](https://streamlit.io) |
| `index.html` | A dependency-free static clone of the dashboard (reads the same CSVs directly) that's what's deployed on GitHub Pages |

There are two live deployments of the same dashboard: this repo's GitHub Pages site (`index.html`), and a Streamlit Community Cloud deployment of `app.py` (linked from this repo's "About" section). Both read from the same `data/*.csv` files, so their content should always match.

## Running it locally

```bash
git clone https://github.com/ppc1004/ai-inflation-forecast.git
cd ai-inflation-forecast
pip install -r requirements.txt

# refresh the data (optional — the repo already ships with data/*.csv)
python src/download_cpi.py
python src/arima_forecast.py
python src/model_comparison.py
python src/rolling_backtest.py

# run the Streamlit app
streamlit run app.py
```

Or just open `index.html` in a browser (served over HTTP, e.g. `python -m http.server`) to use the static version — it fetches the CSVs in `data/` with `fetch()`, so it won't work from a `file://` URL.

## Methodology notes

ARIMA is a classical statistical forecasting method fit on the CPI series' own history — there's no neural network or external training data involved. The confidence band shown on the forecast chart widens further into the future because uncertainty compounds with each additional forecasted step; treat the point forecast as a rough guide, not a precise prediction (see the Model Backtest tab for how the model has actually performed historically, including against a naive baseline).

## License

[MIT](LICENSE)
