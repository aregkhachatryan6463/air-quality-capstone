# Methodology (reference)

This note summarizes the forecasting pipeline implemented in `yerevan_pm25_forecasting.py`. If the code changes, update this file to match.

## Data

Hourly city-average series for Yerevan from the open data release documented in `DATA_README.md`. The modeling target is `avg_pm2.5` (µg/m³). The timeline is expanded to a regular hourly index; missing hours are filled before feature construction.

## Preprocessing

Numeric columns are forward-filled then back-filled along time. PM2.5 values above the 99.5th percentile are capped for the main experiments (configurable). A rolling 24-hour IQR rule is used only to count flagged hours in logs.

## Features

PM2.5 lags at 1, 2, 3, and 24 hours; hour, day-of-week, month; a heating-season indicator (Oct–Dec and Jan–Apr); and optional PM10 / temperature / pressure / humidity columns if at least half of values are present after imputation.

## Split

Strictly chronological: 70% train, 15% validation, 15% test. No random shuffling.

## Models

Persistence, seasonal naive (24-hour cycle), 24-hour rolling mean, AR(24) with intercept fitted on the training segment only (frozen coefficients, one-step forecasts), ordinary least squares, Ridge, random forest, a scaled MLP, and XGBoost when installed. For horizons 2–4 h, supervised models are retrained on the corresponding target column; the AR model is included for 1 h in the horizon table.

## Evaluation

MAE, RMSE, and R² on the held-out test period. Diebold–Mariano statistics compare paired absolute errors on the test set (HAC variance with one lag). `export_paper_assets.py` can add bootstrap intervals for mean absolute error from saved test predictions.
