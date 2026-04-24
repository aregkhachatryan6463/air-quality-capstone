# Methodology (reference)

This note summarizes the active forecasting pipeline implemented in `run_pipeline.py` and `src/pipeline/run.py`.

## Data

Hourly city-average series for Yerevan from the open data release documented in `DATA_README.md`. The modeling target is `avg_pm2.5` (µg/m³). The timeline is aligned to a regular hourly grid before feature construction.

## Preprocessing

Missingness is audited first, then controlled imputation is applied by gap length:
- short gaps: local temporal fills
- medium gaps: constrained interpolation/temporal carry
- long gaps: preserved as missing when uncertainty is high

The pipeline saves missingness and imputation impact summaries to `results/json/results_summary.json`.

## Features

PM2.5 lags at 1, 2, 3, and 24 hours; time/calendar features; and optional PM10 / temperature / pressure / humidity covariates with minimum coverage checks.

## Split

Strictly chronological: 70% train, 15% validation, 15% test.  
Walk-forward validation is also supported (rolling folds with configurable window and step).

## Models

Persistence and seasonal naive baselines; AR(24) frozen; optional SARIMA auto-order; Ridge; Random Forest; XGBoost; optional LightGBM; optional DeepAR adapter.

For horizons 1–4h, supervised models are trained per target horizon and AR(24) uses recursive h-step forecasting.

## Evaluation

MAE, RMSE, and R² on the held-out test period. Bootstrap confidence intervals are reported for MAE. Diebold–Mariano statistics compare paired absolute errors on the test set (HAC variance with one lag). `export_paper_assets.py` exports manuscript tables from pipeline outputs.
