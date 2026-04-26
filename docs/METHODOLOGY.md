# Methodology (reference)

This note tracks the active forecasting pipeline (`run_pipeline.py`, `src/pipeline/run.py`).

## Data

Hourly Yerevan series from the release in `DATA_README.md`. The target is `avg_pm2.5` (µg/m³). Timestamps are aligned to a full hourly index before imputation and features.

## Preprocessing

- **Missingness**: Full audit (per-column rates, gap-length histogram, month×hour and seasonal breakdowns) in `results/json/results_summary.json` under `missingness_audit`.
- **Imputation**: `controlled_impute` — short gaps: interpolation; medium: hour-of-day medians; long: leave missing + `long_gap_flag_`. Residual covariate holes: strictly capped ffill/bfill. Optional ablation: `--imputation-ablation` compares Ridge validation MAE with vs without medium-gap HOD policy.
- **Chronological split**: 70% / 15% / 15% for headline tables.
- **Walk-forward** (primary for stability): Expanding window; configurable test length, step, max folds, optional sliding mode. SARIMA is refit each fold (auto order on training segment; `aic` or `bic` via `ModelConfig.sarima_criterion`). Tree **defaults**: either reuse globally tuned parameters (`walk_forward_refit_hyperopt` false, default) or re-tune per fold (`--walk-forward-refit-tuning`). **DeepAR** in walk-forward: off by default; use `--walk-forward-deepar-folds N` to retrain on the first *N* folds (expensive).

Outputs: `results/tables/walk_forward_results_1h.csv`, `walk_forward_aggregate_by_model.csv`, and comparison to single-split in `validation_chronological_vs_walkforward_1h.csv`.

## Features (leakage-safe)

Past-only PM2.5 lags, rolling means/std/min/max, Fourier pairs for diurnal/annual structure, met covariate lags, PM2.5×met products, and volatility/change. Rows are listwise complete on the feature set. Per-district and per-station “unit” runs can use a **minimal** feature set for speed (`FeatureConfig.use_minimal_feature_set_for_units`).

## Models

Baselines, frozen AR(24), auto SARIMA (pmdarima + statsmodels SARIMAX one-step on test), Ridge, RF, XGBoost, LightGBM, Hyperopt-tuned boosters, optional DeepAR (NeuralForecast) with 90% intervals, mean Winkler, and point metrics.

## Evaluation

- **Point**: MAE, RMSE, R² on the held-out chronological test; multi-horizon direct targets for *h* = 1..4h.
- **Bootstrap**: **Circular block** bootstrap of absolute errors (primary); optional i.i.d. companion values in the same JSON block. Block length is heuristic from error autocorrelation, bounded by `StatsConfig`.
- **Diebold–Mariano**: HAC long-run variance with Bartlett weights and a lag from forecast horizon and sample size; reported per comparison.

## Spatial analysis

`spatial_mae_by_level.csv` and `spatial_level_comparison_mae.png` compare (where computed) **city** test error vs **district**- and **station**-level weighted test MAE from the same model families (lighter stack on units).

## Figures

`results/plots/` (mirrored to `images/`): MAE by horizon, 1h performance with CIs, walk-forward box and mean±s.e. bars, ACF/PACF on training PM2.5, forecast vs actual, optional station map, spatial MAE by level.
