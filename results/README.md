# Pipeline outputs (layout)

`run_pipeline.py` writes all generated artifacts in three places under this folder.  
`plots/`, `tables/`, and `json/` are the only canonical locations: avoid saving CSVs or figures directly under `results/` (root-level outputs are gitignored to prevent clutter). This README documents the **intended structure** for a clean repository and reports.

| Subfolder | Contents |
|-----------|----------|
| **`plots/`** | All figures (PNG), e.g. `mae_by_horizon.png`, `model_performance_1h_ci.png`, `station_groups_map.png`, `forecast_vs_actual_best_1h.png`, `walk_forward_mae_stability.png` |
| **`tables/`** | All tabular CSV results, e.g. `forecast_results_1h.csv`, `forecast_results_by_horizon.csv`, `walk_forward_results_1h.csv`, `validation_chronological_vs_walkforward_1h.csv` (chronological vs mean walk-forward MAE by model), `diebold_mariano_test.csv`, `forecast_results_district_units_1h.csv`, `forecast_results_station_units_1h.csv` |
| **`json/`** | JSON summaries: `results_summary.json` (includes `imputation_sensitivity`, `validation_chronological_vs_walkforward`, SARIMA orders, DeepAR metadata), `data_manifest.json`, `split_info.json` |

After a full run, use **`results/json/results_summary.json`** for a machine-readable index and **`export_paper_assets.py`** to build LaTeX tables for the paper from the CSVs in **`tables/`**.
