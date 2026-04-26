# Short-Term PM2.5 Forecasting in Yerevan

Capstone project:
**Short-Term PM2.5 Forecasting in Yerevan: A Comparative Study of Statistical and Machine Learning Models**  
Areg Khachatryan · Rafayel Shirakyan (Supervisor)

## Refactored structure

- `data/raw/` - downloaded source data
- `data/processed/` - reusable processed feature tables
- `src/download/` - modular data fetching
- `src/preprocessing/` - loading, missingness audits, imputation
- `src/features/` - feature and target engineering
- `src/models/` - ARIMA/SARIMA, tree models, LightGBM, DeepAR adapter, district clustering
- `src/evaluation/` - metrics, statistical tests, validation schemes
- `src/pipeline/` - orchestration logic
- `results/` - generated outputs only: **`results/plots/`** (all figures), **`results/tables/`** (all CSVs), **`results/json/`** (summaries). See `results/README.md`.
- `images/` - optional copies of key figures for the thesis / LaTeX (mirrored from `results/plots/` on each run and by `export_paper_assets.py`). See `images/README.md`.
- `notebooks/` - optional interactive exploration; pipeline entrypoint is `run_pipeline.py` (see `notebooks/README.md`).
- `scripts/` - optional utilities (e.g. `data_quality_supplement.py`). See `scripts/README.md`.
- `figures_data_quality/` - supplementary quality-audit outputs
- `docs/` - methodology and project notes
- `manuscript/` - paper source (`main.tex`, bibliography)
- `paper/` - paper-ready exported tables and summary

## Single runnable pipeline

The project is now runnable from one main script:

```bash
python run_pipeline.py
```

Fast smoke test:

```bash
python run_pipeline.py --skip-download --no-tuning --disable-deepar --disable-sarima --disable-walk-forward
```

This command will:
1. Download data if missing (`from download_data import download_data`)
2. Audit missingness and apply controlled imputation
3. Build features and chronological split
4. Train baseline, ARIMA-class, and tree models (with optional Hyperopt tuning)
5. Run walk-forward validation
6. Write CSVs to `results/tables/`, figures to `results/plots/`, and `results/json/results_summary.json`

## Core model scope

- Baselines: persistence, seasonal naive
- Classical: frozen AR(24), SARIMA auto-order search + rolling one-step
- Machine learning: Ridge, Random Forest, XGBoost, LightGBM (auto-skips if LightGBM env is incompatible)
- Deep learning extension: DeepAR adapter (NeuralForecast backend)
- MLP has been removed from the active pipeline.

## Environment

- Python 3.9+
- Install deps:
  - `pip install -r requirements.txt`
  - or pinned: `pip install -r requirements-lock.txt`
- Optional run controls:
  - `--disable-deepar`, `--disable-sarima`, `--disable-walk-forward`, `--no-tuning`, `--skip-download`

## Legacy scripts

Legacy training scripts were removed to keep the repository clean and reproducible around one execution path.  
Use `run_pipeline.py` for model training/evaluation and `export_paper_assets.py` for manuscript table exports.
