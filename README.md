# Short-Term PM2.5 Forecasting in Yerevan

Capstone project:

**Short-Term PM2.5 Forecasting in Yerevan: A Comparative Study of Statistical and Machine Learning Models**  
Areg Khachatryan · Rafayel Shirakyan (Supervisor)

## Repository contents

- **`yerevan_pm25_data_overview.py`** — Data exploration (coverage, missingness, plots). Run: `python yerevan_pm25_data_overview.py`
- **`yerevan_pm25_forecasting.py`** — End-to-end forecasting: preprocessing, features, train/val/test split, baselines, AR(24), linear/ridge models, random forest, MLP, XGBoost, multi-horizon metrics, figures, and CSV summaries. Run: `python yerevan_pm25_forecasting.py` (optional: `--paper` for `paper/figures/` and a local prediction bundle).
- **`sensitivity_analysis.py`** — Preprocessing and lag sensitivity sweep → `sensitivity_preprocessing_lags.csv`. Run: `python sensitivity_analysis.py`
- **`export_paper_assets.py`** — Builds LaTeX tables and `paper/results_summary.json` from the CSV outputs.
- **`download_data.py`** — Downloads the Air Quality Data zip from Google Drive into the project. Run once: `python download_data.py`
- **`docs/METHODOLOGY.md`** — Short methodology reference aligned with the code.
- **`Yerevan_PM25_Data_Overview.ipynb`** — Notebook version of the data overview.
- **`Areg Khachatryan Capstone Dataset Prep.ipynb`** — Early exploratory work.

Data are not in the repo; see **[DATA_README.md](DATA_README.md)** for how to obtain them (script or manual download from Google Drive).

## How to run

1. Clone the repository and go to the project directory.
2. `pip install -r requirements.txt`
3. `python download_data.py`  (downloads and extracts the data)
4. `python yerevan_pm25_data_overview.py`  (exploration) or `python yerevan_pm25_forecasting.py`  (full pipeline and results)

## Thesis / paper outputs

After running the forecasting script, use `python export_paper_assets.py` to regenerate `paper/tables/*.tex` and `paper/results_summary.json`. High-resolution figures: `python yerevan_pm25_forecasting.py --paper`. See [paper/README.md](paper/README.md).

## Data source

Measurements come from [airquality.am](https://airquality.am/en/air-quality/open-data). The dataset used in this repo is distributed via Google Drive; see [DATA_README.md](DATA_README.md) for the link and folder layout.

## Project aim

Compare short-term (1–4 hour ahead) PM2.5 forecasting approaches for Yerevan:

- Baselines (e.g. persistence, moving averages)
- Classical regression and time-series models
- Machine learning (e.g. tree ensembles)

Emphasis on clarity, interpretability, and reproducibility.
