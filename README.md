# Short-Term PM2.5 Forecasting in Yerevan

Capstone project:

**Short-Term PM2.5 Forecasting in Yerevan: A Comparative Study of Statistical and Machine Learning Models**  
Areg Khachatryan · Rafayel Shirakyan (Supervisor)

## Repository contents

- **`yerevan_pm25_data_overview.py`** — Loads city-level hourly data for Yerevan, summarizes coverage and missingness, and produces PM2.5 visualizations. Run: `python yerevan_pm25_data_overview.py`
- **`download_data.py`** — Downloads the Air Quality Data zip from Google Drive and extracts it into the project. Run once: `python download_data.py`
- **`Yerevan_PM25_Data_Overview.ipynb`** — Notebook version of the data overview.
- **`Areg Khachatryan Capstone Dataset Prep.ipynb`** — Exploratory work: Yerevan dataset construction and initial models.

Data are not in the repo; see **[DATA_README.md](DATA_README.md)** for how to obtain them (script or manual download from Google Drive).

## How to run

1. Clone the repository and go to the project directory.
2. `pip install -r requirements.txt`
3. `python download_data.py`  (downloads and extracts the data)
4. `python yerevan_pm25_data_overview.py`

## Data source

Measurements come from [airquality.am](https://airquality.am/en/air-quality/open-data). The dataset used in this repo is distributed via Google Drive; see [DATA_README.md](DATA_README.md) for the link and folder layout.

## Project aim

Compare short-term (1–4 hour ahead) PM2.5 forecasting approaches for Yerevan:

- Baselines (e.g. persistence, moving averages)
- Classical regression and time-series models
- Machine learning (e.g. tree ensembles)

Emphasis on clarity, interpretability, and reproducibility.
