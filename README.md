# Short-Term PM2.5 Forecasting in Yerevan

This repository contains the code and notebooks for the capstone project:

> **Short-Term PM2.5 Forecasting in Yerevan: A Comparative Study of Statistical and Machine Learning Models**  
> Author: **Areg Khachatryan**  
> Supervisor: **Rafayel Shirakyan**

## Contents

- `yerevan_pm25_data_overview.py` – data overview script (run with `python yerevan_pm25_data_overview.py`); loads city-level hourly data for Yerevan, explores coverage/missingness, and visualizes PM2.5.
- `download_data.py` – downloads the Air Quality Data zip from Google Drive into the project (run once after setting the link in the script; see [DATA_README.md](DATA_README.md)).
- `Yerevan_PM25_Data_Overview.ipynb` – same content as the overview script, in notebook form.
- `Areg Khachatryan Capstone Dataset Prep.ipynb` – earlier exploratory notebook for building a Yerevan dataset and trying initial models.
- **Data** – The large `Air Quality Data/` folder is **not** in the repo (GitHub size limits). It is hosted on **Google Drive**. See **[DATA_README.md](DATA_README.md)** for how to get the data (download script or manual), or use the official [airquality.am](https://airquality.am/en/air-quality/open-data) source.

## Running the project (e.g. for your supervisor)

1. Clone the repo and install dependencies: `pip install -r requirements.txt`
2. Download the data: run `python download_data.py` (after the project author has set the Google Drive link in that script), or follow the manual steps in [DATA_README.md](DATA_README.md).
3. Run the overview script: `python yerevan_pm25_data_overview.py`

## Data source

All measurement data comes from the open data portal of **airquality.am**. The full dataset is hosted on **Google Drive** for this repo; download instructions and folder layout are in **[DATA_README.md](DATA_README.md)**. Schema and license are in the data README/LICENSE once you have the folder.

## Goal (high level)

The project aims to build and compare short-term (1–4 hour ahead) PM2.5 forecasting models for **Yerevan**, using:

- Simple baselines (persistence, moving averages)
- Classical regression / time-series models
- Machine learning models (e.g., tree ensembles)

with a focus on **methodological clarity, interpretability, and reproducibility**, rather than very complex deep learning architectures.

