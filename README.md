# Short-Term PM2.5 Forecasting in Yerevan

This repository contains the code and notebooks for the capstone project:

> **Short-Term PM2.5 Forecasting in Yerevan: A Comparative Study of Statistical and Machine Learning Models**  
> Author: **Areg Khachatryan**  
> Supervisor: **Rafayel Shirakyan**

## Contents

- `Yerevan_PM25_Data_Overview.ipynb` – data overview notebook that loads city-level hourly data for Yerevan from the official `airquality.am` dump, explores coverage/missingness, and visualizes PM2.5 distributions.
- `Areg Khachatryan Capstone Dataset Prep.ipynb` – earlier exploratory notebook for building a Yerevan dataset and trying initial models.
- **Data** – The large `Air Quality Data/` folder is **not** in the repo (GitHub size limits). See **[DATA_README.md](DATA_README.md)** for how to get the data from [airquality.am](https://airquality.am/en/air-quality/open-data).

## Data source

All measurement data comes from the open data portal of **airquality.am**. Because the full dataset is too large for GitHub, it is not stored here. Download instructions and folder layout are in **[DATA_README.md](DATA_README.md)**. Schema and license are described in the data README/LICENSE once you have the folder.

## Goal (high level)

The project aims to build and compare short-term (1–4 hour ahead) PM2.5 forecasting models for **Yerevan**, using:

- Simple baselines (persistence, moving averages)
- Classical regression / time-series models
- Machine learning models (e.g., tree ensembles)

with a focus on **methodological clarity, interpretability, and reproducibility**, rather than very complex deep learning architectures.

