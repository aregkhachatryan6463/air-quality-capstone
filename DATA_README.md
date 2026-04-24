# Data

The **Air Quality Data** folder is not stored in this repository (it exceeds GitHub size limits). It is downloaded from Google Drive.

## Download and run

1. **Install dependencies:**  
   `pip install -r requirements.txt`

2. **Run the main pipeline (recommended):**  
   `python run_pipeline.py`  
   This checks for data and downloads it automatically if missing.

3. **Optional manual download only:**  
   `python download_data.py`

## Manual download

Alternatively, download the zip and unzip it under:

`data/raw/Air Quality Data/`

- [Air Quality Data (Google Drive)](https://drive.google.com/file/d/1QaDT5_XFKUMXbYoZLO8BsfzCstlNgwnp/view?usp=drive_link)

## Data structure

After extraction, the key layout is:

```
data/raw/Air Quality Data/
├── README.txt
├── LICENSE.txt
├── sensors.csv
├── city_avg_daily.csv
├── city_avg_hourly/
├── station_avg_hourly/
├── sensor_avg_hourly/
└── measurements/
```

Data are from the [airquality.am](https://airquality.am/en/air-quality/open-data) open data portal. Schema and license are described in the README and LICENSE inside the data folder.
