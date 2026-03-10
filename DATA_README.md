# Data

The **Air Quality Data** folder is not stored in this repository (it exceeds GitHub size limits). It is provided via Google Drive.

## Download and run

1. **Install dependencies:**  
   `pip install -r requirements.txt`

2. **Download the data:**  
   `python download_data.py`  
   This fetches the data zip from Google Drive and extracts it as `Air Quality Data` in the project root.

3. **Run the overview:**  
   `python yerevan_pm25_data_overview.py`

## Manual download

Alternatively, download the zip from Google Drive and unzip it in the project root so that the folder is named **Air Quality Data**:

- [Air Quality Data (Google Drive)](https://drive.google.com/file/d/1QaDT5_XFKUMXbYoZLO8BsfzCstlNgwnp/view?usp=drive_link)

## Data structure

After extraction, the layout should be:

```
Air Quality Data/
├── README.txt, LICENSE.txt
├── sensors.csv, city_avg_daily.csv
├── city_avg_hourly/   (city_avg_hourly_2019.csv … city_avg_hourly_2026.csv)
├── station_avg_hourly/
├── sensor_avg_hourly/
└── measurements/
```

Data are from the [airquality.am](https://airquality.am/en/air-quality/open-data) open data portal. Schema and license are described in the README and LICENSE inside the data folder.
