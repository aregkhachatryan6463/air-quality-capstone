# Getting the air quality data

The **Air Quality Data** folder is not stored in this repo (it exceeds GitHub’s size limits).

## Option 1: Official source (recommended)

Download the data from the official open data portal:

- **Open data portal**: https://airquality.am/en/air-quality/open-data  

Download and place the contents so your project has this structure:

```
Air Quality Capstone project/
├── Air Quality Data/
│   ├── README.txt
│   ├── LICENSE.txt
│   ├── sensors.csv
│   ├── city_avg_daily.csv
│   ├── city_avg_hourly/
│   │   └── city_avg_hourly_2019.csv, ... city_avg_hourly_2026.csv
│   ├── station_avg_hourly/
│   ├── sensor_avg_hourly/
│   └── measurements/
├── Yerevan_PM25_Data_Overview.ipynb
└── ...
```

The notebooks expect the `Air Quality Data` folder to be in the same directory as the notebooks. Paths are set in the first code cell (e.g. `BASE_PATH = ... 'Air Quality Data'`).

## Option 2: You already have the data locally

If you have cloned or copied this repo and already have an **Air Quality Data** folder on your machine (e.g. from a previous download), just keep it in the project root. The notebooks will use it; only the folder is excluded from Git so the repo stays under GitHub’s size limit.
