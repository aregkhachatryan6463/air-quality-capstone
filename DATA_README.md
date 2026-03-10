# Getting the air quality data

The **Air Quality Data** folder is not in this repo (it exceeds GitHub’s size limits). It is hosted on **Google Drive** so your supervisor and others can run the project.

---

## Option 1: Download from Google Drive (recommended)

### One-time setup (project author)

1. Zip your local **Air Quality Data** folder (the one that contains `city_avg_hourly/`, `sensors.csv`, etc.).
2. Upload the zip to Google Drive.
3. Right‑click the file → **Share** → set to **“Anyone with the link”** (viewer).
4. Copy the link (e.g. `https://drive.google.com/file/d/XXXXX/view?usp=sharing`).
5. Open **`download_data.py`** and set:
   ```python
   GOOGLE_DRIVE_ZIP_LINK = "https://drive.google.com/uc?id=XXXXX"
   ```
   Use the **file ID** from the link (the `XXXXX` part from `/d/XXXXX/view`).

### For your supervisor (or anyone cloning the repo)

1. Clone the repo and `cd` into the project folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the download script:
   ```bash
   python download_data.py
   ```
   This downloads the zip from Google Drive and extracts it as **Air Quality Data** in the project root.
4. Run the overview script: `python yerevan_pm25_data_overview.py`

---

## Option 2: Manual download from Google Drive

If the project author has already set the link in `download_data.py`, you can still download manually:

1. Open the Google Drive link (ask the author or check the link in `download_data.py` / README).
2. Download the zip and place it in the project root.
3. Unzip so that the folder is named **Air Quality Data** and sits next to `yerevan_pm25_data_overview.py`.

---

## Option 3: Official source

Download from the official open data portal and build the folder yourself:

- **Open data portal**: https://airquality.am/en/air-quality/open-data  

Place files so the project has this structure:

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
├── yerevan_pm25_data_overview.py
├── download_data.py
└── ...
```

Scripts use the path `Air Quality Data` relative to the project root (current working directory when you run them).
