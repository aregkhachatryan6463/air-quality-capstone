"""
Short-Term PM2.5 in Yerevan – Data Overview

This script is the starting point for the capstone project.

Goals:
- Understand the structure and quality of the airquality.am data for Yerevan.
- Identify risks (missing data, outliers, coverage issues).
- Prepare a clean, well-documented dataset for short-term PM2.5 forecasting.

In this script we will:
- Load city-level hourly data for Yerevan from the official data dump.
- Explore basic distributions, time coverage, and missingness.
- Flag risky parts of the data and discuss how to handle them.
- Sketch how these observations will inform baseline models in the next step.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Base path for the data (relative to project folder)
BASE_PATH = os.path.join(os.getcwd(), "Air Quality Data")
city_hourly_dir = os.path.join(BASE_PATH, "city_avg_hourly")

print(f"Using data directory: {city_hourly_dir}")

# ---------------------------------------------------------------------------
# Load all available city-level hourly files and filter for Yerevan
# ---------------------------------------------------------------------------
city_files = sorted(glob.glob(os.path.join(city_hourly_dir, "city_avg_hourly_*.csv")))
if not city_files:
    raise FileNotFoundError(
        "No city_avg_hourly_*.csv files found. Check BASE_PATH and directory structure."
    )

print(f"Found {len(city_files)} city_avg_hourly files:")
print("\n".join(city_files))

dfs = []
for f in city_files:
    df_tmp = pd.read_csv(f)
    dfs.append(df_tmp)

city_all = pd.concat(dfs, ignore_index=True)
city_all["timestamp"] = pd.to_datetime(city_all["timestamp"])

# Focus on Yerevan only
df_yerevan = city_all[city_all["city_slug"] == "yerevan"].copy()
df_yerevan = df_yerevan.sort_values("timestamp").reset_index(drop=True)

print(f"\nLoaded {len(df_yerevan)} rows for Yerevan.")
print(f"Time span: {df_yerevan['timestamp'].min()} to {df_yerevan['timestamp'].max()}")
print("\nFirst rows:")
print(df_yerevan.head())

# ---------------------------------------------------------------------------
# Basic structure and missingness per column
# ---------------------------------------------------------------------------
print("\nDataFrame info (types, non-null counts):")
df_yerevan.info()

print("\nFraction of missing values per column:")
missing_fraction = df_yerevan.isna().mean().sort_values(ascending=False)
print(missing_fraction)

# ---------------------------------------------------------------------------
# Check time coverage and identify missing hours in the Yerevan series
# ---------------------------------------------------------------------------
ts_min = df_yerevan["timestamp"].min()
ts_max = df_yerevan["timestamp"].max()
full_index = pd.date_range(ts_min, ts_max, freq="h")

observed_hours = df_yerevan["timestamp"].nunique()
expected_hours = len(full_index)
missing_hours = sorted(set(full_index) - set(df_yerevan["timestamp"]))

print(f"\nExpected hourly timestamps: {expected_hours}")
print(f"Observed hourly timestamps: {observed_hours}")
print(
    f"Missing hourly timestamps: {len(missing_hours)} ({len(missing_hours) / expected_hours:.2%} of the period)"
)

if missing_hours:
    print("\nFirst 10 missing timestamps:")
    for ts in missing_hours[:10]:
        print(ts)

# ---------------------------------------------------------------------------
# Time series plot of city-level average PM2.5 for Yerevan
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df_yerevan["timestamp"], df_yerevan["avg_pm2.5"], linewidth=0.5, color="tab:red")
plt.title("Yerevan City-Level PM2.5 Over Time (Hourly Averages)")
plt.xlabel("Time")
plt.ylabel("avg_pm2.5 (µg/m³)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Distribution and outlier inspection for PM2.5
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    df_yerevan["avg_pm2.5"].dropna(), bins=60, color="tab:red", alpha=0.7
)
axes[0].set_title("Histogram of avg_pm2.5 (Yerevan)")
axes[0].set_xlabel("avg_pm2.5 (µg/m³)")

axes[1].boxplot(
    df_yerevan["avg_pm2.5"].dropna(),
    vert=False,
    patch_artist=True,
    boxprops=dict(facecolor="tab:red"),
)
axes[1].set_title("Boxplot of avg_pm2.5 (Yerevan)")

plt.tight_layout()
plt.show()

print("Top 10 highest avg_pm2.5 values:")
print(
    df_yerevan[["timestamp", "avg_pm2.5"]]
    .sort_values("avg_pm2.5", ascending=False)
    .head(10)
)

# ---------------------------------------------------------------------------
# Data quality summary (for preprocessing and modeling)
# ---------------------------------------------------------------------------
print("""
Data quality notes:
- Missing timestamps: treat via interpolation or dropping short gaps as appropriate.
- Missing values in pollutants/meteorology: exclude or impute (e.g. forward-fill, rolling median).
- Extreme PM2.5 values: consider capping or robust metrics in modeling.

For the full forecasting pipeline (preprocessing, baselines, models, evaluation), run:
  python yerevan_pm25_forecasting.py
""")
