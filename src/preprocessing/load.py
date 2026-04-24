from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.config.settings import DataConfig


def load_city_hourly(config: DataConfig) -> pd.DataFrame:
    city_dir = Path(config.data_root) / "city_avg_hourly"
    files = sorted(glob.glob(str(city_dir / "city_avg_hourly_*.csv")))
    if not files:
        raise FileNotFoundError(f"No city files under {city_dir}")
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    df = raw[raw["city_slug"] == config.city_slug].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def align_hourly_grid(df: pd.DataFrame, *, freq: str = "h") -> pd.DataFrame:
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    full_index = pd.date_range(ts_min, ts_max, freq=freq)
    out = df.set_index("timestamp").reindex(full_index).reset_index()
    out = out.rename(columns={"index": "timestamp"})
    return out

