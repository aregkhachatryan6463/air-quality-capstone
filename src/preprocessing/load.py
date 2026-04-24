from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.config.settings import DataConfig

YEREVAN_DISTRICT_ORDER = [
    "ajapnyak",
    "arabkir",
    "avan",
    "davtashen",
    "erebuni",
    "kanaker-zeytun",
    "kentron",
    "malatia-sebastia",
    "nork-marash",
    "nor-nork",
    "nubarashen",
    "shengavit",
]


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


def load_station_metadata(config: DataConfig) -> pd.DataFrame | None:
    if config.station_metadata_path and config.station_metadata_path.exists():
        return pd.read_csv(config.station_metadata_path)
    sensors = Path(config.data_root) / "sensors.csv"
    if sensors.exists():
        return pd.read_csv(sensors)
    return None


def _filter_yerevan_station_meta(meta: pd.DataFrame, city_slug: str) -> pd.DataFrame:
    out = meta.copy()
    if "city_slug" in out.columns:
        out["city_slug"] = out["city_slug"].astype(str).str.strip().str.lower()
        out = out[out["city_slug"] == city_slug.lower()]
    if "district_slug" in out.columns:
        out["district_slug"] = out["district_slug"].astype(str).str.strip().str.lower()
        out = out[out["district_slug"].isin(set(YEREVAN_DISTRICT_ORDER))]
    if "station_id" in out.columns:
        out = out.drop_duplicates(subset=["station_id"], keep="first")
    return out.reset_index(drop=True)


def load_yerevan_station_metadata(config: DataConfig) -> pd.DataFrame | None:
    meta = load_station_metadata(config)
    if meta is None or meta.empty:
        return meta
    return _filter_yerevan_station_meta(meta, config.city_slug)


def load_station_hourly(config: DataConfig, *, station_ids: set[str] | None = None) -> pd.DataFrame:
    station_dir = Path(config.data_root) / "station_avg_hourly"
    files = sorted(glob.glob(str(station_dir / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No station hourly files under {station_dir}")
    raw = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    ts_col = next((c for c in ("timestamp", "datetime", "date") if c in raw.columns), None)
    if ts_col is None:
        raise ValueError("Station data missing timestamp-like column")
    station_col = next((c for c in ("station_id", "sensor_id", "station", "sensor") if c in raw.columns), None)
    if station_col is None:
        raise ValueError("Station data missing station identifier column")
    if config.target_col not in raw.columns:
        target_alt = next((c for c in ("pm2_5", "pm25", "pm2.5") if c in raw.columns), None)
        if target_alt is None:
            raise ValueError(f"Station data missing target column {config.target_col}")
        raw = raw.rename(columns={target_alt: config.target_col})

    keep = [ts_col, station_col, config.target_col]
    for cov in ("avg_pm10", "avg_temperature", "avg_pressure", "avg_humidity"):
        if cov in raw.columns:
            keep.append(cov)
    df = raw[keep].copy()
    df = df.rename(columns={ts_col: "timestamp", station_col: "station_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["station_id"] = df["station_id"].astype(str)
    if station_ids is not None:
        df = df[df["station_id"].isin(station_ids)].copy()
    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df


def aggregate_city_from_station_districts(
    station_hourly: pd.DataFrame,
    district_mapping: pd.DataFrame,
    *,
    target_col: str = "avg_pm2.5",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if district_mapping.empty:
        raise ValueError("District mapping is empty")
    key_col = "station_id" if "station_id" in district_mapping.columns else district_mapping.columns[0]
    map_df = district_mapping[[key_col, "district_id"]].rename(columns={key_col: "station_id"}).copy()
    map_df["station_id"] = map_df["station_id"].astype(str)
    station_hourly = station_hourly.copy()
    station_hourly["station_id"] = station_hourly["station_id"].astype(str)
    merged = station_hourly.merge(map_df, on="station_id", how="inner")
    if merged.empty:
        raise ValueError("No station rows matched district mapping")

    agg_cols = [c for c in (target_col, "avg_pm10", "avg_temperature", "avg_pressure", "avg_humidity") if c in merged.columns]
    district_hourly = (
        merged.groupby(["timestamp", "district_id"], as_index=False)[agg_cols]
        .mean(numeric_only=True)
        .sort_values(["timestamp", "district_id"])
        .reset_index(drop=True)
    )
    city_hourly = (
        district_hourly.groupby("timestamp", as_index=False)[agg_cols]
        .mean(numeric_only=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return district_hourly, city_hourly

