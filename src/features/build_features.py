from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config.settings import FeatureConfig


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month
    out["heating_season"] = (out["month"].between(10, 12) | out["month"].between(1, 4)).astype(int)
    return out


def add_lag_targets(
    df: pd.DataFrame,
    *,
    target_col: str,
    lag_hours: Sequence[int],
    horizons: Sequence[int],
) -> pd.DataFrame:
    out = df.copy()
    for h in lag_hours:
        out[f"pm25_lag_{h}h"] = out[target_col].shift(h)
    for h in horizons:
        out[f"target_{h}h"] = out[target_col].shift(-h)
    return out


def build_feature_table(df: pd.DataFrame, target_col: str, cfg: FeatureConfig) -> tuple[pd.DataFrame, list[str]]:
    out = add_time_features(df)
    out = add_lag_targets(out, target_col=target_col, lag_hours=cfg.lag_hours, horizons=cfg.horizons)
    covariates = [
        c
        for c in cfg.include_covariates
        if c in out.columns and out[c].notna().mean() >= cfg.min_covariate_coverage
    ]
    flags = [c for c in out.columns if c.startswith("imputation_flag_") or c.startswith("long_gap_flag_")]
    feature_cols = [f"pm25_lag_{h}h" for h in cfg.lag_hours] + [
        "hour",
        "day_of_week",
        "month",
        "heating_season",
    ] + covariates + flags
    out = out.dropna(subset=feature_cols + ["target_1h"]).reset_index(drop=True)
    return out, feature_cols

