from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.config.settings import FeatureConfig


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month
    out["doy"] = out["timestamp"].dt.dayofyear
    out["heating_season"] = (out["month"].between(10, 12) | out["month"].between(1, 4)).astype(int)
    return out


def _fourier_add(out: pd.DataFrame, cfg: FeatureConfig) -> list[str]:
    if not cfg.fourier_annual:
        return []
    p = int(cfg.fourier_paired_terms)
    cols: list[str] = []
    for j in range(1, p + 1):
        c1 = f"sin_hour_{j}"
        c2 = f"cos_hour_{j}"
        out[c1] = np.sin(2.0 * np.pi * j * out["hour"].astype(float) / 24.0)
        out[c2] = np.cos(2.0 * np.pi * j * out["hour"].astype(float) / 24.0)
        cols.extend([c1, c2])
    for j in range(1, p + 1):
        c1 = f"sin_doy_{j}"
        c2 = f"cos_doy_{j}"
        out[c1] = np.sin(2.0 * np.pi * j * out["doy"].astype(float) / 365.25)
        out[c2] = np.cos(2.0 * np.pi * j * out["doy"].astype(float) / 365.25)
        cols.extend([c1, c2])
    return cols


def _rolling_target(
    out: pd.DataFrame,
    target_col: str,
    windows: Sequence[int],
) -> list[str]:
    s = out[target_col].astype(float)
    tail = s.shift(1)  # strictly before forecast origin
    added: list[str] = []
    for w in windows:
        w = int(w)
        m = int(max(2, min(w // 2, 12)))
        added.append(f"pm25_roll_mean_{w}h")
        out[added[-1]] = tail.rolling(w, min_periods=m).mean()
        added.append(f"pm25_roll_std_{w}h")
        out[added[-1]] = tail.rolling(w, min_periods=m).std()
        added.append(f"pm25_roll_min_{w}h")
        out[added[-1]] = tail.rolling(w, min_periods=m).min()
        added.append(f"pm25_roll_max_{w}h")
        out[added[-1]] = tail.rolling(w, min_periods=m).max()
    return added


def _trend_48h(out: pd.DataFrame, target_col: str) -> str:
    """Ols slope of last 24 observed points before t-1 (proxy: diff over 24h)."""
    c = f"pm25_trend_24h"
    s = out[target_col].astype(float)
    out[c] = s.shift(1) - s.shift(25)
    return c


def _met_lags(out: pd.DataFrame, cov: str, lags: Sequence[int]) -> list[str]:
    added: list[str] = []
    s = out[cov].astype(float) if cov in out.columns else None
    if s is None or s.isna().all():
        return added
    for h in lags:
        name = f"{cov.replace('.', '_')}_lag_{h}h"
        out[name] = s.shift(h)
        added.append(name)
    return added


def _volatility_block(out: pd.DataFrame, target_col: str) -> list[str]:
    s = out[target_col].astype(float)
    d1 = s.shift(1) - s.shift(2)
    c1 = "pm25_diff1h"
    out[c1] = d1
    c2 = "pm25_trailing_vol_24h"
    out[c2] = d1.shift(1).rolling(24, min_periods=8).std()
    c3 = "pm25_ewm_vol_24h"
    out[c3] = s.shift(1).diff().abs().ewm(halflife=12, min_periods=4).mean()
    return [c1, c2, c3]


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


def _interaction_columns(out: pd.DataFrame, met_cov: Sequence[str]) -> list[str]:
    cols: list[str] = []
    for a in ("pm25_lag_1h", "pm25_lag_24h"):
        if a not in out.columns:
            continue
        for cov in met_cov:
            c1, c2 = f"{cov.replace('.', '_')}_lag_1h", f"{cov.replace('.', '_')}_lag_24h"
            for mc in (c1, c2):
                if mc in out.columns and out[mc].notna().any():
                    name = f"ix_{a}_x_{mc}"
                    if name not in out.columns:
                        out[name] = out[a] * out[mc]
                    cols.append(name)
    return list(dict.fromkeys(cols))


def _minimal_lag_set() -> tuple[Sequence[int], Sequence[int], Sequence[int]]:
    """Smaller set for per-district / per-station quick runs."""
    lags = (1, 2, 3, 24)
    roll = (12, 24)
    h = (1, 2, 3, 4)
    return lags, roll, h


def build_feature_table(
    df: pd.DataFrame,
    target_col: str,
    cfg: FeatureConfig,
    *,
    minimal: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    if minimal:
        lag_hours, roll_w, _h = _minimal_lag_set()
    else:
        lag_hours, roll_w = (tuple(cfg.lag_hours), tuple(cfg.rolling_windows))

    out = add_time_features(df)
    out = add_lag_targets(
        out,
        target_col=target_col,
        lag_hours=lag_hours,
        horizons=cfg.horizons,
    )
    fourier_cols = _fourier_add(out, cfg) if not minimal else []
    if minimal:
        roll_w = (12, 24)

    roll_cols = _rolling_target(out, target_col, roll_w)

    tr_list: list[str] = []
    if not minimal:
        tr_list = [_trend_48h(out, target_col)]

    vol_cols: list[str] = []
    if not minimal and cfg.include_volatility_features:
        vol_cols = _volatility_block(out, target_col)
    else:
        s = out[target_col].astype(float)
        out["pm25_diff1h"] = s.shift(1) - s.shift(2)
        vol_cols = ["pm25_diff1h"]

    met_lags1 = (1, 24)
    met_added: list[str] = []
    for cov in cfg.include_covariates:
        if cov in out.columns and out[cov].notna().mean() >= cfg.min_covariate_coverage:
            met_added.extend(_met_lags(out, cov, met_lags1))

    ix: list[str] = []
    if (not minimal) and cfg.include_met_pm25_interactions and met_added:
        ix = _interaction_columns(
            out,
            [
                c
                for c in cfg.include_covariates
                if c in out.columns and out[c].notna().mean() >= cfg.min_covariate_coverage
            ],
        )

    covariates = [
        c
        for c in cfg.include_covariates
        if c in out.columns and out[c].notna().mean() >= cfg.min_covariate_coverage
    ]
    flags = [c for c in out.columns if c.startswith("imputation_flag_") or c.startswith("long_gap_flag_")]

    time_cols: list[str] = [
        "hour",
        "day_of_week",
        "month",
        "heating_season",
    ] + fourier_cols
    if not minimal:
        time_cols.append("doy")

    lag_strs = [f"pm25_lag_{h}h" for h in lag_hours]

    feature_cols = (
        lag_strs
        + roll_cols
        + vol_cols
        + tr_list
        + time_cols
        + list(dict.fromkeys(met_added + covariates + ix))
        + flags
    )
    feature_cols = [c for c in feature_cols if c in out.columns]
    if minimal:
        feature_cols = [c for c in feature_cols if not c.startswith("ix_") or c in out.columns]

    out = out.dropna(subset=feature_cols + ["target_1h"]).reset_index(drop=True)
    return out, feature_cols
