from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import DataConfig


def _segment_ids(mask: pd.Series) -> pd.Series:
    return (mask != mask.shift(fill_value=False)).cumsum()


def controlled_impute(df: pd.DataFrame, cols: list[str], config: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    meta: dict[str, Any] = {"per_column": {}}
    for col in cols:
        if col not in out.columns:
            continue
        miss = out[col].isna()
        seg = _segment_ids(miss)
        gap_len = miss.groupby(seg).transform("sum")

        short = miss & (gap_len <= config.short_gap_max)
        medium = miss & (gap_len > config.short_gap_max) & (gap_len <= config.medium_gap_max)
        long_ = miss & (gap_len > config.medium_gap_max)

        # Short gaps: causal interpolation.
        s = out[col].interpolate(limit=config.short_gap_max, limit_direction="forward")
        out.loc[short, col] = s.loc[short]

        # Medium gaps: hour-of-day median (station/city safe fallback).
        if medium.any():
            hod_med = out.groupby(out["timestamp"].dt.hour)[col].transform("median")
            out.loc[medium, col] = hod_med.loc[medium]

        # Long gaps: leave missing; model pipeline will drop/flag.
        out[f"imputation_flag_{col}"] = 0
        out.loc[short | medium, f"imputation_flag_{col}"] = 1
        out[f"long_gap_flag_{col}"] = 0
        out.loc[long_, f"long_gap_flag_{col}"] = 1

        meta["per_column"][col] = {
            "short_imputed": int(short.sum()),
            "medium_imputed": int(medium.sum()),
            "long_left_missing": int(long_.sum()),
            "final_missing": int(out[col].isna().sum()),
            "final_missing_fraction": float(out[col].isna().mean()),
        }

    # Conservative fallback for non-target covariates: time-bounded ffill/bfill.
    for col in cols:
        if col not in out.columns:
            continue
        if out[col].isna().any():
            out[col] = out[col].ffill(limit=config.short_gap_max).bfill(limit=config.short_gap_max)

    return out, meta

