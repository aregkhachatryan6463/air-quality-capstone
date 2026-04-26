from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _gap_lengths(mask: pd.Series) -> list[int]:
    lengths: list[int] = []
    run = 0
    for is_missing in mask.to_numpy():
        if is_missing:
            run += 1
        elif run:
            lengths.append(run)
            run = 0
    if run:
        lengths.append(run)
    return lengths


def _gap_hist_binned(lengths: list[int], max_bin: int = 48) -> dict[str, int]:
    if not lengths:
        return {}
    out: dict[str, int] = {}
    c = Counter(lengths)
    for g, n in sorted(c.items()):
        if g <= max_bin:
            out[str(g)] = int(n)
        else:
            out[f">={max_bin + 1}"] = out.get(f">={max_bin + 1}", 0) + int(n)
    return out


def audit_missingness(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in cols:
        if col not in df.columns:
            continue
        miss = df[col].isna()
        gaps = _gap_lengths(miss)
        out[col] = {
            "missing_fraction": float(miss.mean()),
            "n_missing": int(miss.sum()),
            "max_gap": int(max(gaps)) if gaps else 0,
            "p95_gap": float(np.quantile(gaps, 0.95)) if gaps else 0.0,
            "n_gaps": int(len(gaps)),
            "gap_length_histogram": _gap_hist_binned(gaps, max_bin=48),
        }
    tmp = df.copy()
    tmp["year"] = tmp["timestamp"].dt.year
    tmp["month"] = tmp["timestamp"].dt.month
    tmp["hour"] = tmp["timestamp"].dt.hour
    tmp["doy"] = tmp["timestamp"].dt.dayofyear
    tmp["winter_summer"] = np.where(
        (tmp["month"] >= 11) | (tmp["month"] <= 3), "winter", "summer"
    )
    monthly = {}
    for col in cols:
        if col not in tmp.columns:
            continue
        m = (
            tmp.groupby(["year", "month"])[col]
            .apply(lambda s: float(s.isna().mean()))
            .reset_index(name="missing_fraction")
        )
        monthly[col] = m.to_dict(orient="records")
    out["monthly_missingness"] = monthly

    # Month × hour heatmap (missing rate)
    mhour: dict[str, Any] = {}
    for col in cols:
        if col not in tmp.columns:
            continue
        p = (
            tmp.groupby(["month", "hour"])[col]
            .apply(lambda s: float(s.isna().mean()))
            .unstack(level=0)
        )
        mhour[col] = p.to_dict() if not p.empty else {}
    out["month_hour_missing_rate"] = mhour

    seasonal = {}
    for col in cols:
        if col not in tmp.columns:
            continue
        g = tmp.groupby("winter_summer")[col].apply(lambda s: float(s.isna().mean()))
        seasonal[col] = g.to_dict()
    out["winter_summer_missing_fraction"] = seasonal

    doy = {}
    for col in cols:
        if col not in tmp.columns:
            continue
        g2 = (
            tmp.groupby("doy")[col]
            .apply(lambda s: float(s.isna().mean()))
            .reset_index(name="missing_fraction")
        )
        doy[col] = g2.to_dict(orient="records")
    out["day_of_year_missingness"] = doy

    return out
