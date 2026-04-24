from __future__ import annotations

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
        }
    tmp = df.copy()
    tmp["year"] = tmp["timestamp"].dt.year
    tmp["month"] = tmp["timestamp"].dt.month
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
    return out

