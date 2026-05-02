from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DeepARResult:
    predictions: pd.DataFrame
    backend: str
    details: dict[str, Any]


def winkler_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1) -> float:
    """
    Interval score (Winkler / Gneiting) for central (1-alpha) prediction interval [lo, hi].
    Lower is better; reduces to MAE when interval width -> 0.
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if m.sum() == 0:
        return float("nan")
    y, lo, hi = y[m], lo[m], hi[m]
    width = hi - lo
    pen_lo = (2.0 / alpha) * np.maximum(0, lo - y)
    pen_hi = (2.0 / alpha) * np.maximum(0, y - hi)
    s = 0.5 * width + pen_lo + pen_hi
    return float(np.mean(s))


def train_predict_deepar(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    horizon: int = 1,
    backend: str = "neuralforecast",
    max_steps: int = 300,
    input_size: int | None = None,
    random_seed: int = 42,
) -> DeepARResult:
    """
    Train and predict with DeepAR using a backend abstraction.
    Contract: train_df, future_df use long format: [series_id, timestamp, target, ...covariates]
    """
    if backend not in {"neuralforecast", "fallback_persistence"}:
        raise ValueError(f"Unsupported DeepAR backend: {backend}")
    if backend == "fallback_persistence":
        train = train_df.copy()
        fut = future_df.copy()
        train["series_id"] = train["series_id"].astype(str)
        fut["series_id"] = fut["series_id"].astype(str)
        preds = fut.copy()
        preds["prediction"] = np.nan
        preds["DeepAR_lo_90"] = np.nan
        preds["DeepAR_hi_90"] = np.nan
        for sid, grp in train.groupby("series_id"):
            g = grp.sort_values("timestamp")
            if g.empty:
                continue
            last = float(g["target"].iloc[-1])
            std = float(np.nanstd(g["target"].astype(float).to_numpy()))
            std = max(std, 1e-3)
            mask = preds["series_id"] == sid
            preds.loc[mask, "prediction"] = last
            preds.loc[mask, "DeepAR_lo_90"] = last - 1.64 * std
            preds.loc[mask, "DeepAR_hi_90"] = last + 1.64 * std
        preds["timestamp"] = pd.to_datetime(preds["timestamp"])
        preds = preds.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
        return DeepARResult(
            predictions=preds,
            backend="fallback_persistence",
            details={
                "horizon": int(horizon),
                "max_steps": int(max_steps),
                "input_size": int(input_size) if input_size is not None else None,
                "raw_columns": list(preds.columns),
                "lower_90_col": "DeepAR_lo_90",
                "upper_90_col": "DeepAR_hi_90",
                "fallback": True,
            },
        )
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR
    except Exception:
        return train_predict_deepar(
            train_df,
            future_df,
            horizon=horizon,
            backend="fallback_persistence",
            max_steps=max_steps,
            input_size=input_size,
            random_seed=random_seed,
        )

    ins = int(input_size) if input_size is not None else int(max(24, int(horizon) * 4))
    train = train_df.rename(columns={"series_id": "unique_id", "timestamp": "ds", "target": "y"})
    fut = future_df.rename(columns={"series_id": "unique_id", "timestamp": "ds"})
    model = DeepAR(
        h=int(horizon),
        input_size=ins,
        max_steps=int(max_steps),
        random_seed=random_seed,
    )
    # Prefer lower-case freq to avoid pandas/neuralforecast version inconsistencies.
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(df=train)
    pred_raw: pd.DataFrame
    try:
        pred_raw = nf.predict(futr_df=fut, level=[90])
    except (TypeError, ValueError):
        pred_raw = nf.predict(futr_df=fut)
    pred = pred_raw.reset_index()
    ren: dict[str, str] = {"unique_id": "series_id", "ds": "timestamp"}
    if "DeepAR" in pred.columns:
        ren["DeepAR"] = "prediction"
    pred = pred.rename(columns=ren)
    if "prediction" not in pred.columns:
        pcols = [
            c
            for c in pred.columns
            if str(c).startswith("DeepAR") and "lo" not in c.lower() and "hi" not in c.lower()
        ]
        if pcols:
            pred = pred.rename(columns={pcols[0]: "prediction"})
    if "prediction" not in pred.columns:
        numeric_cols = [c for c in pred.columns if c not in ("series_id", "timestamp")]
        if numeric_cols:
            pred = pred.rename(columns={numeric_cols[0]: "prediction"})
    details: dict[str, Any] = {
        "horizon": int(horizon),
        "raw_columns": list(pred.columns),
        "max_steps": int(max_steps),
        "input_size": ins,
    }
    lo_cols = [c for c in pred.columns if "lo" in c.lower() and "90" in c]
    hi_cols = [c for c in pred.columns if "hi" in c.lower() and "90" in c]
    if lo_cols:
        details["lower_90_col"] = lo_cols[0]
    if hi_cols:
        details["upper_90_col"] = hi_cols[0]
    pred = pred.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return DeepARResult(predictions=pred, backend=backend, details=details)
