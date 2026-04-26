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
    if backend != "neuralforecast":
        raise ValueError(f"Unsupported DeepAR backend: {backend}")
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR
    except ImportError as exc:
        raise RuntimeError(
            "DeepAR backend requires neuralforecast. Install dependencies first."
        ) from exc

    ins = int(input_size) if input_size is not None else int(max(24, int(horizon) * 4))
    train = train_df.rename(columns={"series_id": "unique_id", "timestamp": "ds", "target": "y"})
    fut = future_df.rename(columns={"series_id": "unique_id", "timestamp": "ds"})
    model = DeepAR(
        h=int(horizon),
        input_size=ins,
        max_steps=int(max_steps),
        random_seed=random_seed,
    )
    nf = NeuralForecast(models=[model], freq="H")
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
        pcols = [c for c in pred.columns if str(c).startswith("DeepAR") and "lo" not in c.lower() and "hi" not in c.lower()]
        if pcols:
            pred = pred.rename(columns={pcols[0]: "prediction"})
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
    return DeepARResult(predictions=pred, backend=backend, details=details)
