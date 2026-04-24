from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DeepARResult:
    predictions: pd.DataFrame
    backend: str
    details: dict[str, Any]


def train_predict_deepar(
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    *,
    horizon: int,
    backend: str = "neuralforecast",
) -> DeepARResult:
    """
    Train and predict with DeepAR using a backend abstraction.
    Contract:
      - train_df, future_df use long format: [series_id, timestamp, target, ...covariates]
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

    train = train_df.rename(columns={"series_id": "unique_id", "timestamp": "ds", "target": "y"})
    fut = future_df.rename(columns={"series_id": "unique_id", "timestamp": "ds"})
    model = DeepAR(h=horizon, input_size=max(24, horizon * 4), max_steps=200, random_seed=42)
    nf = NeuralForecast(models=[model], freq="H")
    nf.fit(df=train)
    pred = nf.predict(futr_df=fut).reset_index()
    pred = pred.rename(columns={"unique_id": "series_id", "ds": "timestamp", "DeepAR": "prediction"})
    return DeepARResult(predictions=pred, backend=backend, details={"horizon": horizon})

