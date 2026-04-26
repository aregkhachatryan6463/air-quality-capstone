from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def save_acf_pacf_plot(
    y: np.ndarray,
    out_path: Path,
    *,
    nlags: int = 72,
    log1p: bool = True,
    mirror_path: Path | None = None,
) -> bool:
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    except ImportError:
        return False
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 200:
        return False
    if log1p:
        y = np.log1p(np.clip(y, 0, None))
    nlags = int(min(nlags, len(y) // 2 - 1))
    if nlags < 10:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_acf(y, lags=nlags, ax=axes[0], title="ACF (train; log1p(PM2.5))", alpha=0.05)
    plot_pacf(y, lags=nlags, ax=axes[1], title="PACF (train; log1p(PM2.5))", method="ywm", alpha=0.05)
    fig.suptitle("Yerevan hourly PM2.5 serial dependence (chronological train only)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    if mirror_path is not None:
        mirror_path = Path(mirror_path)
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mirror_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_permutation_importance_csv(
    X: pd.DataFrame,
    y: np.ndarray,
    model: Any,
    out_csv: Path,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    n_samples: int | None = 3000,
) -> pd.DataFrame | None:
    if n_samples and len(y) > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=n_samples, replace=False)
        Xs = X.iloc[idx]
        ys = y[idx]
    else:
        Xs, ys = X, y
    m = np.isfinite(ys)
    for c in Xs.columns:
        m = m & np.isfinite(Xs[c].to_numpy(dtype=float))
    Xs = Xs.loc[m]
    ys = np.asarray(ys, dtype=float)[m]
    if len(ys) < 100:
        return None
    r = permutation_importance(
        model,
        Xs,
        ys,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    df = pd.DataFrame(
        {
            "feature": list(Xs.columns),
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    )
    df = df.sort_values("importance_mean", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
