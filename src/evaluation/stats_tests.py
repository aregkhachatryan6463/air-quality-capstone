from __future__ import annotations

import math
from typing import Any

import numpy as np


def diebold_mariano(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pa) & np.isfinite(pb)
    y = y_true[mask]
    ea = y - pa[mask]
    eb = y - pb[mask]
    d = np.abs(ea) - np.abs(eb)
    n = len(d)
    if n < 30:
        return {"DM": float("nan"), "p_value_two_sided": float("nan"), "n": float(n)}
    mean_d = float(np.mean(d))
    gamma0 = float(np.mean((d - mean_d) ** 2))
    gamma1 = float(np.mean((d[1:] - mean_d) * (d[:-1] - mean_d))) if n > 1 else 0.0
    var_long = max(gamma0 + 2 * gamma1, 1e-12)
    dm = mean_d / math.sqrt(var_long / n)
    z = abs(dm)
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return {"DM": float(dm), "p_value_two_sided": float(p), "n": float(n)}


def bootstrap_mae_ci(y_true: np.ndarray, pred: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    ae = np.abs(y[m] - p[m])
    if len(ae) < 50:
        return {"mae": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(ae), size=len(ae))
        stats[b] = np.mean(ae[idx])
    return {
        "mae": float(np.mean(ae)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
    }

