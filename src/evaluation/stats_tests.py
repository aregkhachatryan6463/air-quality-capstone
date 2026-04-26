from __future__ import annotations

import math
from typing import Any

import numpy as np


def newey_west_bartlett_lag(n: int) -> int:
    """Bartlett lag length: floor(1.2 * n^(1/3)) (common in applied work)."""
    n = int(max(10, n))
    return int(max(1, math.floor(1.2 * n ** (1.0 / 3.0))))


def hac_lag_diefbold_mariano(
    n: int,
    *,
    horizon: int = 1,
    fixed_lag: int = 0,
) -> int:
    """
    If fixed_lag > 0, return it.
    Else: combine a Newey–West style rule of thumb with the forecast horizon.
    """
    if fixed_lag > 0:
        return int(fixed_lag)
    h = int(max(1, horizon))
    l_nw = newey_west_bartlett_lag(n)
    l_h = h
    l = int(max(1, min(n - 1, 50, max(l_nw, h + l_nw - 1))))
    return l


def _bartlett_weight(k: int, hac_lag: int) -> float:
    if hac_lag < 1:
        return 0.0
    if k == 0:
        return 1.0
    if k > hac_lag:
        return 0.0
    return 1.0 - (k / (hac_lag + 1.0))


def _loss_diff_variance_hac_bartlett(d: np.ndarray, hac_lag: int) -> float:
    """HAC long-run variance of the mean of d; DM uses V/n for mean_d."""
    d = np.asarray(d, dtype=float)
    d = d - d.mean()
    n = int(len(d))
    if n < 2:
        return 0.0
    L = int(min(hac_lag, n - 1, 100))
    g0 = float(np.mean(d**2))
    s = g0
    for k in range(1, L + 1):
        wk = _bartlett_weight(k, L)
        gk = float(np.mean(d[k:] * d[:-k]))
        s += 2.0 * wk * gk
    return s / n


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    *,
    hac_lag: int = 0,
    horizon: int = 1,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    pa = np.asarray(pred_a, dtype=float)
    pb = np.asarray(pred_b, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(pa) & np.isfinite(pb)
    y = y_true[mask]
    ea = y - pa[mask]
    eb = y - pb[mask]
    d = np.abs(ea) - np.abs(eb)
    n = int(len(d))
    if n < 30:
        return {"DM": float("nan"), "p_value_two_sided": float("nan"), "n": float(n), "hac_lag": float("nan")}
    mean_d = float(np.mean(d))
    lag_use = hac_lag_diefbold_mariano(n, horizon=int(horizon), fixed_lag=int(hac_lag))
    var_m = _loss_diff_variance_hac_bartlett(d, lag_use)
    if var_m <= 1e-15:
        var_m = 1e-12
    dm = mean_d / math.sqrt(var_m)
    z = abs(dm)
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return {
        "DM": float(dm),
        "p_value_two_sided": float(p),
        "n": float(n),
        "hac_lag": float(lag_use),
    }


def suggest_block_len_absolute_errors(
    ae: np.ndarray,
    min_len: int = 24,
    max_len: int = 168,
    horizon: int = 1,
) -> int:
    """Heuristic: max(2*H, 24) or first ACF e-folding of |e-mean|."""
    ae = np.asarray(ae, dtype=float)
    if len(ae) < 30:
        return int(min(max_len, max(min_len, 2 * int(horizon))))
    x = np.abs(ae) - float(np.mean(np.abs(ae)))
    x = x - x.mean()
    s2 = float(np.sum(x**2)) + 1e-12
    ac1 = float(np.sum(x[1:] * x[:-1])) / s2 if len(x) > 1 else 0.0
    base = int(max(2 * int(horizon), 24, min_len))
    if not np.isfinite(ac1) or ac1 < 0.1:
        return int(min(max_len, max(base, 48)))
    l = 2
    prev = 1.0
    for k in range(2, min(150, len(x) - 1)):
        num = float(np.sum(x[k:] * x[: -k]))
        rho = num / s2
        if abs(rho) < 0.05 and k > base // 2:
            l = k
            break
        prev = rho
    return int(max(min_len, min(max_len, l * 2, base + 1)))


def _circular_block_bootstrap_mean_mae(
    ae: np.ndarray,
    *,
    n_boot: int,
    block_len: int,
    seed: int,
) -> np.ndarray:
    n = int(len(ae))
    L = int(max(1, min(block_len, n)))
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / L))
    stats = np.empty(n_boot, dtype=float)
    aext = np.concatenate([ae, ae])  # circular
    for b in range(n_boot):
        sels = [rng.integers(0, n) for _ in range(k)]
        yb = np.concatenate([aext[s : s + L] for s in sels])[:n]
        stats[b] = float(np.mean(yb))
    return stats


def bootstrap_mae_ci(
    y_true: np.ndarray,
    pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    ae = np.abs(y[m] - p[m])
    if len(ae) < 50:
        return {
            "mae": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "method": "iid",
        }
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(ae), size=len(ae))
        stats[b] = np.mean(ae[idx])
    return {
        "mae": float(np.mean(ae)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
        "method": "iid",
    }


def bootstrap_mae_ci_block(
    y_true: np.ndarray,
    pred: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    block_len: int = 0,
    horizon: int = 1,
    block_len_min: int = 24,
    block_len_max: int = 168,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    ae = np.abs(y[m] - p[m])
    n = int(len(ae))
    if n < 50:
        return {
            "mae": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "block_len": float("nan"),
            "method": "block_circular",
        }
    L = int(block_len) if block_len > 0 else suggest_block_len_absolute_errors(
        ae, min_len=block_len_min, max_len=block_len_max, horizon=horizon
    )
    L = int(max(1, min(n, L)))
    stats = _circular_block_bootstrap_mean_mae(ae, n_boot=n_boot, block_len=L, seed=seed)
    return {
        "mae": float(np.mean(ae)),
        "ci_low": float(np.quantile(stats, 0.025)),
        "ci_high": float(np.quantile(stats, 0.975)),
        "block_len": float(L),
        "method": "block_circular",
    }
