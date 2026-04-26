from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def frozen_ar_one_step(y: np.ndarray, train_end: int, lags: int = 24) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    from statsmodels.tsa.ar_model import AutoReg

    y = np.asarray(y, dtype=float)
    out = np.full(len(y), np.nan)
    train = y[:train_end]
    if len(train) <= lags + 2:
        return out
    res = AutoReg(train, lags=lags, trend="c", old_names=False).fit()
    params = np.asarray(res.params, dtype=float)
    const, coefs = params[0], params[1:]
    windows = sliding_window_view(y, lags)
    idx = np.arange(lags - 1, len(y) - 1)
    rows = windows[idx - lags + 1][:, ::-1]
    out[idx] = const + rows @ coefs
    return out


def frozen_ar_h_step_from_origins(
    y: np.ndarray,
    train_end: int,
    origins: np.ndarray,
    *,
    h: int,
    lags: int = 24,
) -> np.ndarray:
    """
    h-step recursive forecast from specified origin indices using frozen AR coefficients.
    Each origin t predicts y[t + h] without using observations beyond t.
    """
    from statsmodels.tsa.ar_model import AutoReg

    y = np.asarray(y, dtype=float)
    origins = np.asarray(origins, dtype=int)
    out = np.full(len(origins), np.nan, dtype=float)
    train = y[:train_end]
    if len(train) <= lags + 2 or h < 1:
        return out
    res = AutoReg(train, lags=lags, trend="c", old_names=False).fit()
    params = np.asarray(res.params, dtype=float)
    const, coefs = float(params[0]), np.asarray(params[1:], dtype=float)

    for i, t in enumerate(origins):
        if t < lags - 1:
            continue
        state = list(np.asarray(y[t - lags + 1 : t + 1], dtype=float))
        if len(state) != lags or not np.isfinite(state).all():
            continue
        for _ in range(h):
            pred = const + float(sum(coefs[j] * state[-(j + 1)] for j in range(lags)))
            state.append(pred)
            state.pop(0)
        out[i] = state[-1]
    return out


def sarima_order_search(
    y_train: np.ndarray,
    seasonal_m: int = 24,
    *,
    information_criterion: str = "aic",
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    import pmdarima as pm

    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 200:
        return (1, 0, 1), (1, 0, 1, seasonal_m)
    # Truncate for order search (short window keeps Kalman memory low on small-RAM machines).
    if len(y) > 1_500:
        y = y[-1_500:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = pm.auto_arima(
                y,
                seasonal=True,
                m=seasonal_m,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
                max_p=2,
                max_q=2,
                max_d=1,
                max_P=1,
                max_D=1,
                max_Q=1,
                max_order=8,
                information_criterion=information_criterion,
                approximation=True,
            )
            return tuple(model.order), tuple(model.seasonal_order)
        except Exception:
            return (1, 0, 1), (1, 0, 1, seasonal_m)


def sarima_rolling_one_step(
    y: np.ndarray,
    train_end: int,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    arr = np.asarray(y, dtype=float)
    pred = np.full(len(arr), np.nan)
    train = arr[:train_end]
    if len(train) < 50:
        return pred
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
            low_memory=True,
        )
        res_t = model.fit(disp=False, maxiter=120, low_memory=True)
    except Exception:
        return pred
    future_obs = arr[train_end:]
    if future_obs.size == 0:
        try:
            fc = res_t.get_forecast(steps=1)
            pred[-1] = float(np.asarray(fc.predicted_mean).ravel()[0])
        except Exception:
            pass
        return pred

    # Fast path: append all observed future points once, then pull one-step predictions.
    try:
        res_full = res_t.append(future_obs, refit=False)
        p = np.asarray(
            res_full.get_prediction(start=train_end, end=len(arr) - 1, dynamic=False).predicted_mean
        ).ravel()
        pred[train_end - 1 : len(arr) - 1] = p
        pred[-1] = float(np.asarray(res_full.get_forecast(steps=1).predicted_mean).ravel()[0])
    except Exception:
        return pred
    return pred

