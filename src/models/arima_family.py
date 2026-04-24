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


def sarima_order_search(y_train: np.ndarray, seasonal_m: int = 24) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    import pmdarima as pm

    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 200:
        return (1, 0, 1), (1, 0, 1, seasonal_m)
    if len(y) > 12_000:
        y = y[-12_000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(
            y,
            seasonal=True,
            m=seasonal_m,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            max_order=12,
            information_criterion="aic",
            approximation=True,
        )
    return tuple(model.order), tuple(model.seasonal_order)


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
    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res_t = model.fit(disp=False)
    future_obs = arr[train_end:]
    if future_obs.size == 0:
        fc = res_t.get_forecast(steps=1)
        pred[-1] = float(np.asarray(fc.predicted_mean).ravel()[0])
        return pred

    # Fast path: append all observed future points once, then pull one-step predictions.
    res_full = res_t.append(future_obs, refit=False)
    p = np.asarray(
        res_full.get_prediction(start=train_end, end=len(arr) - 1, dynamic=False).predicted_mean
    ).ravel()
    pred[train_end - 1 : len(arr) - 1] = p
    pred[-1] = float(np.asarray(res_full.get_forecast(steps=1).predicted_mean).ravel()[0])
    return pred

