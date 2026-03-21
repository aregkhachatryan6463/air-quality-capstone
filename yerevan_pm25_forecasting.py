"""
Yerevan hourly PM2.5: preprocessing, features, train/val/test evaluation,
baselines, AR, linear and tree models, MLP, optional XGBoost,
multi-horizon metrics, Diebold–Mariano output, and plots.

run_pipeline() supports sensitivity options (capping, lag sets).
"""

from __future__ import annotations

import os
import glob
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("ggplot")

# ---------------------------------------------------------------------------
# Metrics and statistics
# ---------------------------------------------------------------------------


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "absolute",
) -> Dict[str, float]:
    """
    Simple DM test for equal predictive accuracy (H0: same accuracy).
    Uses Newey–West–style lag-1 correction on mean loss differential (Harvey-style).
    """
    y_true = np.asarray(y_true, dtype=float)
    e_a = y_true - np.asarray(pred_a, dtype=float)
    e_b = y_true - np.asarray(pred_b, dtype=float)
    if loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        d = e_a**2 - e_b**2
    n = len(d)
    if n < 30:
        return {"DM": float("nan"), "p_value_two_sided": float("nan"), "n": float(n)}
    mean_d = np.mean(d)
    # HAC variance with one lag (very small sample correction)
    gamma0 = np.mean((d - mean_d) ** 2)
    if n > 1:
        dc = d - mean_d
        gamma1 = np.mean(dc[1:] * dc[:-1])
    else:
        gamma1 = 0.0
    var_long = gamma0 + 2 * gamma1
    var_long = max(var_long, 1e-12)
    dm = mean_d / np.sqrt(var_long / n)
    # Two-sided normal approx: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    import math

    z = abs(float(dm))
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    p = 2.0 * (1.0 - phi)
    return {"DM": float(dm), "p_value_two_sided": float(p), "n": float(n)}


def _print_metrics(name: str, m: Dict[str, float], verbose: bool) -> None:
    if verbose:
        print(f"   {name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R2={m['R2']:.4f}")


def _apply_paper_mpl_style() -> None:
    """Readable fonts and high DPI for camera-ready figures (ASCII unit labels)."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_figure_set(
    output_dir: str,
    *,
    paper_style: bool,
    y_test_arr: np.ndarray,
    y_pred_pers_test: np.ndarray,
    y_pred_ridge_test: np.ndarray,
    y_pred_rf_test: np.ndarray,
    y_pred_mlp_test: np.ndarray,
    y_pred_xgb_test: Optional[np.ndarray],
    has_xgb: bool,
    ar_test: np.ndarray,
    has_ar: bool,
    horizon_df: pd.DataFrame,
    skip_multi_horizon: bool,
    feature_cols: List[str],
    rf: RandomForestRegressor,
    best_tree_pred: np.ndarray,
    best_tree_name: str,
    test_timestamps: pd.Series,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if paper_style:
        _apply_paper_mpl_style()
    unit = "ug/m^3" if paper_style else "µg/m³"
    dpi = 300 if paper_style else 150

    fig, ax = plt.subplots(figsize=(7.2, 3.8) if paper_style else (10, 5))
    ax.scatter(y_test_arr, y_pred_pers_test, alpha=0.2, s=4, label="Persistence", color="gray")
    if has_ar:
        mt = np.isfinite(ar_test)
        ax.scatter(
            y_test_arr[mt],
            ar_test[mt],
            alpha=0.2,
            s=4,
            label="AR (frozen)",
            color="purple",
        )
    ax.scatter(y_test_arr, y_pred_ridge_test, alpha=0.2, s=4, label="Ridge", color="C0")
    ax.scatter(y_test_arr, y_pred_rf_test, alpha=0.2, s=4, label="Random Forest", color="C1")
    ax.scatter(y_test_arr, y_pred_mlp_test, alpha=0.2, s=4, label="MLP", color="C4")
    if has_xgb and y_pred_xgb_test is not None:
        ax.scatter(y_test_arr, y_pred_xgb_test, alpha=0.2, s=4, label="XGBoost", color="C2")
    lims = [float(y_test_arr.min()), float(y_test_arr.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Identity")
    ax.set_xlabel(f"Actual PM2.5 ({unit})")
    ax.set_ylabel(f"Predicted PM2.5 ({unit})")
    ax.set_title("1-h-ahead forecasts vs. actual (test set)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast_vs_actual_1h.png"), dpi=dpi)
    plt.close()

    if not skip_multi_horizon and not horizon_df.empty:
        fig2, ax2 = plt.subplots(figsize=(6.5, 4) if paper_style else (9, 5))
        for model in sorted(horizon_df["model"].unique()):
            sub = horizon_df[horizon_df["model"] == model]
            ax2.plot(sub["horizon_h"], sub["MAE"], "o-", label=model, alpha=0.85, markersize=4)
        ax2.set_xlabel("Horizon (hours)")
        ax2.set_ylabel(f"MAE ({unit})")
        ax2.set_title("Test MAE by forecast horizon")
        ax2.legend(fontsize=6, loc="best", framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mae_by_horizon.png"), dpi=dpi)
        plt.close()

    imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
    imp.plot(kind="barh", figsize=(6.5, 3.6) if paper_style else (8, 4), color="steelblue", alpha=0.85)
    plt.xlabel("Importance (RF)")
    plt.title("Random forest feature importance (1-h horizon)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=dpi)
    plt.close()

    resid = y_test_arr - best_tree_pred
    fig3, ax3 = plt.subplots(figsize=(7.2, 2.6) if paper_style else (10, 3))
    ax3.plot(test_timestamps, resid, lw=0.35, alpha=0.75)
    ax3.axhline(0, color="k", lw=0.5)
    ax3.set_title(f"Test residuals ({best_tree_name}): actual minus predicted")
    ax3.set_ylabel(unit)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_test_best_tree.png"), dpi=dpi)
    plt.close()

    if paper_style:
        plt.style.use("ggplot")


def _ar_frozen_onestep_series(y: np.ndarray, train_end: int, ar_lags: int) -> np.ndarray:
    """
    AR(ar_lags) fit on y[:train_end]; for each row index i, entry [i] is one-step pred of y[i+1]
    using realized lags y[i],...,y[i-ar_lags+1] and coefficients frozen from the training sample.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from statsmodels.tsa.ar_model import AutoReg

    n = len(y)
    out = np.full(n, np.nan)
    train_y = y[:train_end]
    if len(train_y) <= ar_lags + 2:
        return out
    res = AutoReg(train_y, lags=ar_lags, trend="c", old_names=False).fit()
    params = np.asarray(res.params, dtype=float)
    const = params[0]
    coefs = params[1:].astype(float)
    # w[j] = y[j],...,y[j+ar_lags-1]; for forecast of y[i+1] at row i use w[i-ar_lags+1][::-1]
    w = sliding_window_view(y, ar_lags)
    idx = np.arange(ar_lags - 1, n - 1, dtype=int)
    rows = w[idx - ar_lags + 1][:, ::-1]
    out[idx] = const + rows @ coefs
    return out


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------


def build_clean_frame(
    base_path: Optional[str] = None,
    cap_extreme: bool = True,
    cap_percentile: float = 99.5,
    lag_hours: Sequence[int] = (1, 2, 3, 24),
) -> Tuple[pd.DataFrame, List[str], np.ndarray, Dict[str, Any]]:
    """
    Load Yerevan city hourly data, align to hourly grid, impute, optional cap,
    build lag/calendar features and targets through 4h ahead.
    Returns df_clean (0..n-1 index), feature_cols, y_full, prep_meta.
    """
    if base_path is None:
        base_path = os.path.join(os.getcwd(), "Air Quality Data")
    city_hourly_dir = os.path.join(base_path, "city_avg_hourly")
    city_files = sorted(glob.glob(os.path.join(city_hourly_dir, "city_avg_hourly_*.csv")))
    if not city_files:
        raise FileNotFoundError(
            "No city_avg_hourly_*.csv files found. Run download_data.py first."
        )

    dfs = [pd.read_csv(f) for f in city_files]
    city_all = pd.concat(dfs, ignore_index=True)
    city_all["timestamp"] = pd.to_datetime(city_all["timestamp"])
    df = city_all[city_all["city_slug"] == "yerevan"].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    full_index = pd.date_range(ts_min, ts_max, freq="h")
    df = df.set_index("timestamp").reindex(full_index).reset_index()
    df = df.rename(columns={"index": "timestamp"})
    df["city_slug"] = "yerevan"

    target_col = "avg_pm2.5"
    numeric_cols = [
        target_col,
        "avg_pm10",
        "avg_temperature",
        "avg_pressure",
        "avg_humidity",
        "avg_no2",
        "total_rain",
        "avg_wind_speed",
        "avg_wind_direction",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].ffill().bfill()

    pm25 = df[target_col].copy()
    q1 = pm25.rolling(24, min_periods=12).quantile(0.25)
    q3 = pm25.rolling(24, min_periods=12).quantile(0.75)
    iqr = q3 - q1
    outlier_flag = (pm25 > q3 + 1.5 * iqr) | (pm25 < q1 - 1.5 * iqr)
    df["avg_pm2.5_raw"] = pm25.copy()
    if cap_extreme:
        cap_val = pm25.quantile(cap_percentile / 100.0)
        df.loc[pm25 > cap_val, target_col] = cap_val
    else:
        cap_val = float(pm25.max())

    lag_hours = list(lag_hours)
    for h in lag_hours:
        df[f"pm25_lag_{h}h"] = df[target_col].shift(h)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["heating_season"] = (df["month"].between(10, 12) | df["month"].between(1, 4)).astype(int)

    use_covariates = ["avg_pm10", "avg_temperature", "avg_pressure", "avg_humidity"]
    use_covariates = [
        c for c in use_covariates if c in df.columns and df[c].notna().mean() > 0.5
    ]

    feature_cols = (
        [f"pm25_lag_{h}h" for h in lag_hours]
        + ["hour", "day_of_week", "month", "heating_season"]
        + use_covariates
    )

    for h in (1, 2, 3, 4):
        df[f"target_{h}h"] = df[target_col].shift(-h)

    df_clean = df.dropna(subset=feature_cols + ["target_1h"]).copy()
    df_clean = df_clean.reset_index(drop=True)
    y_full = df_clean[target_col].values.astype(float)
    meta = {
        "n_outliers_flagged": int(outlier_flag.sum()),
        "cap_value_applied": float(cap_val) if cap_extreme else None,
        "cap_extreme": cap_extreme,
    }
    return df_clean, feature_cols, y_full, meta


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    base_path: Optional[str] = None,
    cap_extreme: bool = True,
    cap_percentile: float = 99.5,
    lag_hours: Sequence[int] = (1, 2, 3, 24),
    ar_lags: int = 24,
    verbose: bool = True,
    save_plots: bool = True,
    skip_multi_horizon: bool = False,
    paper_figures_dir: Optional[str] = None,
    save_prediction_bundle: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full training/evaluation. Returns dict with tables, predictions, DM stats.
    Set skip_multi_horizon=True for faster sweeps (e.g. sensitivity); 1h metrics unchanged.
    paper_figures_dir: if set, also writes publication-style PNGs (300 dpi) there.
    save_prediction_bundle: if set, path to .npz with y_test and per-model test preds (for bootstrap CIs).
    """
    if output_dir is None:
        output_dir = os.getcwd()

    df_clean, feature_cols, y_full, prep_meta = build_clean_frame(
        base_path=base_path,
        cap_extreme=cap_extreme,
        cap_percentile=cap_percentile,
        lag_hours=lag_hours,
    )
    target_col = "avg_pm2.5"
    n = len(df_clean)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df_clean.iloc[:train_end]
    val = df_clean.iloc[train_end:val_end]
    test = df_clean.iloc[val_end:]

    if verbose:
        print("1. Data loaded and prepared")
        print(f"   Rows: {n}, train/val/test: {len(train)}/{len(val)}/{len(test)}")
        print(
            f"   Preprocess: cap_extreme={cap_extreme}, lags={list(lag_hours)}, "
            f"IQR-flagged hours ~{prep_meta['n_outliers_flagged']}"
        )

    X_train, y_train = train[feature_cols], train["target_1h"]
    X_val, y_val = val[feature_cols], val["target_1h"]
    X_test, y_test = test[feature_cols], test["target_1h"]
    y_val_arr = y_val.values.astype(float)
    y_test_arr = y_test.values.astype(float)

    # --- Baselines: persistence, seasonal naive, rolling mean ---
    roll24_full = (
        pd.Series(y_full).rolling(24, min_periods=1).mean().values.astype(float)
    )

    def seasonal_naive_pred_for_rows(rows: pd.DataFrame, h: int) -> np.ndarray:
        """At row index i, forecast y_{i+h} = y_{i+h-24} (hourly seasonality)."""
        idx = rows.index.to_numpy()
        j = idx + h - 24
        return np.where(j >= 0, y_full[j], np.nan)

    def persistence_h(rows: pd.DataFrame, h: int) -> np.ndarray:
        """Naive h-step: y_hat_{t+h} = y_t (value at forecast origin row)."""
        _ = h
        return rows[target_col].values.astype(float)

    def rolling24_pred(rows: pd.DataFrame) -> np.ndarray:
        """1h only: mean of last 24 hours at origin row."""
        idx = rows.index.to_numpy()
        return roll24_full[idx]

    y_pred_pers_val = persistence_h(val, 1)
    y_pred_pers_test = persistence_h(test, 1)
    y_pred_sn_val = seasonal_naive_pred_for_rows(val, 1)
    y_pred_sn_test = seasonal_naive_pred_for_rows(test, 1)
    y_pred_roll_val = rolling24_pred(val)
    y_pred_roll_test = rolling24_pred(test)

    # --- Classical: frozen AR (one-step, coeffs from train only) ---
    try:
        ar_series = _ar_frozen_onestep_series(y_full, train_end, ar_lags)
        ar_val = ar_series[train_end:val_end]
        ar_test = ar_series[val_end:n]
        has_ar = bool(np.isfinite(ar_test).any())
    except Exception:
        ar_series = None
        ar_val = np.full(len(val), np.nan)
        ar_test = np.full(len(test), np.nan)
        has_ar = False

    # --- Sklearn models ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr_val = lr.predict(X_val)
    y_pred_lr_test = lr.predict(X_test)

    ridge = Ridge(alpha=1.0, solver="lsqr", random_state=42)
    ridge.fit(X_train, y_train)
    y_pred_ridge_val = ridge.predict(X_val)
    y_pred_ridge_test = ridge.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf_val = rf.predict(X_val)
    y_pred_rf_test = rf.predict(X_test)

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=400,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=42,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)
    y_pred_mlp_val = mlp.predict(X_val)
    y_pred_mlp_test = mlp.predict(X_test)

    has_xgb = False
    y_pred_xgb_val = y_pred_xgb_test = None
    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train)
        y_pred_xgb_val = xgb.predict(X_val)
        y_pred_xgb_test = xgb.predict(X_test)
        has_xgb = True
    except ImportError:
        pass

    if verbose:
        print("\n2. Models (1h-ahead), validation / test")
        print("\n   Baselines")
        _print_metrics("Persistence val", eval_metrics(y_val_arr, y_pred_pers_val), verbose)
        _print_metrics("Persistence test", eval_metrics(y_test_arr, y_pred_pers_test), verbose)
        _print_metrics("Seasonal naive (24h) val", eval_metrics(y_val_arr, y_pred_sn_val), verbose)
        _print_metrics("Seasonal naive (24h) test", eval_metrics(y_test_arr, y_pred_sn_test), verbose)
        _print_metrics("Rolling mean 24h val", eval_metrics(y_val_arr, y_pred_roll_val), verbose)
        _print_metrics("Rolling mean 24h test", eval_metrics(y_test_arr, y_pred_roll_test), verbose)
        if has_ar:
            mask = np.isfinite(ar_val)
            _print_metrics(
                f"AR({ar_lags}) frozen val",
                eval_metrics(y_val_arr[mask], ar_val[mask]),
                verbose,
            )
            mask_t = np.isfinite(ar_test)
            _print_metrics(
                f"AR({ar_lags}) frozen test",
                eval_metrics(y_test_arr[mask_t], ar_test[mask_t]),
                verbose,
            )
        print("\n   Regression / ML")
        _print_metrics("Linear Regression val", eval_metrics(y_val_arr, y_pred_lr_val), verbose)
        _print_metrics("Linear Regression test", eval_metrics(y_test_arr, y_pred_lr_test), verbose)
        _print_metrics("Ridge val", eval_metrics(y_val_arr, y_pred_ridge_val), verbose)
        _print_metrics("Ridge test", eval_metrics(y_test_arr, y_pred_ridge_test), verbose)
        _print_metrics("Random Forest val", eval_metrics(y_val_arr, y_pred_rf_val), verbose)
        _print_metrics("Random Forest test", eval_metrics(y_test_arr, y_pred_rf_test), verbose)
        _print_metrics("MLP (scaled) val", eval_metrics(y_val_arr, y_pred_mlp_val), verbose)
        _print_metrics("MLP (scaled) test", eval_metrics(y_test_arr, y_pred_mlp_test), verbose)
        if has_xgb:
            _print_metrics("XGBoost val", eval_metrics(y_val_arr, y_pred_xgb_val), verbose)
            _print_metrics("XGBoost test", eval_metrics(y_test_arr, y_pred_xgb_test), verbose)

    # --- Results table (test) ---
    def _mrow(name: str, y_t: np.ndarray, y_p: np.ndarray) -> Tuple[str, float, float, float]:
        m = eval_metrics(y_t, y_p)
        return (name, m["MAE"], m["RMSE"], m["R2"])

    results_rows: List[Tuple[str, float, float, float]] = [
        _mrow("Persistence", y_test_arr, y_pred_pers_test),
        _mrow("Seasonal_naive_24h", y_test_arr, y_pred_sn_test),
        _mrow("Rolling_mean_24h", y_test_arr, y_pred_roll_test),
    ]
    if has_ar:
        mt = np.isfinite(ar_test)
        results_rows.append(_mrow(f"AR({ar_lags})_frozen", y_test_arr[mt], ar_test[mt]))
    results_rows.extend(
        [
            _mrow("LinearRegression", y_test_arr, y_pred_lr_test),
            _mrow("Ridge", y_test_arr, y_pred_ridge_test),
            _mrow("RandomForest", y_test_arr, y_pred_rf_test),
            _mrow("MLP", y_test_arr, y_pred_mlp_test),
        ]
    )
    if has_xgb:
        results_rows.append(_mrow("XGBoost", y_test_arr, y_pred_xgb_test))

    results_df = pd.DataFrame(results_rows, columns=["Model", "MAE", "RMSE", "R2"])
    results_path = os.path.join(output_dir, "forecast_results_1h.csv")
    results_df.to_csv(results_path, index=False)
    if verbose:
        print(f"\n3. Saved: {results_path}")

    # --- Diebold–Mariano: best tree vs Ridge vs AR (if available) ---
    dm_rows = []
    best_tree_pred = y_pred_rf_test
    best_tree_name = "RandomForest"
    if has_xgb:
        m_rf = eval_metrics(y_test_arr, y_pred_rf_test)["MAE"]
        m_xgb = eval_metrics(y_test_arr, y_pred_xgb_test)["MAE"]
        if m_xgb < m_rf:
            best_tree_pred = y_pred_xgb_test
            best_tree_name = "XGBoost"
    dm_rows.append(
        {
            "comparison": f"{best_tree_name}_vs_Ridge_MAEloss",
            **diebold_mariano(y_test_arr, y_pred_ridge_test, best_tree_pred, loss="absolute"),
        }
    )
    if has_ar:
        mt = np.isfinite(ar_test)
        dm_rows.append(
            {
                "comparison": f"{best_tree_name}_vs_AR_MAEloss",
                **diebold_mariano(
                    y_test_arr[mt], ar_test[mt], best_tree_pred[mt], loss="absolute"
                ),
            }
        )
    dm_df = pd.DataFrame(dm_rows)
    dm_path = os.path.join(output_dir, "diebold_mariano_test.csv")
    dm_df.to_csv(dm_path, index=False)
    if verbose:
        print(f"   Saved: {dm_path}")

    # --- Multi-horizon (test): all main models ---
    horizons = [1, 2, 3, 4]
    horizon_records: List[Dict[str, Any]] = []

    def eval_horizon_block(h: int) -> None:
        target_h = f"target_{h}h"
        tr = df_clean.iloc[:train_end].dropna(subset=feature_cols + [target_h])
        te = df_clean.iloc[val_end:].dropna(subset=feature_cols + [target_h])
        if len(tr) < 100 or len(te) < 50:
            return
        X_tr, y_tr = tr[feature_cols], tr[target_h].values.astype(float)
        X_te, y_te = te[feature_cols], te[target_h].values.astype(float)
        pers = persistence_h(te, h)
        sn = seasonal_naive_pred_for_rows(te, h)
        roll = rolling24_pred(te) if h == 1 else np.full(len(te), np.nan)
        ar_h = np.full(len(te), np.nan)
        if h == 1 and ar_series is not None:
            idx_te = te.index.to_numpy()
            ar_h = ar_series[idx_te]

        def add_row(model: str, pred: np.ndarray) -> None:
            m = np.isfinite(pred)
            if m.sum() < 10:
                return
            met = eval_metrics(y_te[m], pred[m])
            horizon_records.append({"horizon_h": h, "model": model, **met})

        add_row("Persistence", pers)
        add_row("Seasonal_naive_24h", sn)
        if h == 1:
            add_row("Rolling_mean_24h", roll)
        if h == 1 and ar_series is not None:
            add_row(f"AR({ar_lags})_frozen", ar_h)

        r_h = Ridge(alpha=1.0, solver="lsqr", random_state=42)
        r_h.fit(X_tr, y_tr)
        add_row("Ridge", r_h.predict(X_te))

        rf_h = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        rf_h.fit(X_tr, y_tr)
        add_row("RandomForest", rf_h.predict(X_te))

        mlp_h = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        max_iter=400,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=15,
                        random_state=42,
                    ),
                ),
            ]
        )
        mlp_h.fit(X_tr, y_tr)
        add_row("MLP", mlp_h.predict(X_te))

        if has_xgb:
            from xgboost import XGBRegressor

            x_h = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
            x_h.fit(X_tr, y_tr)
            add_row("XGBoost", x_h.predict(X_te))

    if not skip_multi_horizon:
        for h in horizons:
            eval_horizon_block(h)

    horizon_df = pd.DataFrame(horizon_records)
    if not skip_multi_horizon:
        hz_path = os.path.join(output_dir, "forecast_results_by_horizon.csv")
        horizon_df.to_csv(hz_path, index=False)
        if verbose:
            print(f"   Saved: {hz_path}")

    test_preds = {
        "y_test": y_test_arr,
        "Persistence": y_pred_pers_test,
        "Seasonal_naive": y_pred_sn_test,
        "Rolling_mean": y_pred_roll_test,
        "AR": ar_test,
        "Ridge": y_pred_ridge_test,
        "RandomForest": y_pred_rf_test,
        "MLP": y_pred_mlp_test,
        "XGBoost": y_pred_xgb_test if has_xgb else None,
    }

    fig_kw = dict(
        y_test_arr=y_test_arr,
        y_pred_pers_test=y_pred_pers_test,
        y_pred_ridge_test=y_pred_ridge_test,
        y_pred_rf_test=y_pred_rf_test,
        y_pred_mlp_test=y_pred_mlp_test,
        y_pred_xgb_test=y_pred_xgb_test,
        has_xgb=has_xgb,
        ar_test=ar_test,
        has_ar=has_ar,
        horizon_df=horizon_df,
        skip_multi_horizon=skip_multi_horizon,
        feature_cols=feature_cols,
        rf=rf,
        best_tree_pred=best_tree_pred,
        best_tree_name=best_tree_name,
        test_timestamps=test["timestamp"],
    )
    if save_plots:
        _save_figure_set(output_dir, paper_style=False, **fig_kw)
        if paper_figures_dir:
            _save_figure_set(paper_figures_dir, paper_style=True, **fig_kw)
        if verbose:
            msg = "   Saved plots: forecast_vs_actual_1h, mae_by_horizon, feature_importance, residuals"
            if paper_figures_dir:
                msg += f"; paper copies in {paper_figures_dir}"
            print(msg)

    # --- Optional: save test predictions for paper bootstrap / archival ---
    if save_prediction_bundle:
        bundle_dir = os.path.dirname(os.path.abspath(save_prediction_bundle))
        if bundle_dir:
            os.makedirs(bundle_dir, exist_ok=True)
        to_save: Dict[str, np.ndarray] = {"y_test": y_test_arr}
        for k, v in test_preds.items():
            if k == "y_test" or v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if len(arr) == len(y_test_arr):
                to_save[k] = arr
        np.savez_compressed(save_prediction_bundle, **to_save)
        if verbose:
            print(f"   Saved prediction bundle: {save_prediction_bundle}")

    return {
        "results_1h": results_df,
        "horizon": horizon_df,
        "diebold_mariano": dm_df,
        "test_predictions": test_preds,
        "feature_cols": feature_cols,
        "prep_meta": prep_meta,
        "split": {"train_end": train_end, "val_end": val_end, "n": n},
        "timestamps_test": test["timestamp"].values,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Yerevan PM2.5 forecasting pipeline")
    p.add_argument(
        "--paper",
        action="store_true",
        help="Write paper/figures (300 dpi), paper/bundle/*.npz, and suggest running export_paper_assets.py",
    )
    args = p.parse_args()
    root = os.getcwd()
    paper_fig = os.path.join(root, "paper", "figures") if args.paper else None
    bundle = os.path.join(root, "paper", "bundle", "test_1h_predictions.npz") if args.paper else None
    if args.paper:
        os.makedirs(os.path.join(root, "paper", "figures"), exist_ok=True)
        os.makedirs(os.path.join(root, "paper", "bundle"), exist_ok=True)
    out = run_pipeline(
        verbose=True,
        save_plots=True,
        paper_figures_dir=paper_fig,
        save_prediction_bundle=bundle,
        output_dir=root,
    )
    print("\nDone.")
    if args.paper:
        print("Paper assets: paper/figures/, paper/bundle/test_1h_predictions.npz")
        print("Next: python export_paper_assets.py")
    return out


if __name__ == "__main__":
    main()
