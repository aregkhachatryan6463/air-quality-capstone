"""
Short-Term PM2.5 Forecasting in Yerevan – Full Pipeline

Capstone: comparative study of baseline, statistical, and machine learning models
for 1–4 hour ahead PM2.5 forecasts using city-level hourly data from airquality.am.

Pipeline: load → preprocess → feature engineering → time-based split →
         baselines & models → evaluation (MAE, RMSE, R²) and plots.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("ggplot")

# ---------------------------------------------------------------------------
# 1. Paths and load data
# ---------------------------------------------------------------------------
BASE_PATH = os.path.join(os.getcwd(), "Air Quality Data")
city_hourly_dir = os.path.join(BASE_PATH, "city_avg_hourly")

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

print("1. Data loaded")
print(f"   Rows: {len(df)}, span: {df['timestamp'].min()} to {df['timestamp'].max()}")

# ---------------------------------------------------------------------------
# 2. Temporal alignment and missing timestamps
# ---------------------------------------------------------------------------
ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
full_index = pd.date_range(ts_min, ts_max, freq="h")
df = df.set_index("timestamp").reindex(full_index).reset_index()
df = df.rename(columns={"index": "timestamp"})
df["city_slug"] = "yerevan"

print(f"\n2. Temporal alignment: {len(df)} hourly rows (including {df['avg_pm2.5'].isna().sum()} missing PM2.5)")

# ---------------------------------------------------------------------------
# 3. Preprocessing: imputation and outlier handling
# ---------------------------------------------------------------------------
# Columns to impute (forward then backward fill for short gaps)
numeric_cols = [
    "avg_pm2.5", "avg_pm10", "avg_temperature", "avg_pressure", "avg_humidity",
    "avg_no2", "total_rain", "avg_wind_speed", "avg_wind_direction"
]
numeric_cols = [c for c in numeric_cols if c in df.columns]
for c in numeric_cols:
    if df[c].isna().any():
        df[c] = df[c].ffill().bfill()

# Outlier handling: rolling IQR (methodology from proposal) then cap
pm25 = df["avg_pm2.5"].copy()
q1 = pm25.rolling(24, min_periods=12).quantile(0.25)
q3 = pm25.rolling(24, min_periods=12).quantile(0.75)
iqr = q3 - q1
outlier_flag = (pm25 > q3 + 1.5 * iqr) | (pm25 < q1 - 1.5 * iqr)
n_outliers = outlier_flag.sum()
print(f"   Rolling (24h) IQR: {int(n_outliers)} hours flagged as outliers")
cap_pct = 99.5
cap_val = pm25.quantile(cap_pct / 100)
df["avg_pm2.5_raw"] = pm25.copy()
df.loc[pm25 > cap_val, "avg_pm2.5"] = cap_val
print(f"   PM2.5 capped at {cap_val:.1f} µg/m³ (>{cap_pct}th percentile) for modeling")

# ---------------------------------------------------------------------------
# 4. Feature engineering: lags and time/calendar features
# ---------------------------------------------------------------------------
target_col = "avg_pm2.5"
lags = [1, 2, 3, 24]
for h in lags:
    df[f"pm25_lag_{h}h"] = df[target_col].shift(h)

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
# Heating season (Oct–April) as in Yerevan
df["heating_season"] = (df["month"].between(10, 12) | df["month"].between(1, 4)).astype(int)

# Optional covariates (use only if enough non-NaN after imputation)
use_covariates = ["avg_pm10", "avg_temperature", "avg_pressure", "avg_humidity"]
use_covariates = [c for c in use_covariates if c in df.columns and df[c].notna().mean() > 0.5]

feature_cols = [f"pm25_lag_{h}h" for h in lags] + ["hour", "day_of_week", "month", "heating_season"] + use_covariates

# Target: 1h ahead (and optional 2h, 3h, 4h for horizon comparison)
df["target_1h"] = df[target_col].shift(-1)
df["target_2h"] = df[target_col].shift(-2)
df["target_3h"] = df[target_col].shift(-3)
df["target_4h"] = df[target_col].shift(-4)

# Drop rows with NaN in features or target (from lags/shift)
df_clean = df.dropna(subset=feature_cols + ["target_1h"]).copy()
print(f"\n4. Feature engineering: {len(feature_cols)} features, {len(df_clean)} rows after dropna")

# Correlation of features with 1h-ahead target (for methodology/report)
corr_target = df_clean[feature_cols + ["target_1h"]].corr()["target_1h"].drop("target_1h", errors="ignore")
corr_target = corr_target.reindex(corr_target.abs().sort_values(ascending=False).index)
print("   Top feature correlations with target_1h:")
for f in corr_target.head(8).index:
    print(f"      {f}: {corr_target[f]:.3f}")

# ---------------------------------------------------------------------------
# 5. Time-based train / validation / test split
# ---------------------------------------------------------------------------
n = len(df_clean)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train = df_clean.iloc[:train_end]
val = df_clean.iloc[train_end:val_end]
test = df_clean.iloc[val_end:]

X_train = train[feature_cols]
y_train = train["target_1h"]
X_val = val[feature_cols]
y_val = val["target_1h"]
X_test = test[feature_cols]
y_test = test["target_1h"]

print(f"\n5. Split (time-based): train {len(train)}, val {len(val)}, test {len(test)}")

# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def print_metrics(name: str, m: dict):
    print(f"   {name}: MAE={m['MAE']:.3f}, RMSE={m['RMSE']:.3f}, R²={m['R2']:.4f}")

# ---------------------------------------------------------------------------
# 7. Baselines
# ---------------------------------------------------------------------------
# Persistence: 1h-ahead forecast = current PM2.5 (naive baseline)
y_pred_pers_val = val["avg_pm2.5"].values
y_pred_pers_test = test["avg_pm2.5"].values
metrics_pers_val = eval_metrics(y_val, y_pred_pers_val)
metrics_pers_test = eval_metrics(y_test, y_pred_pers_test)
print("\n6. Baselines")
print_metrics("Persistence (1h) val", metrics_pers_val)
print_metrics("Persistence (1h) test", metrics_pers_test)

# ---------------------------------------------------------------------------
# 8. Statistical and ML models
# ---------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr_val = lr.predict(X_val)
y_pred_lr_test = lr.predict(X_test)
print("\n7. Models (1h-ahead)")
print_metrics("Linear Regression val", eval_metrics(y_val, y_pred_lr_val))
print_metrics("Linear Regression test", eval_metrics(y_test, y_pred_lr_test))

ridge = Ridge(alpha=1.0, solver="lsqr", random_state=42)
ridge.fit(X_train, y_train)
y_pred_ridge_val = ridge.predict(X_val)
y_pred_ridge_test = ridge.predict(X_test)
print_metrics("Ridge val", eval_metrics(y_val, y_pred_ridge_val))
print_metrics("Ridge test", eval_metrics(y_test, y_pred_ridge_test))

rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf_val = rf.predict(X_val)
y_pred_rf_test = rf.predict(X_test)
print_metrics("Random Forest val", eval_metrics(y_val, y_pred_rf_val))
print_metrics("Random Forest test", eval_metrics(y_test, y_pred_rf_test))

try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb_val = xgb.predict(X_val)
    y_pred_xgb_test = xgb.predict(X_test)
    print_metrics("XGBoost val", eval_metrics(y_val, y_pred_xgb_val))
    print_metrics("XGBoost test", eval_metrics(y_test, y_pred_xgb_test))
    has_xgb = True
except ImportError:
    has_xgb = False
    y_pred_xgb_test = None

# ---------------------------------------------------------------------------
# 9. Results table
# ---------------------------------------------------------------------------
results = [
    ("Persistence", metrics_pers_test["MAE"], metrics_pers_test["RMSE"], metrics_pers_test["R2"]),
    ("Linear Regression", eval_metrics(y_test, y_pred_lr_test)["MAE"], eval_metrics(y_test, y_pred_lr_test)["RMSE"], eval_metrics(y_test, y_pred_lr_test)["R2"]),
    ("Ridge", eval_metrics(y_test, y_pred_ridge_test)["MAE"], eval_metrics(y_test, y_pred_ridge_test)["RMSE"], eval_metrics(y_test, y_pred_ridge_test)["R2"]),
    ("Random Forest", eval_metrics(y_test, y_pred_rf_test)["MAE"], eval_metrics(y_test, y_pred_rf_test)["RMSE"], eval_metrics(y_test, y_pred_rf_test)["R2"]),
]
if has_xgb:
    results.append(("XGBoost", eval_metrics(y_test, y_pred_xgb_test)["MAE"], eval_metrics(y_test, y_pred_xgb_test)["RMSE"], eval_metrics(y_test, y_pred_xgb_test)["R2"]))
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "R2"])
print("\n8. Test set performance summary")
print(results_df.to_string(index=False))
results_path = os.path.join(os.getcwd(), "forecast_results_1h.csv")
results_df.to_csv(results_path, index=False)
print(f"   Saved: {results_path}")

# ---------------------------------------------------------------------------
# 10. Forecast vs actual (test set)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_test.values, y_pred_pers_test, alpha=0.3, s=5, label="Persistence", color="gray")
ax.scatter(y_test.values, y_pred_lr_test, alpha=0.3, s=5, label="Linear Regression", color="C3")
ax.scatter(y_test.values, y_pred_ridge_test, alpha=0.3, s=5, label="Ridge", color="C0")
ax.scatter(y_test.values, y_pred_rf_test, alpha=0.3, s=5, label="Random Forest", color="C1")
if has_xgb:
    ax.scatter(y_test.values, y_pred_xgb_test, alpha=0.3, s=5, label="XGBoost", color="C2")
lims = [min(y_test.min(), y_test.min()), max(y_test.max(), y_test.max())]
ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
ax.set_xlabel("Actual PM2.5 (µg/m³)")
ax.set_ylabel("Predicted PM2.5 (µg/m³)")
ax.set_title("1-Hour-Ahead Forecast vs Actual (Test Set)")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "forecast_vs_actual_1h.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\n   Saved: forecast_vs_actual_1h.png")

# ---------------------------------------------------------------------------
# 11. Multi-horizon comparison (Persistence vs Ridge, 1–4h ahead)
# ---------------------------------------------------------------------------
horizons = [1, 2, 3, 4]
pers_mae, ridge_mae = [], []
for h in horizons:
    target_h = f"target_{h}h"
    train_h = df_clean.iloc[:train_end].dropna(subset=feature_cols + [target_h])
    test_sub = df_clean.iloc[val_end:].dropna(subset=feature_cols + [target_h])
    if len(train_h) < 100 or len(test_sub) == 0:
        continue
    X_tr = train_h[feature_cols]
    y_tr = train_h[target_h]
    X_t = test_sub[feature_cols]
    y_t = test_sub[target_h].values
    # Persistence: predict = current PM2.5
    y_pers = test_sub["avg_pm2.5"].values
    # Ridge trained for this horizon
    ridge_h = Ridge(alpha=1.0, solver="lsqr", random_state=42)
    ridge_h.fit(X_tr, y_tr)
    y_ridge = ridge_h.predict(X_t)
    pers_mae.append(eval_metrics(y_t, y_pers)["MAE"])
    ridge_mae.append(eval_metrics(y_t, y_ridge)["MAE"])
horizons = horizons[: len(pers_mae)]

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(horizons, pers_mae, "o-", label="Persistence", color="gray")
ax2.plot(horizons, ridge_mae, "s-", label="Ridge", color="C0")
ax2.set_xlabel("Forecast horizon (hours)")
ax2.set_ylabel("MAE (µg/m³)")
ax2.set_title("MAE by Forecast Horizon (Test Set)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "mae_by_horizon.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: mae_by_horizon.png")

# ---------------------------------------------------------------------------
# 12. Feature importance (Random Forest)
# ---------------------------------------------------------------------------
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
imp.plot(kind="barh", figsize=(8, 4), color="steelblue", alpha=0.8)
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance (1h-ahead PM2.5)")
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), "feature_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("   Saved: feature_importance.png")

print("\nDone.")
