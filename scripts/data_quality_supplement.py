"""
Supervisor Action 2: data quality transparency for the paper.

Produces saved figures (no interactive display):
- Missingness heatmap (calendar year × month) on the hourly grid for PM2.5
- Hours observed vs imputed (ffill/bfill) per year
- ADF tests: full hourly series (after imputation), monthly mean series, and per calendar month pooled hours
- PM2.5 time series with COVID window (Mar 2020–Jun 2021) highlighted

Run from project root after ``python download_data.py``:

    python scripts/data_quality_supplement.py
"""

from __future__ import annotations

import glob
import os
import warnings
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=FutureWarning)

OUT_DIR = os.path.join(os.getcwd(), "figures_data_quality")
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-06-30")


def _load_yerevan_grid(base_path: str | None = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return aligned hourly grid, raw PM2.5 on grid, and ffill/bfill-filled series."""
    if base_path is None:
        base_path = os.path.join(os.getcwd(), "Air Quality Data")
    city_hourly_dir = os.path.join(base_path, "city_avg_hourly")
    files = sorted(glob.glob(os.path.join(city_hourly_dir, "city_avg_hourly_*.csv")))
    if not files:
        raise FileNotFoundError("No city_avg_hourly_*.csv — run download_data.py first.")

    city_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    city_all["timestamp"] = pd.to_datetime(city_all["timestamp"])
    df = city_all[city_all["city_slug"] == "yerevan"].sort_values("timestamp").reset_index(drop=True)

    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    full_index = pd.date_range(ts_min, ts_max, freq="h")
    g = df.set_index("timestamp").reindex(full_index)
    g = g.rename_axis("timestamp").reset_index()
    g["city_slug"] = "yerevan"

    target = "avg_pm2.5"
    raw = g[target].copy()
    filled = raw.ffill().bfill()
    return g, raw, filled


def _adf_report(name: str, x: np.ndarray, maxlag: int | None = None) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"name": name, "n": len(x), "adf_stat": np.nan, "pvalue": np.nan, "note": "too_short"}
    if maxlag is None and len(x) > 10_000:
        maxlag = min(96, int(12 * (len(x) / 100.0) ** 0.25))
    res = adfuller(x, maxlag=maxlag, autolag="AIC" if maxlag is None else None)
    return {
        "name": name,
        "n": len(x),
        "adf_stat": float(res[0]),
        "pvalue": float(res[1]),
        "used_lag": int(res[2]),
        "nobs": int(res[3]),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    grid, raw_pm25, filled_pm25 = _load_yerevan_grid()

    grid["year"] = grid["timestamp"].dt.year
    grid["month"] = grid["timestamp"].dt.month
    grid["pm25_raw"] = raw_pm25.values
    grid["pm25_filled"] = filled_pm25.values
    grid["missing_pm25"] = raw_pm25.isna().astype(int)

    # --- Missingness heatmap (year × month): fraction of hours missing in each cell
    pivot_counts = grid.groupby(["year", "month"])["missing_pm25"].agg(["sum", "count"])
    pivot_counts["frac_missing"] = pivot_counts["sum"] / pivot_counts["count"]
    heat = pivot_counts["frac_missing"].unstack(level="month")
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        heat,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Fraction of hours missing"},
    )
    plt.title("PM2.5 missingness by calendar year and month (hourly grid)")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    p_heat = os.path.join(OUT_DIR, "pm25_missingness_year_month.png")
    plt.savefig(p_heat, dpi=200)
    plt.close()

    # --- Observed vs imputed hours per year (grid slots)
    was_na = raw_pm25.isna()
    still_na = filled_pm25.isna()
    imputed = was_na & ~still_na
    grid["imputed"] = imputed.astype(int)
    per_year = (
        grid.groupby("year")
        .agg(observed_hours=("missing_pm25", lambda s: int((s == 0).sum())), imputed_hours=("imputed", "sum"))
        .reset_index()
    )
    per_year["imputed_hours"] = per_year["imputed_hours"].astype(int)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(per_year))
    w = 0.35
    ax.bar(x - w / 2, per_year["observed_hours"], width=w, label="Observed (raw)", color="steelblue")
    ax.bar(x + w / 2, per_year["imputed_hours"], width=w, label="Filled by ffill/bfill", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(per_year["year"].astype(str))
    ax.set_ylabel("Hours")
    ax.set_title("PM2.5: observed vs imputed hours per year (hourly grid)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p_bar = os.path.join(OUT_DIR, "pm25_observed_vs_imputed_by_year.png")
    plt.savefig(p_bar, dpi=200)
    plt.close()

    # --- COVID / structural break context
    ts = grid["timestamp"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, filled_pm25, lw=0.4, color="tab:red", alpha=0.85, label="PM2.5 (ffill/bfill)")
    ax.axvspan(COVID_START, COVID_END, color="gray", alpha=0.25, label="COVID window (approx.)")
    ax.set_title("Yerevan hourly PM2.5 with COVID window highlighted")
    ax.set_xlabel("Time")
    ax.set_ylabel("µg/m³")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p_cov = os.path.join(OUT_DIR, "pm25_timeseries_covid_window.png")
    plt.savefig(p_cov, dpi=200)
    plt.close()

    # --- ADF: full hourly (imputed), monthly mean of hourly, and by calendar month (pooled hours)
    adf_rows: List[Dict[str, Any]] = []
    hourly_filled = filled_pm25.values.astype(float)
    adf_rows.append(_adf_report("hourly_PM25_ffill_bfill", hourly_filled))

    monthly_mean = (
        pd.Series(hourly_filled, index=grid["timestamp"]).resample("ME").mean().dropna()
    )
    adf_rows.append(_adf_report("monthly_mean_PM25", monthly_mean.values))

    for m in range(1, 13):
        mask = grid["month"] == m
        adf_rows.append(_adf_report(f"hourly_PM25_month_{m:02d}_pooled", hourly_filled[mask.values]))

    adf_df = pd.DataFrame(adf_rows)
    adf_path = os.path.join(OUT_DIR, "adf_stationarity_tests.csv")
    adf_df.to_csv(adf_path, index=False)

    print(f"Saved figures under: {OUT_DIR}")
    print(f"  {p_heat}")
    print(f"  {p_bar}")
    print(f"  {p_cov}")
    print(f"Saved ADF table: {adf_path}")
    print("\nADF summary (first rows):")
    print(adf_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
