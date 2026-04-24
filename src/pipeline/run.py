from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config.settings import PipelineConfig
from src.evaluation.metrics import regression_metrics
from src.evaluation.stats_tests import bootstrap_mae_ci, diebold_mariano
from src.evaluation.validation import chronological_split, walk_forward_slices
from src.features.build_features import build_feature_table
from src.models.arima_family import (
    frozen_ar_h_step_from_origins,
    frozen_ar_one_step,
    sarima_order_search,
    sarima_rolling_one_step,
)
from src.models.clustering import YEREVAN_DISTRICT_ORDER, build_district_mapping
from src.models.deepar_model import train_predict_deepar
from src.models.tree_models import build_baseline_models, make_tree_model, tune_hyperopt
from src.preprocessing.imputation import controlled_impute
from src.preprocessing.load import (
    aggregate_city_from_station_districts,
    align_hourly_grid,
    load_city_hourly,
    load_station_hourly,
    load_yerevan_station_metadata,
)
from src.preprocessing.missingness import audit_missingness

def _ensure_dirs(cfg: PipelineConfig) -> None:
    assert cfg.output is not None
    for p in (cfg.output.output_root, cfg.output.plots_dir, cfg.output.tables_dir, cfg.output.json_dir, cfg.output.images_dir):
        p.mkdir(parents=True, exist_ok=True)
    (cfg.project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)


def _save_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _model_palette(models: list[str]) -> dict[str, str]:
    base = {
        "Persistence": "#1F77B4",
        "Seasonal_naive_24h": "#FF7F0E",
        "AR(24)_frozen": "#2CA02C",
        "SARIMA_auto": "#D62728",
        "Ridge": "#9467BD",
        "RandomForest": "#8C564B",
        "XGBoost": "#E377C2",
        "LightGBM": "#17BECF",
        "DeepAR": "#7F7F7F",
    }
    return {m: base.get(m, "#4C4C4C") for m in models}


def _save_figure(fig: plt.Figure, out_path: Path, *, mirror_path: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mirror_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mae_by_horizon(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    agg = df.groupby("model", as_index=False)["MAE"].mean().sort_values("MAE")
    top_models = agg["model"].head(6).tolist()
    base_models = [m for m in ("Persistence", "Seasonal_naive_24h") if m in df["model"].unique()]
    show_models = list(dict.fromkeys(base_models + top_models))
    sub_df = df[df["model"].isin(show_models)].copy()
    palette = _model_palette(show_models)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for model in show_models:
        sub = sub_df[sub_df["model"] == model].sort_values("horizon_h")
        if sub.empty:
            continue
        lw = 2.6 if model in top_models[:3] else 1.8
        alpha = 1.0 if model in top_models[:3] else 0.9
        ax.plot(
            sub["horizon_h"],
            sub["MAE"],
            marker="o",
            linewidth=lw,
            markersize=5,
            label=model,
            color=palette.get(model),
            alpha=alpha,
        )
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("PM2.5 Forecast Error by Horizon (Top Models + Baselines)")
    ax.grid(alpha=0.25, linestyle="--")
    ax.set_xticks(sorted(sub_df["horizon_h"].unique()))
    ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_model_performance_1h(
    results_1h: pd.DataFrame,
    bootstrap: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    if results_1h.empty:
        return
    df = results_1h.copy().sort_values("MAE", ascending=True)
    models = df["Model"].tolist()
    mae = df["MAE"].to_numpy(dtype=float)
    ci_low = np.array([bootstrap.get(m, {}).get("ci_low", np.nan) for m in models], dtype=float)
    ci_high = np.array([bootstrap.get(m, {}).get("ci_high", np.nan) for m in models], dtype=float)
    xerr = np.vstack([mae - ci_low, ci_high - mae])

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    colors = [_model_palette(models)[m] for m in models]
    ypos = np.arange(len(models))
    ax.barh(ypos, mae, color=colors, alpha=0.85)
    finite_mask = np.isfinite(xerr).all(axis=0)
    ax.errorbar(
        mae[finite_mask],
        ypos[finite_mask],
        xerr=xerr[:, finite_mask],
        fmt="none",
        ecolor="#222222",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (1-hour horizon)")
    ax.set_title("1-Hour Model Performance with 95% Bootstrap CI")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_walk_forward_stability(wf_df: pd.DataFrame, out_path: Path) -> None:
    if wf_df.empty:
        return
    models = wf_df.groupby("model")["MAE"].mean().sort_values().index.tolist()
    palette = _model_palette(models)
    data = [wf_df.loc[wf_df["model"] == m, "MAE"].to_numpy(dtype=float) for m in models]
    fig, ax = plt.subplots(figsize=(9.6, 5.1))
    bp = ax.boxplot(data, patch_artist=True, labels=models, showfliers=True)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.35)
        patch.set_linewidth(1.1)
    for med in bp["medians"]:
        med.set_color("#111111")
        med.set_linewidth(1.4)
    ax.set_ylabel("MAE across walk-forward folds")
    ax.set_title("Walk-Forward Stability of Model Error")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_forecast_vs_actual(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
) -> None:
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    show_n = min(len(ts), 24 * 14)
    ts_short = ts.iloc[:show_n]
    y_t = y_true[:show_n]
    y_p = y_pred[:show_n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(ts_short, y_t, color="#1F77B4", linewidth=1.8, label="Actual")
    ax1.plot(ts_short, y_p, color="#D62728", linewidth=1.5, alpha=0.9, label=f"Predicted ({model_name})")
    ax1.set_ylabel("PM2.5 (µg/m³)")
    ax1.set_title("Forecast vs Actual PM2.5 (first 14 days of test window)")
    ax1.grid(alpha=0.22, linestyle="--")
    ax1.legend(loc="upper right", framealpha=0.95)

    resid = y_p - y_t
    ax2.axhline(0.0, color="#333333", linewidth=1.0)
    ax2.plot(ts_short, resid, color="#9467BD", linewidth=1.0)
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Timestamp")
    ax2.grid(alpha=0.22, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, out_path)


def _plot_station_groups_map(
    station_meta: pd.DataFrame | None,
    district_mapping: pd.DataFrame,
    out_path: Path,
) -> dict[str, Any]:
    summary = {
        "generated": False,
        "stations_total_unique": 0,
        "stations_with_location": 0,
        "stations_missing_location": 0,
        "group_station_counts": {},
    }
    if station_meta is None or station_meta.empty or district_mapping.empty:
        return summary
    lon_col = "longitude" if "longitude" in station_meta.columns else None
    lat_col = "latitude" if "latitude" in station_meta.columns else None
    sid_col = "station_id" if "station_id" in station_meta.columns else None
    if lon_col is None or lat_col is None or sid_col is None:
        return summary
    map_sid_col = "station_id" if "station_id" in district_mapping.columns else district_mapping.columns[0]
    district_name_col = "district_name" if "district_name" in district_mapping.columns else None

    station_unique = station_meta[[sid_col, lat_col, lon_col]].drop_duplicates(subset=[sid_col]).copy()
    station_unique["__has_location"] = station_unique[lat_col].notna() & station_unique[lon_col].notna()
    summary["stations_total_unique"] = int(len(station_unique))
    summary["stations_missing_location"] = int((~station_unique["__has_location"]).sum())
    station_unique = station_unique[station_unique["__has_location"]].drop(columns=["__has_location"])
    summary["stations_with_location"] = int(len(station_unique))

    map_cols = [map_sid_col, "district_id"] + ([district_name_col] if district_name_col else [])
    mapping_unique = district_mapping[map_cols].drop_duplicates(subset=[map_sid_col]).rename(columns={map_sid_col: sid_col})
    plot_df = station_unique.merge(
        mapping_unique,
        on=sid_col,
        how="inner",
    ).dropna(subset=[lat_col, lon_col, "district_id"])
    if plot_df.empty:
        return summary

    group_counts = plot_df.groupby("district_id")[sid_col].nunique().sort_index()
    full_counts = {str(i): 0 for i in range(len(YEREVAN_DISTRICT_ORDER))}
    for k, v in group_counts.items():
        full_counts[str(int(k))] = int(v)
    summary["group_station_counts"] = full_counts

    fig, ax = plt.subplots(figsize=(8.2, 7.4))
    cmap = plt.get_cmap("tab10")
    district_ids = sorted(plot_df["district_id"].unique().tolist())
    used_web_basemap = False
    # Try to render a real map baselayer; if unavailable, gracefully fall back to lon/lat plot.
    try:
        import contextily as ctx

        # WGS84 lon/lat -> Web Mercator meters (EPSG:3857)
        r = 6378137.0
        plot_df = plot_df.copy()
        plot_df["_x"] = np.deg2rad(plot_df[lon_col].astype(float)) * r
        plot_df["_y"] = np.log(np.tan(np.pi / 4.0 + np.deg2rad(plot_df[lat_col].astype(float)) / 2.0)) * r
        for i, d in enumerate(district_ids):
            sub = plot_df[plot_df["district_id"] == d]
            n_group = int(sub[sid_col].nunique())
            label_name = str(sub[district_name_col].iloc[0]) if district_name_col and not sub.empty else f"Group {d}"
            ax.scatter(
                sub["_x"],
                sub["_y"],
                s=62,
                alpha=0.92,
                color=cmap(i % 10),
                edgecolor="#111111",
                linewidth=0.5,
                label=f"{label_name} (n={n_group})",
                zorder=5,
            )
        x_pad = max((plot_df["_x"].max() - plot_df["_x"].min()) * 0.08, 400)
        y_pad = max((plot_df["_y"].max() - plot_df["_y"].min()) * 0.08, 400)
        ax.set_xlim(plot_df["_x"].min() - x_pad, plot_df["_x"].max() + x_pad)
        ax.set_ylim(plot_df["_y"].min() - y_pad, plot_df["_y"].max() + y_pad)
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom="auto")
        ax.set_xlabel("Web Mercator X")
        ax.set_ylabel("Web Mercator Y")
        used_web_basemap = True
    except Exception:
        for i, d in enumerate(district_ids):
            sub = plot_df[plot_df["district_id"] == d]
            n_group = int(sub[sid_col].nunique())
            label_name = str(sub[district_name_col].iloc[0]) if district_name_col and not sub.empty else f"Group {d}"
            ax.scatter(
                sub[lon_col],
                sub[lat_col],
                s=58,
                alpha=0.9,
                color=cmap(i % 10),
                edgecolor="#222222",
                linewidth=0.5,
                label=f"{label_name} (n={n_group})",
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.25, linestyle="--")

    ax.set_title("Yerevan Monitoring Stations by District Group")
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    ax.text(
        0.01,
        0.01,
        f"Unique stations plotted: {summary['stations_with_location']} | Missing location: {summary['stations_missing_location']}",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    _save_figure(fig, out_path)
    summary["generated"] = True
    summary["used_web_basemap"] = used_web_basemap
    return summary


def _forecast_1h_single_series(
    raw_one: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    min_feature_rows: int,
    series_id_value: str,
) -> pd.DataFrame | None:
    """Train the same 1h baseline + AR + ML stack on one aligned time series (district or station). No Hyperopt / SARIMA / DeepAR."""
    if raw_one.empty or "timestamp" not in raw_one.columns or cfg.data.target_col not in raw_one.columns:
        return None
    one = raw_one.copy()
    one["timestamp"] = pd.to_datetime(one["timestamp"])
    one = one.sort_values("timestamp").reset_index(drop=True)
    one = align_hourly_grid(one, freq=cfg.data.freq)
    one["city_slug"] = cfg.data.city_slug
    one["series_id"] = series_id_value

    audit_cols = [cfg.data.target_col] + [c for c in cfg.features.include_covariates if c in one.columns]
    audit_cols = [c for c in audit_cols if c in one.columns]
    one, _ = controlled_impute(one, audit_cols, cfg.data)
    one[cfg.data.target_col + "_raw"] = one[cfg.data.target_col]

    df_clean, feature_cols = build_feature_table(one, cfg.data.target_col, cfg.features)
    if len(df_clean) < min_feature_rows:
        return None

    n = len(df_clean)
    split = chronological_split(n, cfg.validation.train_ratio, cfg.validation.val_ratio)
    tr = df_clean.iloc[: split.train_end]
    va = df_clean.iloc[split.train_end : split.val_end]
    te = df_clean.iloc[split.val_end :]

    X_tr, y_tr = tr[feature_cols], tr["target_1h"].values.astype(float)
    X_va, y_va = va[feature_cols], va["target_1h"].values.astype(float)
    X_te, y_te = te[feature_cols], te["target_1h"].values.astype(float)
    y_full = df_clean[cfg.data.target_col].values.astype(float)

    preds_test: dict[str, np.ndarray] = {}

    preds_test["Persistence"] = te[cfg.data.target_col].to_numpy(dtype=float)
    idx_te = te.index.to_numpy()
    preds_test["Seasonal_naive_24h"] = np.where(idx_te + 1 - 24 >= 0, y_full[idx_te + 1 - 24], np.nan)

    ar_series = frozen_ar_one_step(y_full, split.train_end, lags=24)
    preds_test["AR(24)_frozen"] = ar_series[split.val_end : n]

    ridge = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
    ridge.fit(X_tr, y_tr)
    preds_test["Ridge"] = ridge.predict(X_te)

    models = build_baseline_models(cfg.models.random_state)
    tree_model_names = [n for n in ("RandomForest", "XGBoost", "LightGBM") if n in models]
    for name in tree_model_names:
        model = models[name]
        model.fit(X_tr, y_tr)
        preds_test[name] = model.predict(X_te)

    rows = []
    n_test = int(len(y_te))
    for name, pred in preds_test.items():
        m = regression_metrics(y_te, pred)
        rows.append({"Model": name, "n_test": n_test, **m})
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def _weighted_mean_mae_by_model(unit_df: pd.DataFrame) -> dict[str, float]:
    if unit_df.empty or "n_test" not in unit_df.columns:
        return {}
    out: dict[str, float] = {}
    for model, sub in unit_df.groupby("Model"):
        w = sub["n_test"].astype(float)
        if float(w.sum()) <= 0:
            continue
        out[str(model)] = float(np.average(sub["MAE"].astype(float), weights=w))
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def run_full_pipeline(cfg: PipelineConfig, *, enable_tuning: bool = True) -> dict[str, Any]:
    assert cfg.output is not None
    _ensure_dirs(cfg)
    station_meta = load_yerevan_station_metadata(cfg.data)
    district_result = build_district_mapping(station_meta)

    data_source = "city_avg_hourly"
    district_hourly: pd.DataFrame | None = None
    station_hourly_cached: pd.DataFrame | None = None
    city = load_city_hourly(cfg.data)
    if cfg.data.prefer_district_grouping:
        try:
            station_ids = set()
            if station_meta is not None and not station_meta.empty and "station_id" in station_meta.columns:
                station_ids = set(station_meta["station_id"].astype(str).unique().tolist())
            station_hourly = load_station_hourly(cfg.data, station_ids=station_ids if station_ids else None)
            station_hourly_cached = station_hourly
            district_hourly, city_grouped = aggregate_city_from_station_districts(
                station_hourly,
                district_result.mapping,
                target_col=cfg.data.target_col,
            )
            if len(city_grouped) >= 24 * 30:
                city = city_grouped
                data_source = "district_grouped_station_avg"
        except Exception:
            pass
    city = align_hourly_grid(city, freq=cfg.data.freq)
    city["city_slug"] = cfg.data.city_slug
    city["series_id"] = "city_aggregate"

    audit_cols = [cfg.data.target_col] + [c for c in cfg.features.include_covariates if c in city.columns]
    missing_audit = audit_missingness(city, audit_cols)
    city, imputation_meta = controlled_impute(city, audit_cols, cfg.data)
    city[cfg.data.target_col + "_raw"] = city[cfg.data.target_col]

    df_clean, feature_cols = build_feature_table(city, cfg.data.target_col, cfg.features)
    n = len(df_clean)
    split = chronological_split(n, cfg.validation.train_ratio, cfg.validation.val_ratio)
    tr = df_clean.iloc[: split.train_end]
    va = df_clean.iloc[split.train_end : split.val_end]
    te = df_clean.iloc[split.val_end :]

    X_tr, y_tr = tr[feature_cols], tr["target_1h"].values.astype(float)
    X_va, y_va = va[feature_cols], va["target_1h"].values.astype(float)
    X_te, y_te = te[feature_cols], te["target_1h"].values.astype(float)
    y_full = df_clean[cfg.data.target_col].values.astype(float)

    preds_val: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}

    # Baselines
    preds_val["Persistence"] = tr[cfg.data.target_col].iloc[-1:].repeat(len(va)).to_numpy(dtype=float)
    preds_test["Persistence"] = te[cfg.data.target_col].to_numpy(dtype=float)
    idx_va = va.index.to_numpy()
    idx_te = te.index.to_numpy()
    preds_val["Seasonal_naive_24h"] = np.where(idx_va + 1 - 24 >= 0, y_full[idx_va + 1 - 24], np.nan)
    preds_test["Seasonal_naive_24h"] = np.where(idx_te + 1 - 24 >= 0, y_full[idx_te + 1 - 24], np.nan)

    # Classical models
    ar_series = frozen_ar_one_step(y_full, split.train_end, lags=24)
    preds_val["AR(24)_frozen"] = ar_series[split.train_end : split.val_end]
    preds_test["AR(24)_frozen"] = ar_series[split.val_end : n]

    order: tuple[int, int, int] | None = None
    seasonal_order: tuple[int, int, int, int] | None = None
    sarima = np.full_like(y_full, np.nan, dtype=float)
    if cfg.models.include_sarima:
        order, seasonal_order = sarima_order_search(y_full[: split.train_end], seasonal_m=24)
        sarima = sarima_rolling_one_step(y_full, split.train_end, order, seasonal_order)
        preds_val["SARIMA_auto"] = sarima[split.train_end : split.val_end]
        preds_test["SARIMA_auto"] = sarima[split.val_end : n]

    # Ridge + tree models (MLP removed by design)
    ridge = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
    ridge.fit(X_tr, y_tr)
    preds_val["Ridge"] = ridge.predict(X_va)
    preds_test["Ridge"] = ridge.predict(X_te)

    models = build_baseline_models(cfg.models.random_state)
    tree_model_names = [n for n in ("RandomForest", "XGBoost", "LightGBM") if n in models]
    for name in tree_model_names:
        model = models[name]
        model.fit(X_tr, y_tr)
        preds_val[name] = model.predict(X_va)
        preds_test[name] = model.predict(X_te)

    tuned_info: dict[str, Any] = {}
    if enable_tuning and cfg.tuning.enabled:
        for name in tree_model_names:
            best_params, meta = tune_hyperopt(
                name,
                X_tr,
                y_tr,
                X_va,
                y_va,
                max_evals=cfg.tuning.max_evals,
                random_state=cfg.tuning.random_state,
            )
            tuned_info[name] = {"best_params": best_params, **meta}
        for name, info in tuned_info.items():
            tuned_model = make_tree_model(
                name,
                random_state=cfg.models.random_state,
                params=info["best_params"],
            )
            tuned_model.fit(X_tr, y_tr)
            preds_val[f"{name}_tuned"] = tuned_model.predict(X_va)
            preds_test[f"{name}_tuned"] = tuned_model.predict(X_te)

    # Main results
    rows = []
    for name, pred in preds_test.items():
        m = regression_metrics(y_te, pred)
        rows.append({"Model": name, **m})
    results_1h = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    results_1h.to_csv(cfg.output.tables_dir / "forecast_results_1h.csv", index=False)

    # Multi horizon (direct)
    hz_rows = []
    for h in cfg.features.horizons:
        target_h = f"target_{h}h"
        tr_h = df_clean.iloc[: split.train_end].dropna(subset=feature_cols + [target_h])
        te_h = df_clean.iloc[split.val_end :].dropna(subset=feature_cols + [target_h])
        if len(tr_h) < 100 or len(te_h) < 50:
            continue
        Xtr_h, ytr_h = tr_h[feature_cols], tr_h[target_h].values.astype(float)
        Xte_h, yte_h = te_h[feature_cols], te_h[target_h].values.astype(float)
        idx = te_h.index.to_numpy()
        pred_map = {
            "Persistence": te_h[cfg.data.target_col].values.astype(float),
            "Seasonal_naive_24h": np.where(idx + h - 24 >= 0, y_full[idx + h - 24], np.nan),
        }
        if h == 1:
            pred_map["AR(24)_frozen"] = ar_series[idx]
            if cfg.models.include_sarima:
                pred_map["SARIMA_auto"] = sarima[idx]
        else:
            pred_map["AR(24)_frozen"] = frozen_ar_h_step_from_origins(
                y_full,
                split.train_end,
                idx,
                h=h,
                lags=24,
            )
        r = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state).fit(Xtr_h, ytr_h)
        pred_map["Ridge"] = r.predict(Xte_h)
        for name in tree_model_names:
            model = build_baseline_models(cfg.models.random_state)[name]
            model.fit(Xtr_h, ytr_h)
            pred_map[name] = model.predict(Xte_h)
            if enable_tuning and cfg.tuning.enabled and name in tuned_info:
                tuned_model = make_tree_model(
                    name,
                    random_state=cfg.models.random_state,
                    params=tuned_info[name]["best_params"],
                )
                tuned_model.fit(Xtr_h, ytr_h)
                pred_map[f"{name}_tuned"] = tuned_model.predict(Xte_h)
        for name, pred in pred_map.items():
            met = regression_metrics(yte_h, pred)
            hz_rows.append({"horizon_h": h, "model": name, **met})
    hz_df = pd.DataFrame(hz_rows)
    hz_df.to_csv(cfg.output.tables_dir / "forecast_results_by_horizon.csv", index=False)

    # Walk-forward validation summary
    wf_rows = []
    if cfg.validation.walk_forward_enabled:
        min_train = split.train_end
        for fold_id, (tr_sl, te_sl) in enumerate(
            walk_forward_slices(
                n,
                min_train_size=min_train,
                test_size=cfg.validation.walk_forward_test_size,
                step=cfg.validation.walk_forward_step,
            ),
            start=1,
        ):
            tr_f = df_clean.iloc[tr_sl]
            te_f = df_clean.iloc[te_sl]
            Xtr_f, ytr_f = tr_f[feature_cols], tr_f["target_1h"].values.astype(float)
            Xte_f, yte_f = te_f[feature_cols], te_f["target_1h"].values.astype(float)
            idx_f = te_f.index.to_numpy()
            wf_preds: dict[str, np.ndarray] = {
                "Persistence": te_f[cfg.data.target_col].to_numpy(dtype=float),
                "Seasonal_naive_24h": np.where(idx_f + 1 - 24 >= 0, y_full[idx_f + 1 - 24], np.nan),
            }
            ar_fold = frozen_ar_one_step(y_full, tr_sl.stop, lags=24)
            wf_preds["AR(24)_frozen"] = ar_fold[te_sl]
            for name in ("Ridge", *tree_model_names):
                if name == "Ridge":
                    model = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
                else:
                    model = build_baseline_models(cfg.models.random_state)[name]
                model.fit(Xtr_f, ytr_f)
                wf_preds[name] = model.predict(Xte_f)
                if enable_tuning and cfg.tuning.enabled and name in tuned_info:
                    tuned_model = make_tree_model(
                        name,
                        random_state=cfg.models.random_state,
                        params=tuned_info[name]["best_params"],
                    )
                    tuned_model.fit(Xtr_f, ytr_f)
                    wf_preds[f"{name}_tuned"] = tuned_model.predict(Xte_f)
            for name, pred in wf_preds.items():
                met = regression_metrics(yte_f, pred)
                wf_rows.append({"fold": fold_id, "model": name, **met})
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(cfg.output.tables_dir / "walk_forward_results_1h.csv", index=False)

    deepar_info: dict[str, Any] = {"enabled": cfg.models.include_deepar, "status": "skipped"}
    if cfg.models.include_deepar:
        try:
            train_long = tr[["series_id", "timestamp", cfg.data.target_col]].rename(
                columns={cfg.data.target_col: "target"}
            )
            future_long = te[["series_id", "timestamp"]].copy()
            deepar_res = train_predict_deepar(
                train_long,
                future_long,
                horizon=1,
                backend=cfg.models.deepar_backend,
            )
            pred = (
                deepar_res.predictions[["timestamp", "prediction"]]
                .set_index("timestamp")
                .reindex(te["timestamp"])
                .prediction.to_numpy(dtype=float)
            )
            deepar_metrics = regression_metrics(y_te, pred)
            preds_test["DeepAR"] = pred
            rows.append({"Model": "DeepAR", **deepar_metrics})
            deepar_info = {
                "enabled": True,
                "status": "ok",
                "backend": deepar_res.backend,
                "metrics_1h": deepar_metrics,
            }
        except Exception as ex:
            deepar_info = {"enabled": True, "status": "error", "error": str(ex)}

    if "DeepAR" in preds_test:
        results_1h = pd.DataFrame(
            [{"Model": n, **regression_metrics(y_te, p)} for n, p in preds_test.items()]
        ).sort_values("MAE").reset_index(drop=True)
        results_1h.to_csv(cfg.output.tables_dir / "forecast_results_1h.csv", index=False)

    # Statistical comparisons
    best_model = str(results_1h.iloc[0]["Model"])
    ranked_models = [str(m) for m in results_1h["Model"].tolist()]
    second_best = ranked_models[1] if len(ranked_models) > 1 else None
    dm_rows: list[dict[str, Any]] = []
    dm_targets: list[str] = []
    if second_best is not None and second_best != best_model:
        dm_targets.append(second_best)
    for candidate in ("Ridge", "AR(24)_frozen"):
        if candidate in preds_test and candidate != best_model and candidate not in dm_targets:
            dm_targets.append(candidate)
    for target in dm_targets:
        dm_rows.append(
            {
                "comparison": f"{best_model}_vs_{target}_MAEloss",
                **diebold_mariano(y_te, preds_test[target], preds_test[best_model]),
            }
        )
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(cfg.output.tables_dir / "diebold_mariano_test.csv", index=False)

    bootstrap = {
        model: bootstrap_mae_ci(y_te, pred) for model, pred in preds_test.items()
    }

    _plot_mae_by_horizon(hz_df, cfg.output.plots_dir / "mae_by_horizon.png")
    _plot_model_performance_1h(results_1h, bootstrap, cfg.output.plots_dir / "model_performance_1h_ci.png")
    _plot_walk_forward_stability(wf_df, cfg.output.plots_dir / "walk_forward_mae_stability.png")
    _plot_forecast_vs_actual(
        te["timestamp"],
        y_te,
        preds_test[best_model],
        best_model,
        cfg.output.plots_dir / "forecast_vs_actual_best_1h.png",
    )
    station_group_plot_path = cfg.output.plots_dir / "station_groups_map.png"
    station_group_map_info = _plot_station_groups_map(
        station_meta,
        district_result.mapping,
        station_group_plot_path,
    )

    district_unit_rows: list[dict[str, Any]] = []
    if cfg.data.run_district_unit_forecasts and district_hourly is not None and not district_hourly.empty:
        for did in sorted(district_hourly["district_id"].dropna().unique()):
            did_int = int(did)
            sub = district_hourly.loc[district_hourly["district_id"] == did].drop(columns=["district_id"]).copy()
            if sub.empty:
                continue
            sid_label = f"district_{did_int}"
            dist_name = (
                YEREVAN_DISTRICT_ORDER[did_int] if 0 <= did_int < len(YEREVAN_DISTRICT_ORDER) else str(did_int)
            )
            unit_res = _forecast_1h_single_series(
                sub,
                cfg,
                min_feature_rows=cfg.data.district_unit_min_feature_rows,
                series_id_value=sid_label,
            )
            if unit_res is None or unit_res.empty:
                continue
            for _, r in unit_res.iterrows():
                district_unit_rows.append(
                    {
                        "district_id": did_int,
                        "district_name": dist_name,
                        "Model": str(r["Model"]),
                        "MAE": float(r["MAE"]),
                        "RMSE": float(r["RMSE"]),
                        "R2": float(r["R2"]),
                        "n_test": int(r["n_test"]),
                    }
                )
    district_unit_df = pd.DataFrame(district_unit_rows)

    station_unit_rows: list[dict[str, Any]] = []
    if cfg.data.run_station_unit_forecasts and station_hourly_cached is not None and not station_hourly_cached.empty:
        counts = station_hourly_cached.groupby("station_id").size().sort_values(ascending=False)
        n_done = 0
        for sid, _n in counts.items():
            if n_done >= cfg.data.max_station_unit_forecasts:
                break
            sub = station_hourly_cached.loc[station_hourly_cached["station_id"] == sid].drop(columns=["station_id"]).copy()
            unit_res = _forecast_1h_single_series(
                sub,
                cfg,
                min_feature_rows=cfg.data.station_unit_min_feature_rows,
                series_id_value=f"station_{sid}",
            )
            if unit_res is None or unit_res.empty:
                continue
            n_done += 1
            for _, r in unit_res.iterrows():
                station_unit_rows.append(
                    {
                        "station_id": str(sid),
                        "Model": str(r["Model"]),
                        "MAE": float(r["MAE"]),
                        "RMSE": float(r["RMSE"]),
                        "R2": float(r["R2"]),
                        "n_test": int(r["n_test"]),
                    }
                )
    station_unit_df = pd.DataFrame(station_unit_rows)

    district_unit_csv = cfg.output.tables_dir / "forecast_results_district_units_1h.csv"
    station_unit_csv = cfg.output.tables_dir / "forecast_results_station_units_1h.csv"
    if not district_unit_df.empty:
        district_unit_df.to_csv(district_unit_csv, index=False)
    if not station_unit_df.empty:
        station_unit_df.to_csv(station_unit_csv, index=False)

    split_info = {
        "train": {
            "timestamp_min": tr["timestamp"].min().isoformat(),
            "timestamp_max": tr["timestamp"].max().isoformat(),
            "n_rows": int(len(tr)),
        },
        "val": {
            "timestamp_min": va["timestamp"].min().isoformat(),
            "timestamp_max": va["timestamp"].max().isoformat(),
            "n_rows": int(len(va)),
        },
        "test": {
            "timestamp_min": te["timestamp"].min().isoformat(),
            "timestamp_max": te["timestamp"].max().isoformat(),
            "n_rows": int(len(te)),
        },
    }
    _save_summary(cfg.output.json_dir / "split_info.json", split_info)

    summary = {
        "best_mae_model_1h": best_model,
        "best_mae_value_1h": float(results_1h.iloc[0]["MAE"]),
        "n_models_1h": int(len(results_1h)),
        "split_info": split_info,
        "sarima_orders": {"enabled": cfg.models.include_sarima, "order": order, "seasonal_order": seasonal_order},
        "missingness_audit": missing_audit,
        "imputation_summary": imputation_meta,
        "walk_forward_available_folds": int(wf_df["fold"].nunique()) if not wf_df.empty else 0,
        "walk_forward_mean_mae_by_model": (
            wf_df.groupby("model")["MAE"].mean().sort_values().to_dict() if not wf_df.empty else {}
        ),
        "bootstrap_mae_95ci": bootstrap,
        "diebold_mariano": dm_df.to_dict(orient="records"),
        "tuned_models": tuned_info,
        "district_clustering": {
            "method": district_result.method,
            "n_districts": district_result.n_districts,
            "diagnostics": district_result.diagnostics,
            "n_yerevan_station_metadata": int(len(station_meta)) if station_meta is not None else 0,
            "station_group_map_generated": bool(station_group_map_info.get("generated")),
            "station_group_map_path": str(station_group_plot_path),
            "station_group_map_summary": station_group_map_info,
        },
        "deepar": deepar_info,
        "district_unit_forecasts": {
            "enabled": bool(cfg.data.run_district_unit_forecasts),
            "n_result_rows": int(len(district_unit_df)),
            "n_districts_evaluated": int(district_unit_df["district_id"].nunique()) if not district_unit_df.empty else 0,
            "csv_path": str(district_unit_csv) if not district_unit_df.empty else None,
            "weighted_mean_mae_by_model": _weighted_mean_mae_by_model(district_unit_df),
        },
        "station_unit_forecasts": {
            "enabled": bool(cfg.data.run_station_unit_forecasts),
            "n_result_rows": int(len(station_unit_df)),
            "n_stations_evaluated": int(station_unit_df["station_id"].nunique()) if not station_unit_df.empty else 0,
            "max_station_unit_forecasts": int(cfg.data.max_station_unit_forecasts),
            "csv_path": str(station_unit_csv) if not station_unit_df.empty else None,
            "weighted_mean_mae_by_model": _weighted_mean_mae_by_model(station_unit_df),
        },
        "notes": {
            "mlp_removed": True,
            "deepar_planned": cfg.models.include_deepar,
            "district_level_mode": data_source,
            "tree_models_enabled": tree_model_names,
            "lightgbm_available": "LightGBM" in tree_model_names,
            "outputs_layout": "All figures under results/plots/; all tabular CSVs under results/tables/; JSON under results/json/.",
            "city_forecast_primary_csv": "tables/forecast_results_1h.csv",
            "district_unit_forecast_csv": "tables/forecast_results_district_units_1h.csv",
            "station_unit_forecast_csv": "tables/forecast_results_station_units_1h.csv",
        },
    }
    _save_summary(cfg.output.json_dir / "results_summary.json", summary)
    _save_summary(cfg.output.json_dir / "data_manifest.json", {
        "date_min": city["timestamp"].min().isoformat(),
        "date_max": city["timestamp"].max().isoformat(),
        "n_rows_raw": int(len(city)),
        "n_rows_model_table": int(len(df_clean)),
        "columns": list(city.columns),
    })

    if cfg.output.save_processed_data:
        city.to_csv(cfg.project_root / "data" / "processed" / "city_hourly_aligned_imputed.csv", index=False)
        df_clean.to_csv(cfg.project_root / "data" / "processed" / "model_feature_table.csv", index=False)
        if district_hourly is not None and not district_hourly.empty:
            district_hourly.to_csv(cfg.project_root / "data" / "processed" / "district_hourly_grouped.csv", index=False)

    return {
        "results_1h": results_1h,
        "horizon": hz_df,
        "diebold_mariano": dm_df,
        "district_unit_1h": district_unit_df,
        "station_unit_1h": station_unit_df,
        "summary": summary,
    }

