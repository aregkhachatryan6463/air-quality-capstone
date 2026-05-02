from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config.settings import PipelineConfig
from src.evaluation.metrics import regression_metrics
from src.evaluation.stats_tests import (
    bootstrap_mae_ci,
    bootstrap_mae_ci_block,
    diebold_mariano,
)
from src.evaluation.validation import chronological_split, walk_forward_slices
from src.features.build_features import build_feature_table
from src.models.deepar_model import train_predict_deepar, winkler_score
from src.pipeline.interpretation import save_acf_pacf_plot, save_permutation_importance_csv
from src.models.arima_family import (
    frozen_ar_h_step_from_origins,
    frozen_ar_one_step,
    sarima_order_search,
    sarima_orders_main_run,
    sarima_rolling_one_step,
)
from src.models.clustering import YEREVAN_DISTRICT_ORDER, build_district_mapping
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
    # ColorBrewer Dark2-inspired + distinct grays; tuned models reuse base model hue
    base = {
        "Persistence": "#1B9E77",
        "Seasonal_naive_24h": "#D95F02",
        "AR(24)_frozen": "#7570B3",
        "SARIMA_auto": "#E7298A",
        "Ridge": "#66A61E",
        "RandomForest": "#A6761D",
        "XGBoost": "#E6AB02",
        "LightGBM": "#1F78B4",
        "DeepAR": "#666666",
        "Ensemble_avg_top3": "#8C564B",
        "Ensemble_invMAE_top3": "#17BECF",
    }
    out: dict[str, str] = {}
    for m in models:
        if m in base:
            out[m] = base[m]
        elif m.endswith("_tuned") and m.replace("_tuned", "") in base:
            out[m] = base[m.replace("_tuned", "")]
        else:
            out[m] = "#4C4C4C"
    return out


def _base_model_color(name: str) -> str:
    p = _model_palette([name])
    return p.get(name, "#4C4C4C")


def _save_figure(fig: plt.Figure, out_path: Path, *, mirror_path: Path | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(mirror_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_mae_by_horizon(df: pd.DataFrame, out_path: Path, *, mirror_path: Path | None = None) -> None:
    if df.empty:
        return
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    agg = df.groupby("model", as_index=False)["MAE"].mean().sort_values("MAE")
    top_models = agg["model"].head(6).tolist()
    base_models = [m for m in ("Persistence", "Seasonal_naive_24h") if m in df["model"].unique()]
    show_models = list(dict.fromkeys(base_models + top_models))
    sub_df = df[df["model"].isin(show_models)].copy()
    cmap = plt.get_cmap("tab10")
    palette = {m: cmap(i % 10) for i, m in enumerate(show_models)}

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    for i, model in enumerate(show_models):
        sub = sub_df[sub_df["model"] == model].sort_values("horizon_h")
        if sub.empty:
            continue
        lw = 3.0 if model in top_models[:3] else 2.0
        ax.plot(
            sub["horizon_h"],
            sub["MAE"],
            marker="o",
            linewidth=lw,
            markersize=6.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=model,
            color=palette.get(model),
            zorder=10 - i,
        )
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (µg/m³; lower is better)")
    ax.set_title("Test MAE by horizon — direct multi-step, city-mean PM2.5 (Yerevan)")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.7, color="#888888")
    ax.set_xticks(sorted(sub_df["horizon_h"].unique()))
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.98,
        edgecolor="#b0b0b0",
    )
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(bottom=0.28)
    _save_figure(fig, out_path, mirror_path=mirror_path)


def _imputation_sensitivity_block(df_clean: pd.DataFrame, train_end: int, target_col: str) -> dict[str, Any]:
    """Train-segment imputation flag rates for methodology / paper reporting."""
    train = df_clean.iloc[:train_end]
    flag_cols = [c for c in train.columns if c.startswith("imputation_flag_")]
    out: dict[str, Any] = {
        "n_train_rows": int(len(train)),
        "imputation_flag_columns": flag_cols,
    }
    if not flag_cols or len(train) == 0:
        return out
    any_imp = np.zeros(len(train), dtype=bool)
    for c in flag_cols:
        s = train[c]
        if s.dtype == object:
            continue
        any_imp = any_imp | (s.fillna(0).astype(int) > 0)
    out["train_rows_any_imputation"] = int(any_imp.sum())
    out["train_fraction_any_imputation"] = float(any_imp.mean())
    tf = f"imputation_flag_{target_col}"
    if tf in train.columns and target_col in train.columns:
        s2 = train[tf].fillna(0).astype(float).to_numpy()
        y2 = train[target_col].astype(float).to_numpy()
        m2 = np.isfinite(y2) & np.isfinite(s2)
        if m2.sum() > 20 and np.std(s2[m2]) > 0 and np.std(y2[m2]) > 0:
            out["target_vs_impute_flag_correlation"] = float(np.corrcoef(y2[m2], s2[m2])[0, 1])
    return out


def _validation_chronological_vs_walkforward(results_1h: pd.DataFrame, wf_df: pd.DataFrame) -> pd.DataFrame:
    if wf_df.empty or results_1h.empty:
        return pd.DataFrame()
    main = results_1h.set_index("Model")["MAE"]
    wf_mean = wf_df.groupby("model")["MAE"].mean()
    common = main.index.intersection(wf_mean.index)
    rows: list[dict[str, Any]] = []
    for m in common:
        rows.append(
            {
                "model": m,
                "mae_chronological_test": float(main.loc[m]),
                "mae_walkforward_mean": float(wf_mean.loc[m]),
                "delta_mae_wf_minus_chrono": float(wf_mean.loc[m] - main.loc[m]),
            }
        )
    return pd.DataFrame(rows).sort_values("mae_chronological_test").reset_index(drop=True)


def _plot_model_performance_1h(
    results_1h: pd.DataFrame,
    bootstrap: dict[str, dict[str, float]],
    out_path: Path,
    *,
    mirror_path: Path | None = None,
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
    colors = [_base_model_color(m) for m in models]
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
    _save_figure(fig, out_path, mirror_path=mirror_path)


def _plot_walk_forward_stability(
    wf_df: pd.DataFrame, out_path: Path, *, mirror_path: Path | None = None
) -> None:
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
    _save_figure(fig, out_path, mirror_path=mirror_path)


def _plot_walk_forward_errorbars(
    wf_df: pd.DataFrame, out_path: Path, *, mirror_path: Path | None = None
) -> None:
    if wf_df.empty or "model" not in wf_df.columns:
        return
    g = wf_df.groupby("model")["MAE"].agg(["mean", "std", "count"]).reset_index()
    g = g.sort_values("mean")
    models = g["model"].tolist()
    means = g["mean"].to_numpy()
    st = g["std"].to_numpy()
    n = g["count"].to_numpy()
    se = st / np.sqrt(np.maximum(1, n))
    ypos = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = [_base_model_color(m) for m in models]
    ax.barh(ypos, means, xerr=se, color=colors, alpha=0.85, ecolor="#333333", capsize=3, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(models)
    ax.invert_yaxis()
    ax.set_xlabel("Mean MAE (± s.e. across walk-forward folds)")
    ax.set_title("Walk-forward: mean test error with fold-to-fold uncertainty")
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    _save_figure(fig, out_path, mirror_path=mirror_path)


def _plot_spatial_mae_levels(
    levels: list[dict[str, Any]],
    out_path: Path,
    *,
    mirror_path: Path | None = None,
) -> None:
    if not levels:
        return
    df = pd.DataFrame(levels)
    if df.empty or "MAE" not in df.columns:
        return
    models = (
        df.groupby("Model")["MAE"]
        .mean()
        .sort_values()
        .index.tolist()[:7]
    )
    sub = df[df["Model"].isin(models)]
    if sub.empty:
        return
    pvt = sub.pivot_table(index="Model", columns="level", values="MAE", aggfunc="mean")
    pvt = pvt.reindex(models)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    pvt.plot(kind="bar", ax=ax, rot=18, width=0.78, colormap="tab20")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_xlabel("Model")
    ax.set_title("MAE by spatial aggregation level (weighted where applicable)")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="Level", loc="best", framealpha=0.95)
    fig.tight_layout()
    _save_figure(fig, out_path, mirror_path=mirror_path)


def _plot_forecast_vs_actual(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    out_path: Path,
    *,
    mirror_path: Path | None = None,
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
    _save_figure(fig, out_path, mirror_path=mirror_path)


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

    df_clean, feature_cols = build_feature_table(
        one,
        cfg.data.target_col,
        cfg.features,
        minimal=cfg.features.use_minimal_feature_set_for_units,
    )
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


def _ensemble_candidates(pred_map: dict[str, np.ndarray]) -> list[str]:
    blocked = {"Persistence", "Seasonal_naive_24h"}
    return [k for k in pred_map.keys() if k not in blocked]


def _build_ensembles_from_val_test(
    y_val: np.ndarray,
    val_pred_map: dict[str, np.ndarray],
    test_pred_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    cand = [m for m in _ensemble_candidates(val_pred_map) if m in test_pred_map]
    if len(cand) < 2:
        return {}
    scored: list[tuple[str, float]] = []
    for m in cand:
        mae = regression_metrics(y_val, val_pred_map[m])["MAE"]
        if np.isfinite(mae):
            scored.append((m, float(mae)))
    if len(scored) < 2:
        return {}
    scored.sort(key=lambda x: x[1])
    top = [m for m, _ in scored[: min(3, len(scored))]]
    ens: dict[str, np.ndarray] = {}
    ens["Ensemble_avg_top3"] = np.nanmean(
        np.column_stack([test_pred_map[m] for m in top]), axis=1
    )
    maes = np.array([v for _, v in scored[: min(3, len(scored))]], dtype=float)
    inv = 1.0 / np.maximum(maes, 1e-6)
    w = inv / inv.sum()
    ens["Ensemble_invMAE_top3"] = np.sum(
        np.column_stack([test_pred_map[m] for m in top]) * w.reshape(1, -1), axis=1
    )
    return ens


def _arima_family_horizon_summary(
    hz_df: pd.DataFrame,
    *,
    ar_name: str = "AR(24)_frozen",
    sarima_name: str = "SARIMA_auto",
) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    if hz_df.empty:
        return out
    out["available"] = True
    out["horizons"] = sorted(hz_df["horizon_h"].dropna().unique().tolist())
    ar = hz_df.loc[hz_df["model"] == ar_name, ["horizon_h", "MAE"]].copy()
    sar = hz_df.loc[hz_df["model"] == sarima_name, ["horizon_h", "MAE"]].copy()
    if not ar.empty:
        ar = ar.sort_values("horizon_h")
        out["ar_mae_by_horizon"] = {
            str(int(r["horizon_h"])): float(r["MAE"]) for _, r in ar.iterrows()
        }
        if len(ar) >= 2:
            out["ar_mae_growth_last_minus_first"] = float(
                ar["MAE"].iloc[-1] - ar["MAE"].iloc[0]
            )
    if not sar.empty:
        sar = sar.sort_values("horizon_h")
        out["sarima_mae_by_horizon"] = {
            str(int(r["horizon_h"])): float(r["MAE"]) for _, r in sar.iterrows()
        }
        if len(sar) >= 2:
            out["sarima_mae_growth_last_minus_first"] = float(
                sar["MAE"].iloc[-1] - sar["MAE"].iloc[0]
            )
    if not ar.empty and not sar.empty:
        m = ar.merge(sar, on="horizon_h", suffixes=("_ar", "_sarima"))
        if not m.empty:
            out["ar_minus_sarima_mae_by_horizon"] = {
                str(int(r["horizon_h"])): float(r["MAE_ar"] - r["MAE_sarima"])
                for _, r in m.iterrows()
            }
            out["ar_beats_sarima_horizon_count"] = int((m["MAE_ar"] < m["MAE_sarima"]).sum())
            out["sarima_beats_ar_horizon_count"] = int((m["MAE_sarima"] < m["MAE_ar"]).sum())
    return out


def run_full_pipeline(cfg: PipelineConfig, *, enable_tuning: bool = True) -> dict[str, Any]:
    assert cfg.output is not None
    _ensure_dirs(cfg)
    print("[pipeline] Loading station metadata and city/district series...", flush=True)
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
    city_unimputed = city.copy()
    city, imputation_meta = controlled_impute(city, audit_cols, cfg.data)
    city[cfg.data.target_col + "_raw"] = city[cfg.data.target_col]
    imputation_sensitivity: dict[str, Any] = {}
    if cfg.data.run_imputation_sensitivity_ablation:
        city_alt, _ = controlled_impute(
            city_unimputed, audit_cols, cfg.data, skip_medium_hod=True
        )
        city_alt[cfg.data.target_col + "_raw"] = city_alt[cfg.data.target_col]
        try:
            d_alt, f_alt = build_feature_table(city_alt, cfg.data.target_col, cfg.features, minimal=False)
            if len(d_alt) > 200:
                sp2 = chronological_split(
                    len(d_alt), cfg.validation.train_ratio, cfg.validation.val_ratio
                )
                tr2, va2, te2 = (
                    d_alt.iloc[: sp2.train_end],
                    d_alt.iloc[sp2.train_end : sp2.val_end],
                    d_alt.iloc[sp2.val_end :],
                )
                r_ab = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
                r_ab.fit(tr2[f_alt], tr2["target_1h"].astype(float))
                p_ab = r_ab.predict(va2[f_alt])
                m_ab = regression_metrics(va2["target_1h"].to_numpy(), p_ab)
                imputation_sensitivity["ridge_mae_val_no_hod_for_medium_gaps"] = m_ab
        except Exception as ex:  # noqa: BLE001
            imputation_sensitivity["ablation_error"] = str(ex)

    df_clean, feature_cols = build_feature_table(city, cfg.data.target_col, cfg.features, minimal=False)
    n = len(df_clean)
    print(f"[pipeline] Model table rows={n}, {len(feature_cols)} features. Fitting baselines + SARIMA + ML...", flush=True)
    split = chronological_split(n, cfg.validation.train_ratio, cfg.validation.val_ratio)
    tr = df_clean.iloc[: split.train_end]
    va = df_clean.iloc[split.train_end : split.val_end]
    te = df_clean.iloc[split.val_end :]
    if cfg.data.run_imputation_sensitivity_ablation and imputation_sensitivity is not None:
        r0 = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
        r0.fit(tr[feature_cols], tr["target_1h"].values.astype(float))
        imputation_sensitivity["ridge_mae_val_with_hod_policy"] = regression_metrics(
            va["target_1h"].to_numpy(), r0.predict(va[feature_cols])
        )

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
        # Fixed structure from prior IC search on this dataset (avoids minutes-long pmdarima stepwise on CI/laptops).
        order, seasonal_order = sarima_orders_main_run()
        print(
            f"[pipeline] SARIMA orders (fixed replication): order={order} seasonal={seasonal_order}. Fitting SARIMAX...",
            flush=True,
        )
        sarima = sarima_rolling_one_step(y_full, split.train_end, order, seasonal_order)
        print("[pipeline] SARIMA test-horizon predictions complete.", flush=True)
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
        print(f"[pipeline] Hyperopt tuning ({cfg.tuning.max_evals} evals × {len(tree_model_names)} tree models)...", flush=True)
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

    ens_1h = _build_ensembles_from_val_test(y_va, preds_val, preds_test)
    preds_test.update(ens_1h)

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
        va_h = df_clean.iloc[split.train_end : split.val_end].dropna(subset=feature_cols + [target_h])
        te_h = df_clean.iloc[split.val_end :].dropna(subset=feature_cols + [target_h])
        if len(tr_h) < 100 or len(va_h) < 30 or len(te_h) < 50:
            continue
        Xtr_h, ytr_h = tr_h[feature_cols], tr_h[target_h].values.astype(float)
        Xva_h, yva_h = va_h[feature_cols], va_h[target_h].values.astype(float)
        Xte_h, yte_h = te_h[feature_cols], te_h[target_h].values.astype(float)
        idx = te_h.index.to_numpy()
        idx_va_h = va_h.index.to_numpy()
        pred_map = {
            "Persistence": te_h[cfg.data.target_col].values.astype(float),
            "Seasonal_naive_24h": np.where(idx + h - 24 >= 0, y_full[idx + h - 24], np.nan),
        }
        val_pred_map = {
            "Persistence": va_h[cfg.data.target_col].values.astype(float),
            "Seasonal_naive_24h": np.where(idx_va_h + h - 24 >= 0, y_full[idx_va_h + h - 24], np.nan),
        }
        if h == 1:
            pred_map["AR(24)_frozen"] = ar_series[idx]
            val_pred_map["AR(24)_frozen"] = ar_series[idx_va_h]
            if cfg.models.include_sarima:
                pred_map["SARIMA_auto"] = sarima[idx]
                val_pred_map["SARIMA_auto"] = sarima[idx_va_h]
        else:
            pred_map["AR(24)_frozen"] = frozen_ar_h_step_from_origins(
                y_full,
                split.train_end,
                idx,
                h=h,
                lags=24,
            )
            val_pred_map["AR(24)_frozen"] = frozen_ar_h_step_from_origins(
                y_full,
                split.train_end,
                idx_va_h,
                h=h,
                lags=24,
            )
        r = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state).fit(Xtr_h, ytr_h)
        pred_map["Ridge"] = r.predict(Xte_h)
        val_pred_map["Ridge"] = r.predict(Xva_h)
        for name in tree_model_names:
            model = build_baseline_models(cfg.models.random_state)[name]
            model.fit(Xtr_h, ytr_h)
            pred_map[name] = model.predict(Xte_h)
            val_pred_map[name] = model.predict(Xva_h)
            if enable_tuning and cfg.tuning.enabled and name in tuned_info:
                tuned_model = make_tree_model(
                    name,
                    random_state=cfg.models.random_state,
                    params=tuned_info[name]["best_params"],
                )
                tuned_model.fit(Xtr_h, ytr_h)
                pred_map[f"{name}_tuned"] = tuned_model.predict(Xte_h)
                val_pred_map[f"{name}_tuned"] = tuned_model.predict(Xva_h)
        pred_map.update(_build_ensembles_from_val_test(yva_h, val_pred_map, pred_map))
        for name, pred in pred_map.items():
            met = regression_metrics(yte_h, pred)
            hz_rows.append({"horizon_h": h, "model": name, **met})
    hz_df = pd.DataFrame(hz_rows)
    hz_df.to_csv(cfg.output.tables_dir / "forecast_results_by_horizon.csv", index=False)
    arima_hz = _arima_family_horizon_summary(hz_df)
    if arima_hz.get("available"):
        ar_rows: list[dict[str, Any]] = []
        for h in arima_hz.get("horizons", []):
            h_str = str(int(h))
            ar_rows.append(
                {
                    "horizon_h": int(h),
                    "AR(24)_frozen_MAE": arima_hz.get("ar_mae_by_horizon", {}).get(h_str, np.nan),
                    "SARIMA_auto_MAE": arima_hz.get("sarima_mae_by_horizon", {}).get(h_str, np.nan),
                    "AR_minus_SARIMA_MAE": arima_hz.get("ar_minus_sarima_mae_by_horizon", {}).get(h_str, np.nan),
                }
            )
        pd.DataFrame(ar_rows).to_csv(cfg.output.tables_dir / "arima_family_horizon_analysis.csv", index=False)
    print("[pipeline] Multi-horizon table done. Walk-forward validation (can be slow)...", flush=True)

    # Walk-forward: primary stability evaluation (includes SARIMA, optional per-fold DeepAR)
    wf_rows: list[dict[str, Any]] = []
    v = cfg.validation
    wf_sarima_orders: list[dict[str, Any]] = []
    if v.walk_forward_enabled:
        min_train = split.train_end
        for fold_id, (tr_sl, te_sl) in enumerate(
            walk_forward_slices(
                n,
                min_train_size=min_train,
                test_size=int(v.walk_forward_test_size),
                step=int(v.walk_forward_step),
                mode=v.walk_forward_mode,
                train_window=v.walk_forward_train_window,
                max_folds=v.max_walk_forward_folds,
            ),
            start=1,
        ):
            tr_f = df_clean.iloc[tr_sl]
            te_f = df_clean.iloc[te_sl]
            Xtr_f, ytr_f = tr_f[feature_cols], tr_f["target_1h"].values.astype(float)
            Xte_f, yte_f = te_f[feature_cols], te_f["target_1h"].values.astype(float)
            train_end = int(tr_sl.stop)
            idx_f = te_f.index.to_numpy()
            wf_preds: dict[str, np.ndarray] = {
                "Persistence": te_f[cfg.data.target_col].to_numpy(dtype=float),
                "Seasonal_naive_24h": np.where(idx_f + 1 - 24 >= 0, y_full[idx_f + 1 - 24], np.nan),
            }
            ar_fold = frozen_ar_one_step(y_full, train_end, lags=24)
            wf_preds["AR(24)_frozen"] = ar_fold[te_sl]
            if cfg.models.include_sarima and v.walk_forward_sarima:
                if v.walk_forward_reuse_main_sarima_orders and order is not None and seasonal_order is not None:
                    o_f, so_f = order, seasonal_order
                else:
                    o_f, so_f = sarima_order_search(
                        y_full[:train_end],
                        seasonal_m=24,
                        information_criterion=cfg.models.sarima_criterion,
                    )
                wf_sarima_orders.append(
                    {
                        "fold": fold_id,
                        "order": o_f,
                        "seasonal_order": so_f,
                        "reused_from_main": bool(
                            v.walk_forward_reuse_main_sarima_orders
                            and order is not None
                        ),
                    }
                )
                s_fold = sarima_rolling_one_step(y_full, train_end, o_f, so_f)
                wf_preds["SARIMA_auto"] = s_fold[te_sl]
            for name in ("Ridge", *tree_model_names):
                if name == "Ridge":
                    model = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
                    model.fit(Xtr_f, ytr_f)
                    wf_preds["Ridge"] = model.predict(Xte_f)
                    continue
                if (
                    v.walk_forward_refit_hyperopt
                    and enable_tuning
                    and cfg.tuning.enabled
                ):
                    m_tr = int(max(200, int(0.85 * len(Xtr_f))))
                    Xa, ya = Xtr_f.iloc[:m_tr], ytr_f[:m_tr]
                    Xb, yb = Xtr_f.iloc[m_tr:], ytr_f[m_tr:]
                    if len(ya) > 100 and len(yb) > 30:
                        bp, _ = tune_hyperopt(
                            name,
                            Xa,
                            ya,
                            Xb,
                            yb,
                            max_evals=max(5, min(cfg.tuning.max_evals, 20)),
                            random_state=cfg.tuning.random_state,
                        )
                        model = make_tree_model(
                            name,
                            random_state=cfg.models.random_state,
                            params=bp,
                        )
                    else:
                        model = build_baseline_models(cfg.models.random_state)[name]
                else:
                    model = build_baseline_models(cfg.models.random_state)[name]
                model.fit(Xtr_f, ytr_f)
                wf_preds[name] = model.predict(Xte_f)
                if (
                    (not v.walk_forward_refit_hyperopt)
                    and enable_tuning
                    and cfg.tuning.enabled
                    and name in tuned_info
                ):
                    tuned_model = make_tree_model(
                        name,
                        random_state=cfg.models.random_state,
                        params=tuned_info[name]["best_params"],
                    )
                    tuned_model.fit(Xtr_f, ytr_f)
                    wf_preds[f"{name}_tuned"] = tuned_model.predict(Xte_f)
            if (
                cfg.models.include_deepar
                and v.walk_forward_refit_deepar
                and v.walk_forward_max_folds_deepar > 0
                and fold_id <= int(v.walk_forward_max_folds_deepar)
            ):
                try:
                    train_long = tr_f[["series_id", "timestamp", cfg.data.target_col]].rename(
                        columns={cfg.data.target_col: "target"}
                    )
                    future_long = te_f[["series_id", "timestamp"]].copy()
                    d_res = train_predict_deepar(
                        train_long,
                        future_long,
                        horizon=1,
                        backend=cfg.models.deepar_backend,
                        max_steps=cfg.models.deepar_max_steps,
                        input_size=cfg.models.deepar_input_size,
                        random_seed=cfg.models.random_state,
                    )
                    pred_d = (
                        d_res.predictions[["timestamp", "prediction"]]
                        .set_index("timestamp")
                        .reindex(te_f["timestamp"])
                        .prediction.to_numpy(dtype=float)
                    )
                    wf_preds["DeepAR"] = pred_d
                except Exception:
                    pass
            for name, pred in wf_preds.items():
                met = regression_metrics(yte_f, pred)
                wf_rows.append({"fold": fold_id, "model": name, **met})
    wf_df = pd.DataFrame(wf_rows)
    if not wf_df.empty:
        wf_df.to_csv(cfg.output.tables_dir / "walk_forward_results_1h.csv", index=False)
        wf_stats = (
            wf_df.groupby("model")[["MAE", "RMSE", "R2"]]
            .agg(["mean", "std", "min", "max", "count"])
            .round(5)
        )
        wf_stats.to_csv(cfg.output.tables_dir / "walk_forward_aggregate_by_model.csv")
    else:
        wf_stats = pd.DataFrame()
    if not wf_df.empty:
        print(
            f"[pipeline] Walk-forward: {int(wf_df['fold'].nunique())} folds, {wf_df['model'].nunique()} model series.",
            flush=True,
        )

    print("[pipeline] Optional DeepAR + stats + figures + unit forecasts...", flush=True)
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
                max_steps=cfg.models.deepar_max_steps,
                input_size=cfg.models.deepar_input_size,
                random_seed=cfg.models.random_state,
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
                **{k: v for k, v in deepar_res.details.items() if k != "raw_columns"},
            }
            lo_c = deepar_res.details.get("lower_90_col")
            hi_c = deepar_res.details.get("upper_90_col")
            dfp = deepar_res.predictions
            if lo_c and hi_c and lo_c in dfp.columns and hi_c in dfp.columns:
                pr = dfp.set_index("timestamp").reindex(te["timestamp"])
                lo = pr[lo_c].to_numpy(dtype=float)
                hi = pr[hi_c].to_numpy(dtype=float)
                m = np.isfinite(y_te) & np.isfinite(lo) & np.isfinite(hi)
                if m.sum() > 0:
                    inside = (y_te[m] >= lo[m]) & (y_te[m] <= hi[m])
                    deepar_info["interval_90_coverage"] = float(np.mean(inside))
                    deepar_info["n_points_interval_eval"] = int(m.sum())
                    deepar_info["winkler_90"] = winkler_score(
                        y_te, lo, hi, alpha=0.1
                    )
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
    h_dm = max(1, int(getattr(cfg.stats, "horizon_hours_for_dm", 1)))
    hac_fix = int(cfg.stats.hac_lag) if cfg.stats.hac_lag else 0
    for target in dm_targets:
        dm_rows.append(
            {
                "comparison": f"{best_model}_vs_{target}_MAEloss",
                **diebold_mariano(
                    y_te,
                    preds_test[target],
                    preds_test[best_model],
                    hac_lag=hac_fix,
                    horizon=h_dm,
                ),
            }
        )
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(cfg.output.tables_dir / "diebold_mariano_test.csv", index=False)

    bootstrap: dict[str, Any] = {}
    for model, pred in preds_test.items():
        if not cfg.stats.block_bootstrap:
            b = {**bootstrap_mae_ci(y_te, pred, n_boot=cfg.stats.bootstrap_n, seed=cfg.tuning.random_state), "method": "iid"}
        else:
            b = bootstrap_mae_ci_block(
                y_te,
                pred,
                n_boot=cfg.stats.bootstrap_n,
                seed=cfg.tuning.random_state,
                block_len=0,
                horizon=h_dm,
                block_len_min=cfg.stats.block_len_min,
                block_len_max=cfg.stats.block_len_max,
            )
            if not np.isfinite(b.get("ci_low", float("nan"))):
                b = {**bootstrap_mae_ci(y_te, pred, n_boot=cfg.stats.bootstrap_n, seed=cfg.tuning.random_state), "method": "iid"}
            else:
                b = {**b, "method": "block_circular"}
        if cfg.stats.iid_bootstrap_also and cfg.stats.block_bootstrap and b.get("method") == "block_circular":
            b_i = bootstrap_mae_ci(
                y_te, pred, n_boot=cfg.stats.bootstrap_n, seed=cfg.tuning.random_state + 1
            )
            b["iid_bootstrap_mae"] = b_i.get("mae", float("nan"))
            b["iid_bootstrap_ci_low"] = b_i.get("ci_low", float("nan"))
            b["iid_bootstrap_ci_high"] = b_i.get("ci_high", float("nan"))
        bootstrap[model] = b

    img = cfg.output.images_dir
    _plot_mae_by_horizon(
        hz_df, cfg.output.plots_dir / "mae_by_horizon.png", mirror_path=img / "mae_by_horizon.png"
    )
    _plot_model_performance_1h(
        results_1h,
        bootstrap,
        cfg.output.plots_dir / "model_performance_1h_ci.png",
        mirror_path=img / "model_performance_1h_ci.png",
    )
    _plot_walk_forward_stability(
        wf_df, cfg.output.plots_dir / "walk_forward_mae_stability.png", mirror_path=img / "walk_forward_mae_stability.png"
    )
    _plot_walk_forward_errorbars(
        wf_df,
        cfg.output.plots_dir / "walk_forward_mean_with_errorbars.png",
        mirror_path=img / "walk_forward_mean_with_errorbars.png",
    )
    y_train_ser = df_clean.iloc[: split.train_end][cfg.data.target_col].to_numpy(dtype=float)
    save_acf_pacf_plot(
        y_train_ser,
        cfg.output.plots_dir / "acf_pacf_train.png",
        mirror_path=img / "acf_pacf_train.png",
    )
    _plot_forecast_vs_actual(
        te["timestamp"],
        y_te,
        preds_test[best_model],
        best_model,
        cfg.output.plots_dir / "forecast_vs_actual_best_1h.png",
        mirror_path=img / "forecast_vs_actual_best_1h.png",
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

    spatial_mae_rows: list[dict[str, Any]] = []
    for _, r in results_1h.iterrows():
        spatial_mae_rows.append(
            {
                "Model": str(r["Model"]),
                "level": "city_chronological_test",
                "MAE": float(r["MAE"]),
            }
        )
    wd = _weighted_mean_mae_by_model(district_unit_df)
    ws = _weighted_mean_mae_by_model(station_unit_df)
    for m, v in wd.items():
        spatial_mae_rows.append({"Model": m, "level": "district_units_weighted", "MAE": v})
    for m, v in ws.items():
        spatial_mae_rows.append({"Model": m, "level": "station_sample_weighted", "MAE": v})
    spatial_mae_df = pd.DataFrame(spatial_mae_rows)
    if not spatial_mae_df.empty:
        spatial_mae_df.to_csv(cfg.output.tables_dir / "spatial_mae_by_level.csv", index=False)
        _plot_spatial_mae_levels(
            spatial_mae_rows,
            cfg.output.plots_dir / "spatial_level_comparison_mae.png",
            mirror_path=img / "spatial_level_comparison_mae.png",
        )

    val_comp_df = _validation_chronological_vs_walkforward(results_1h, wf_df)
    val_comp_path = cfg.output.tables_dir / "validation_chronological_vs_walkforward_1h.csv"
    if not val_comp_df.empty:
        val_comp_df.to_csv(val_comp_path, index=False)
    imputation_sens = _imputation_sensitivity_block(df_clean, split.train_end, cfg.data.target_col)
    imputation_sens = {**imputation_sens, **imputation_sensitivity}
    if enable_tuning and cfg.tuning.enabled and "XGBoost" in tuned_info:
        try:
            xm = make_tree_model(
                "XGBoost",
                random_state=cfg.models.random_state,
                params=tuned_info["XGBoost"]["best_params"],
            )
            xm.fit(X_tr, y_tr)
            save_permutation_importance_csv(
                X_te,
                y_te,
                xm,
                cfg.output.tables_dir / "permutation_importance_xgboost_tuned.csv",
            )
        except Exception:
            pass

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
        "imputation_sensitivity": imputation_sens,
        "validation_chronological_vs_walkforward": (
            {
                "csv": "tables/validation_chronological_vs_walkforward_1h.csv",
                "n_models_compared": int(len(val_comp_df)),
                "rows": val_comp_df.to_dict(orient="records") if not val_comp_df.empty else [],
            }
        ),
        "walk_forward_available_folds": int(wf_df["fold"].nunique()) if not wf_df.empty else 0,
        "walk_forward_sarima_orders_by_fold": wf_sarima_orders,
        "walk_forward_mean_mae_by_model": (
            wf_df.groupby("model")["MAE"].mean().sort_values().to_dict() if not wf_df.empty else {}
        ),
        "walk_forward_mae_std_by_model": (
            wf_df.groupby("model")["MAE"].std().to_dict() if not wf_df.empty else {}
        ),
        "config_snapshot": {
            "sarima_criterion": cfg.models.sarima_criterion,
            "block_bootstrap": cfg.stats.block_bootstrap,
            "walk_forward_mode": cfg.validation.walk_forward_mode,
            "max_walk_forward_folds": cfg.validation.max_walk_forward_folds,
        },
        "bootstrap_mae_95ci": bootstrap,
        "arima_family_horizon_analysis": arima_hz,
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
            "outputs_layout": "All figures under results/plots/ (mirrored to top-level images/ for papers); all tabular CSVs under results/tables/; JSON under results/json/.",
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

