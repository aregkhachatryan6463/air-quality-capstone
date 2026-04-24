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
from src.models.clustering import build_district_mapping
from src.models.deepar_model import train_predict_deepar
from src.models.tree_models import build_baseline_models, tune_hyperopt
from src.preprocessing.imputation import controlled_impute
from src.preprocessing.load import align_hourly_grid, load_city_hourly
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


def run_full_pipeline(cfg: PipelineConfig, *, enable_tuning: bool = True) -> dict[str, Any]:
    assert cfg.output is not None
    _ensure_dirs(cfg)

    city = load_city_hourly(cfg.data)
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

    # District-level mapping (adaptive fallback if metadata unavailable).
    station_meta = None
    if cfg.data.station_metadata_path and cfg.data.station_metadata_path.exists():
        station_meta = pd.read_csv(cfg.data.station_metadata_path)
    district_result = build_district_mapping(station_meta)

    # Main results
    rows = []
    for name, pred in preds_test.items():
        m = regression_metrics(y_te, pred)
        rows.append({"Model": name, **m})
    results_1h = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    results_1h.to_csv(cfg.output.output_root / "forecast_results_1h.csv", index=False)

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
        for name, pred in pred_map.items():
            met = regression_metrics(yte_h, pred)
            hz_rows.append({"horizon_h": h, "model": name, **met})
    hz_df = pd.DataFrame(hz_rows)
    hz_df.to_csv(cfg.output.output_root / "forecast_results_by_horizon.csv", index=False)

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
            for name in ("Ridge", *tree_model_names):
                if name == "Ridge":
                    model = Ridge(alpha=1.0, solver="lsqr", random_state=cfg.models.random_state)
                else:
                    model = build_baseline_models(cfg.models.random_state)[name]
                model.fit(Xtr_f, ytr_f)
                met = regression_metrics(yte_f, model.predict(Xte_f))
                wf_rows.append({"fold": fold_id, "model": name, **met})
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(cfg.output.output_root / "walk_forward_results_1h.csv", index=False)

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
        results_1h.to_csv(cfg.output.output_root / "forecast_results_1h.csv", index=False)

    # Statistical comparisons
    best_model = str(results_1h.iloc[0]["Model"])
    dm_df = pd.DataFrame(
        [
            {
                "comparison": f"{best_model}_vs_Ridge_MAEloss",
                **diebold_mariano(y_te, preds_test["Ridge"], preds_test[best_model]),
            },
            {
                "comparison": f"{best_model}_vs_AR_MAEloss",
                **diebold_mariano(y_te, preds_test["AR(24)_frozen"], preds_test[best_model]),
            },
        ]
    )
    dm_df.to_csv(cfg.output.output_root / "diebold_mariano_test.csv", index=False)

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
    _save_summary(cfg.output.output_root / "split_info.json", split_info)

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
        },
        "deepar": deepar_info,
        "notes": {
            "mlp_removed": True,
            "deepar_planned": cfg.models.include_deepar,
            "district_level_mode": "adaptive_metadata_required",
            "tree_models_enabled": tree_model_names,
            "lightgbm_available": "LightGBM" in tree_model_names,
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

    return {
        "results_1h": results_1h,
        "horizon": hz_df,
        "diebold_mariano": dm_df,
        "summary": summary,
    }

