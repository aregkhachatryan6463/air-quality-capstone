"""
Build LaTeX tables and a JSON summary from pipeline CSV outputs.

Optional: if paper/bundle/test_1h_predictions.npz exists (from
yerevan_pm25_forecasting.py --paper), also writes a bootstrap MAE table.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _escape_tex(s: str) -> str:
    return (
        str(s)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def df_to_latex_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    floatfmt: str = "%.3f",
) -> str:
    """Minimal booktabs-style table (requires booktabs package)."""
    cols = list(df.columns)
    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{_escape_tex(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    col_spec = "l" + "r" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_escape_tex(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt % float(v))
            else:
                cells.append(_escape_tex(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def bootstrap_mae_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    ae = np.abs(y - yhat)
    n = len(ae)
    if n < 50:
        return {"mae": float(np.nan), "ci_low": float(np.nan), "ci_high": float(np.nan)}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = float(np.mean(ae[idx]))
    q = (1 - confidence) / 2
    return {
        "mae": float(np.mean(ae)),
        "ci_low": float(np.quantile(stats, q)),
        "ci_high": float(np.quantile(stats, 1 - q)),
    }


def main() -> None:
    root = os.getcwd()
    results_root = os.path.join(root, "results")
    legacy_root = root
    paths = {
        "main": os.path.join(results_root, "forecast_results_1h.csv"),
        "horizon": os.path.join(results_root, "forecast_results_by_horizon.csv"),
        "dm": os.path.join(results_root, "diebold_mariano_test.csv"),
        "bundle": os.path.join(root, "paper", "bundle", "test_1h_predictions.npz"),
    }
    legacy_paths = {
        "main": os.path.join(legacy_root, "forecast_results_1h.csv"),
        "horizon": os.path.join(legacy_root, "forecast_results_by_horizon.csv"),
        "dm": os.path.join(legacy_root, "diebold_mariano_test.csv"),
    }
    for key in ("main", "horizon", "dm"):
        if not os.path.isfile(paths[key]) and os.path.isfile(legacy_paths[key]):
            paths[key] = legacy_paths[key]
    for key in ("main", "horizon", "dm"):
        if not os.path.isfile(paths[key]):
            print(f"Missing {paths[key]} — run run_pipeline.py first.", file=sys.stderr)
            sys.exit(1)

    paper_root = os.path.join(root, "paper")
    tab_dir = os.path.join(paper_root, "tables")
    os.makedirs(tab_dir, exist_ok=True)

    df_main = pd.read_csv(paths["main"])
    df_hz = pd.read_csv(paths["horizon"])
    df_dm = pd.read_csv(paths["dm"])

    # Main results: round for display
    disp = df_main.copy()
    for c in ("MAE", "RMSE", "R2"):
        if c in disp.columns:
            disp[c] = disp[c].map(lambda x: round(float(x), 4))
    with open(os.path.join(tab_dir, "table_main_results.tex"), "w", encoding="utf-8") as f:
        f.write(
            df_to_latex_table(
                disp,
                "Test-set performance (1-h horizon), city-average hourly Yerevan PM2.5.",
                "tab:main_results",
                floatfmt="%.4f",
            )
        )

    # Pivot MAE by horizon
    if not df_hz.empty:
        pvt = df_hz.pivot_table(index="model", columns="horizon_h", values="MAE", aggfunc="first")
        pvt = pvt.reindex(sorted(pvt.index))
        pvt.columns = [f"h={int(c)}" for c in pvt.columns]
        pvt = pvt.reset_index().rename(columns={"model": "Model"})
        for c in pvt.columns:
            if c != "Model":
                pvt[c] = pvt[c].map(lambda x: round(float(x), 3))
        with open(os.path.join(tab_dir, "table_mae_by_horizon.tex"), "w", encoding="utf-8") as f:
            f.write(
                df_to_latex_table(
                    pvt,
                    "Test MAE by forecast horizon (hours).",
                    "tab:mae_horizon",
                    floatfmt="%.3f",
                )
            )

    df_dm_disp = df_dm.copy()
    for c in ("DM", "p_value_two_sided", "n"):
        if c in df_dm_disp.columns:
            df_dm_disp[c] = df_dm_disp[c].map(lambda x: round(float(x), 6) if c != "n" else int(x))
    with open(os.path.join(tab_dir, "table_diebold_mariano.tex"), "w", encoding="utf-8") as f:
        f.write(
            df_to_latex_table(
                df_dm_disp,
                "Diebold--Mariano tests on test-set absolute errors (paired; HAC lag 1).",
                "tab:dm",
                floatfmt="%.6f",
            )
        )

    summary: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "best_mae_model_1h": str(df_main.loc[df_main["MAE"].idxmin(), "Model"]),
        "best_mae_value_1h": float(df_main["MAE"].min()),
        "n_models_1h": int(len(df_main)),
    }
    if not df_hz.empty:
        summary["horizon_models"] = sorted(df_hz["model"].unique().tolist())

    boot_rows: List[Dict[str, Any]] = []
    if os.path.isfile(paths["bundle"]):
        z = np.load(paths["bundle"])
        y = z["y_test"]
        for name in sorted(z.files):
            if name == "y_test":
                continue
            pred = z[name]
            if pred.shape != y.shape:
                continue
            bi = bootstrap_mae_ci(y, pred)
            boot_rows.append({"model": name, **bi})
        summary["bootstrap_mae_95ci"] = boot_rows
        if boot_rows:
            bdf = pd.DataFrame(boot_rows)
            bdf["interval"] = bdf.apply(
                lambda r: f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                axis=1,
            )
            out_b = bdf[["model", "mae", "interval"]].copy()
            out_b.columns = ["Model", "MAE", "95pct_CI_bootstrap"]
            with open(os.path.join(tab_dir, "table_bootstrap_mae_ci.tex"), "w", encoding="utf-8") as f:
                f.write(
                    df_to_latex_table(
                        out_b,
                        "Bootstrap 95\\% intervals for mean absolute error (test set; i.i.d. resampling of hours).",
                        "tab:bootstrap_mae",
                        floatfmt="%.3f",
                    )
                )
    else:
        summary["bootstrap_mae_95ci"] = None
        print("No prediction bundle found; bootstrap table skipped.", file=sys.stderr)

    def _sanitize_json(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _sanitize_json(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize_json(v) for v in x]
        if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
            return None
        return x

    with open(os.path.join(paper_root, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, allow_nan=False)

    print(f"Wrote {tab_dir}/*.tex and paper/results_summary.json")


if __name__ == "__main__":
    main()
