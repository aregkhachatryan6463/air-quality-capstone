"""
Sensitivity to PM2.5 capping and lag sets: runs run_pipeline() over configurations
and writes sensitivity_preprocessing_lags.csv (temp dirs; skips multi-horizon for speed).
"""

import os
import tempfile

import pandas as pd

from yerevan_pm25_forecasting import run_pipeline


def main() -> None:
    out_final = os.getcwd()
    rows = []
    lag_sets = (
        (1, 2, 3, 24),
        (1, 24),
        (1, 2, 3, 6, 12, 24),
    )
    for cap in (True, False):
        for lags in lag_sets:
            with tempfile.TemporaryDirectory() as tmp:
                r = run_pipeline(
                    cap_extreme=cap,
                    lag_hours=lags,
                    verbose=False,
                    save_plots=False,
                    skip_multi_horizon=True,
                    output_dir=tmp,
                )
            tag = f"cap={cap}_lags={list(lags)}"
            for _, row in r["results_1h"].iterrows():
                rows.append(
                    {
                        "config": tag,
                        "cap_extreme": cap,
                        "lags": str(list(lags)),
                        "model": row["Model"],
                        "MAE": row["MAE"],
                        "RMSE": row["RMSE"],
                        "R2": row["R2"],
                    }
                )
    df = pd.DataFrame(rows)
    path = os.path.join(out_final, "sensitivity_preprocessing_lags.csv")
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
