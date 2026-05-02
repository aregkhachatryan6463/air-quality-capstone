from __future__ import annotations

import argparse
import os

from download_data import download_data
from src.config.settings import PipelineConfig
from src.pipeline.run import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete PM2.5 forecasting pipeline")
    parser.add_argument("--project-root", default=os.getcwd(), help="Project root path")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download check")
    parser.add_argument("--no-tuning", action="store_true", help="Disable Hyperopt tuning")
    parser.add_argument("--max-evals", type=int, default=None, help="Override Hyperopt max evaluations")
    parser.add_argument("--disable-deepar", action="store_true", help="Disable DeepAR training/evaluation")
    parser.add_argument("--disable-sarima", action="store_true", help="Disable SARIMA benchmark")
    parser.add_argument("--disable-walk-forward", action="store_true", help="Disable walk-forward validation")
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Comma-separated forecast horizons in hours (example: 1,2,3,4,6,12,24)",
    )
    parser.add_argument("--disable-district-grouping", action="store_true", help="Force city aggregate source (no district grouping)")
    parser.add_argument("--no-district-unit-forecasts", action="store_true", help="Skip per-administrative-district 1h forecasts")
    parser.add_argument("--no-station-unit-forecasts", action="store_true", help="Skip per-station 1h forecasts")
    parser.add_argument(
        "--imputation-ablation",
        action="store_true",
        help="Compare Ridge val-MAE with/without hour-of-day fill for medium gaps (methodology)",
    )
    parser.add_argument(
        "--walk-forward-refit-tuning",
        action="store_true",
        help="Re-fit Hyperopt in every walk-forward fold (slow; default: reuse main-split tuned params)",
    )
    parser.add_argument(
        "--walk-forward-deepar-folds",
        type=int,
        default=0,
        help="If >0, retrain DeepAR on each of the first N walk-forward folds (0=DeepAR in WF off)",
    )
    parser.add_argument(
        "--walk-forward-sarima-search",
        action="store_true",
        help="In each walk-forward fold, re-run auto_arima (very slow). Default: reuse main-split SARIMA orders.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig.default(args.project_root)
    if args.disable_deepar:
        cfg.models.include_deepar = False
    if args.disable_sarima:
        cfg.models.include_sarima = False
    if args.disable_walk_forward:
        cfg.validation.walk_forward_enabled = False
    if args.horizons:
        parsed = []
        for x in args.horizons.split(","):
            x = x.strip()
            if not x:
                continue
            h = int(x)
            if h <= 0:
                raise ValueError(f"Invalid horizon '{x}'. Horizons must be positive integers.")
            parsed.append(h)
        if not parsed:
            raise ValueError("No valid horizons were provided to --horizons.")
        cfg.features.horizons = tuple(sorted(set(parsed)))
    if args.disable_district_grouping:
        cfg.data.prefer_district_grouping = False
    if args.no_district_unit_forecasts:
        cfg.data.run_district_unit_forecasts = False
    if args.no_station_unit_forecasts:
        cfg.data.run_station_unit_forecasts = False
    if args.max_evals is not None and args.max_evals > 0:
        cfg.tuning.max_evals = int(args.max_evals)
    if args.imputation_ablation:
        cfg.data.run_imputation_sensitivity_ablation = True
    if args.walk_forward_refit_tuning:
        cfg.validation.walk_forward_refit_hyperopt = True
    if int(args.walk_forward_deepar_folds) > 0:
        cfg.validation.walk_forward_refit_deepar = True
        cfg.validation.walk_forward_max_folds_deepar = int(args.walk_forward_deepar_folds)
    if args.walk_forward_sarima_search:
        cfg.validation.walk_forward_reuse_main_sarima_orders = False
    if not args.skip_download:
        download_data(target_dir=str(cfg.project_root / "data" / "raw" / "Air Quality Data"))
    print("Starting full pipeline (tuning, SARIMA, walk-forward, DeepAR per config; may take 30–90+ min)...", flush=True)
    result = run_full_pipeline(cfg, enable_tuning=not args.no_tuning)
    print("Done.", flush=True)
    print(f"Best 1h model: {result['summary']['best_mae_model_1h']}", flush=True)
    print(f"JSON summary: {cfg.output.json_dir / 'results_summary.json'}", flush=True)
    print(f"Tables (CSV): {cfg.output.tables_dir}", flush=True)
    print(f"Figures (PNG): {cfg.output.plots_dir}", flush=True)
    print(f"Paper figure copies: {cfg.output.images_dir}", flush=True)


if __name__ == "__main__":
    main()

