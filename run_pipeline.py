from __future__ import annotations

import argparse
import os

from src.config.settings import PipelineConfig
from src.download.fetch import ensure_data
from src.pipeline.run import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete PM2.5 forecasting pipeline")
    parser.add_argument("--project-root", default=os.getcwd(), help="Project root path")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download check")
    parser.add_argument("--no-tuning", action="store_true", help="Disable Hyperopt tuning")
    parser.add_argument("--disable-deepar", action="store_true", help="Disable DeepAR training/evaluation")
    parser.add_argument("--disable-sarima", action="store_true", help="Disable SARIMA benchmark")
    parser.add_argument("--disable-walk-forward", action="store_true", help="Disable walk-forward validation")
    args = parser.parse_args()

    cfg = PipelineConfig.default(args.project_root)
    if args.disable_deepar:
        cfg.models.include_deepar = False
    if args.disable_sarima:
        cfg.models.include_sarima = False
    if args.disable_walk_forward:
        cfg.validation.walk_forward_enabled = False
    if not args.skip_download:
        ensure_data(cfg.project_root)
    result = run_full_pipeline(cfg, enable_tuning=not args.no_tuning)
    print("Done.")
    print(f"Best 1h model: {result['summary']['best_mae_model_1h']}")
    print(f"results_summary.json: {cfg.output.json_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()

