from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence


@dataclass
class DataConfig:
    data_root: Path
    city_slug: str = "yerevan"
    target_col: str = "avg_pm2.5"
    freq: str = "h"
    short_gap_max: int = 3
    medium_gap_max: int = 24
    station_metadata_path: Path | None = None
    prefer_district_grouping: bool = True
    run_district_unit_forecasts: bool = True
    run_station_unit_forecasts: bool = True
    district_unit_min_feature_rows: int = 1500
    station_unit_min_feature_rows: int = 5000
    max_station_unit_forecasts: int = 100
    # Run ablation: skip hour-of-day median for medium gaps (sensitivity; slower)
    run_imputation_sensitivity_ablation: bool = False


@dataclass
class FeatureConfig:
    # Past-only lags for target and rolling alignment (in hours)
    lag_hours: Sequence[int] = (1, 2, 3, 6, 12, 24, 48, 72)
    horizons: Sequence[int] = (1, 2, 3, 4)
    include_covariates: Sequence[str] = (
        "avg_pm10",
        "avg_temperature",
        "avg_pressure",
        "avg_humidity",
    )
    min_covariate_coverage: float = 0.5
    # Rolling window lengths (hours) for shifted rolling stats; uses only t-1 and earlier
    rolling_windows: Sequence[int] = (6, 12, 24, 48)
    # Fourier / calendar seasonality
    fourier_annual: bool = True
    fourier_paired_terms: int = 2
    # Target–met product interactions (uses pm25 lags 1,24 and covar lags 1,24)
    include_met_pm25_interactions: bool = True
    # Volatility / change (causal: diff and trailing vol)
    include_volatility_features: bool = True
    # Station-level unit runs: smaller feature set for speed (override in _forecast_1h_single_series)
    use_minimal_feature_set_for_units: bool = True


@dataclass
class ValidationConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    walk_forward_enabled: bool = True
    walk_forward_test_size: int = 24 * 30
    walk_forward_step: int = 24 * 30
    walk_forward_min_folds: int = 3
    max_walk_forward_folds: int | None = 12
    # expanding: train is [0, t); sliding: train is [t-train_window, t)
    walk_forward_mode: Literal["expanding", "sliding"] = "expanding"
    walk_forward_train_window: int | None = None  # required for sliding, e.g. 24*365*3
    # SARIMA in every fold; by default reuse orders from the main 70% train search (refit per fold; much faster)
    walk_forward_sarima: bool = True
    walk_forward_reuse_main_sarima_orders: bool = True
    # Re-run Hyperopt per fold (very expensive); if False, reuse first global-tuned params
    walk_forward_refit_hyperopt: bool = False
    # DeepAR per fold (very expensive)
    walk_forward_refit_deepar: bool = False
    # Cap DeepAR to first N walk-forward folds (0 = no DeepAR in WF)
    walk_forward_max_folds_deepar: int = 0


@dataclass
class StatsConfig:
    bootstrap_n: int = 2000
    # primary CI method
    block_bootstrap: bool = True
    # fallback block length: max(2*horizon_hours, 24) when autocorrelation not used
    block_len_min: int = 24
    block_len_max: int = 168
    # Diebold–Mariano: hac_lag 0 = auto (Newey–West style rule + horizon)
    hac_lag: int = 0
    horizon_hours_for_dm: int = 1
    iid_bootstrap_also: bool = True  # keep secondary row in JSON


@dataclass
class TuningConfig:
    enabled: bool = True
    max_evals: int = 80
    random_state: int = 42


@dataclass
class ModelConfig:
    include_sarima: bool = True
    include_deepar: bool = True
    deepar_backend: str = "neuralforecast"
    deepar_max_steps: int = 300
    deepar_input_size: int | None = None
    # multi-horizon DeepAR in auxiliary eval (1 = main path only)
    deepar_multistep_mode: bool = True
    random_state: int = 42
    sarima_criterion: Literal["aic", "bic"] = "aic"


@dataclass
class OutputConfig:
    output_root: Path
    plots_dir: Path
    tables_dir: Path
    json_dir: Path
    images_dir: Path
    save_processed_data: bool = True
    # Case-study: run full stack on 1–2 top districts
    case_study_district_ids: tuple[int, ...] = (7, 10)  # e.g. Kentron, Shengavit — indices into order where valid


@dataclass
class PipelineConfig:
    project_root: Path
    data: DataConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig | None = None

    @staticmethod
    def default(project_root: str | Path) -> "PipelineConfig":
        root = Path(project_root).resolve()
        output_root = root / "results"
        out = OutputConfig(
            output_root=output_root,
            plots_dir=output_root / "plots",
            tables_dir=output_root / "tables",
            json_dir=output_root / "json",
            images_dir=root / "images",
        )
        data = DataConfig(data_root=root / "data" / "raw" / "Air Quality Data")
        return PipelineConfig(project_root=root, data=data, output=out)
