from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class DataConfig:
    data_root: Path
    city_slug: str = "yerevan"
    target_col: str = "avg_pm2.5"
    freq: str = "h"
    short_gap_max: int = 3
    medium_gap_max: int = 24
    station_metadata_path: Path | None = None


@dataclass
class FeatureConfig:
    lag_hours: Sequence[int] = (1, 2, 3, 24)
    horizons: Sequence[int] = (1, 2, 3, 4)
    include_covariates: Sequence[str] = (
        "avg_pm10",
        "avg_temperature",
        "avg_pressure",
        "avg_humidity",
    )
    min_covariate_coverage: float = 0.5


@dataclass
class ValidationConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    walk_forward_enabled: bool = True
    walk_forward_test_size: int = 24 * 30
    walk_forward_step: int = 24 * 30
    walk_forward_min_folds: int = 3


@dataclass
class TuningConfig:
    enabled: bool = True
    max_evals: int = 40
    random_state: int = 42


@dataclass
class ModelConfig:
    include_sarima: bool = True
    include_deepar: bool = True
    deepar_backend: str = "neuralforecast"
    random_state: int = 42


@dataclass
class OutputConfig:
    output_root: Path
    plots_dir: Path
    tables_dir: Path
    json_dir: Path
    images_dir: Path
    save_processed_data: bool = True


@dataclass
class PipelineConfig:
    project_root: Path
    data: DataConfig
    features: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
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

