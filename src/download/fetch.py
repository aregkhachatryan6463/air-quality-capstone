from __future__ import annotations

from pathlib import Path

from download_data import download_data


def ensure_data(root: str | Path) -> Path:
    root = Path(root).resolve()
    return download_data(target_dir=str(root / "data" / "raw" / "Air Quality Data"))

