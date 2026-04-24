"""
Download Air Quality Data from Google Drive.

Package-style usage:
    from download_data import download_data
    download_data()
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

DATA_ZIP_URL = "https://drive.google.com/uc?id=1QaDT5_XFKUMXbYoZLO8BsfzCstlNgwnp"


def _resolve_target_dir(project_root: Path, target_dir: str | None) -> Path:
    if target_dir:
        return Path(target_dir).resolve()
    return (project_root / "data" / "raw" / "Air Quality Data").resolve()


def download_data(
    target_dir: str | None = None,
    *,
    force: bool = False,
    quiet: bool = False,
) -> Path:
    """Download and extract data if missing, then return dataset directory."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required. Install with `pip install gdown`.") from exc

    project_root = Path(__file__).resolve().parent
    out_dir = _resolve_target_dir(project_root, target_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if out_dir.is_dir() and any(out_dir.iterdir()) and not force:
        if not quiet:
            print(f"Data already present at: {out_dir}")
        return out_dir

    if not quiet:
        print("Downloading data from Google Drive...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "air_quality_data.zip"
        gdown.download(DATA_ZIP_URL, str(zip_path), quiet=quiet, fuzzy=True)
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise RuntimeError("Download failed or zip is empty.")
        if not quiet:
            print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(out_dir.parent))

    # Normalize extracted folder name if needed.
    if not out_dir.exists():
        candidates = [p for p in out_dir.parent.iterdir() if p.is_dir()]
        candidates = [p for p in candidates if p.name != out_dir.name]
        if len(candidates) == 1:
            candidates[0].rename(out_dir)
        elif candidates:
            for cand in candidates:
                if "Air Quality Data" in cand.name:
                    cand.rename(out_dir)
                    break
    if not out_dir.exists():
        raise RuntimeError("Extraction finished, but data folder was not found.")
    if not quiet:
        print(f"Done. Data is in: {out_dir}")
    return out_dir


def main() -> None:
    download_data()


if __name__ == "__main__":
    main()
