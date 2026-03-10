"""
Download Air Quality Data from Google Drive.

The dataset is hosted on Google Drive due to size. Run from the project root:
    python download_data.py
Creates the folder 'Air Quality Data' in the current directory.
"""

import os
import sys
import zipfile
import tempfile

DATA_ZIP_URL = "https://drive.google.com/uc?id=1QaDT5_XFKUMXbYoZLO8BsfzCstlNgwnp"


def main():
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown

    project_root = os.getcwd()
    target_dir = os.path.join(project_root, "Air Quality Data")
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        print(f"Folder 'Air Quality Data' already exists and is not empty. Skipping download.")
        return

    print("Downloading data from Google Drive...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "air_quality_data.zip")
        gdown.download(DATA_ZIP_URL, zip_path, quiet=False, fuzzy=True)

        if not os.path.isfile(zip_path) or os.path.getsize(zip_path) == 0:
            print("Download failed or file is empty. Check the link and sharing settings.")
            sys.exit(1)

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(project_root)

    if os.path.isdir(target_dir):
        print(f"Done. Data is in: {target_dir}")
    else:
        # Zip might have a different top-level folder name
        for name in os.listdir(project_root):
            path = os.path.join(project_root, name)
            if os.path.isdir(path) and name != "Air Quality Data":
                alt = os.path.join(project_root, "Air Quality Data")
                os.rename(path, alt)
                print(f"Renamed '{name}' to 'Air Quality Data'. Done: {alt}")
                return
        print("Extraction finished. If 'Air Quality Data' is missing, ensure the zip contains that folder.")


if __name__ == "__main__":
    main()
