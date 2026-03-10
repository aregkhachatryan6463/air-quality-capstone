"""
Download Air Quality Data from Google Drive.

The full dataset is too large for GitHub. It is hosted on Google Drive.
Before running: upload your 'Air Quality Data' folder as a ZIP to Google Drive,
share it with "Anyone with the link can view", then set GOOGLE_DRIVE_ZIP_LINK below.

Run from the project root:
    python download_data.py

This will create the folder 'Air Quality Data' in the current directory.
"""

import os
import sys
import zipfile
import tempfile

# ---------------------------------------------------------------------------
# SET THIS: Share your zip on Google Drive and paste the link here.
# The zip should contain a folder named "Air Quality Data" with all CSVs inside.
# Link format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
# Or use the direct download format: https://drive.google.com/uc?id=FILE_ID
# ---------------------------------------------------------------------------
GOOGLE_DRIVE_ZIP_LINK = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # <-- Replace YOUR_FILE_ID


def main():
    if "YOUR_FILE_ID" in GOOGLE_DRIVE_ZIP_LINK:
        print("Please set GOOGLE_DRIVE_ZIP_LINK in download_data.py to your Google Drive zip link.")
        print("Upload 'Air Quality Data' as a ZIP, share with 'Anyone with the link', then paste the link.")
        sys.exit(1)

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
        gdown.download(GOOGLE_DRIVE_ZIP_LINK, zip_path, quiet=False, fuzzy=True)

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
