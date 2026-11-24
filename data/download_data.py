"""
Automated downloader and extractor for the UCI Bank Marketing Dataset.
Complies with course rules: reproducible script, no large raw datasets in Git.
"""

import os
import zipfile
import urllib.request
import sys

DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(TARGET_DIR, "bank_marketing.zip")
EXTRACTED_SUBZIP = os.path.join(TARGET_DIR, "bank.zip")
TARGET_CSV = os.path.join(TARGET_DIR, "bank-full.csv")

def download_and_extract():
    if os.path.exists(TARGET_CSV):
        print(f"[INFO] '{TARGET_CSV}' already exists. Skipping download.")
        return

    print(f"[INFO] Downloading UCI Bank Marketing dataset from: {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print(f"[INFO] Download complete. Extracting parent archive...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(TARGET_DIR)

        if os.path.exists(EXTRACTED_SUBZIP):
            print(f"[INFO] Extracting inner archive '{EXTRACTED_SUBZIP}'...")
            with zipfile.ZipFile(EXTRACTED_SUBZIP, 'r') as inner_zip:
                inner_zip.extractall(TARGET_DIR)

        # Cleanup intermediate zips
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        if os.path.exists(EXTRACTED_SUBZIP):
            os.remove(EXTRACTED_SUBZIP)
        additional_zip = os.path.join(TARGET_DIR, "bank-additional.zip")
        if os.path.exists(additional_zip):
            os.remove(additional_zip)

        if os.path.exists(TARGET_CSV):
            print(f"[SUCCESS] Dataset extracted successfully to: {TARGET_CSV}")
        else:
            print(f"[WARNING] Extraction completed, but '{TARGET_CSV}' was not found.")
    except Exception as e:
        print(f"[ERROR] Failed to download or extract dataset: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_and_extract()
