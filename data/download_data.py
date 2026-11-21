"""Dataset Download & Extraction Script for UCI Bank Marketing.

Owner / Author: Esraa Gamal (@esraagamal11mohamed-cell)
Role: Role 2 - Data Engineer
Course: Neural Networks (AIE231)
Specification Reference: Section 6.1 ("data/ download script and README")

Usage:
    python data/download_data.py
"""

import os
import sys
import zipfile
import urllib.request
import shutil
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_CSV = os.path.join(DATA_DIR, "bank-additional-full.csv")


def ensure_dataset():
    """Ensure bank-additional-full.csv is present in the data/ directory."""
    if os.path.exists(TARGET_CSV):
        df = pd.read_csv(TARGET_CSV, sep=";")
        print(f"[OK] Dataset already present at '{TARGET_CSV}'. Shape: {df.shape}")
        return TARGET_CSV

    print(f"[*] Downloading UCI Bank Marketing dataset from: {UCI_URL}...")
    zip_dest = os.path.join(DATA_DIR, "bank_marketing_archive.zip")
    
    try:
        urllib.request.urlretrieve(UCI_URL, zip_dest)
        print("[+] Download complete. Extracting archive...")
        
        # Step 1: Extract primary zip
        tmp_dir = os.path.join(DATA_DIR, "_tmp_extract")
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_dest, "r") as z:
            z.extractall(tmp_dir)
            
        # Step 2: Extract bank-additional.zip
        inner_zip = os.path.join(tmp_dir, "bank-additional.zip")
        if os.path.exists(inner_zip):
            tmp_inner = os.path.join(DATA_DIR, "_tmp_inner")
            os.makedirs(tmp_inner, exist_ok=True)
            with zipfile.ZipFile(inner_zip, "r") as z:
                z.extractall(tmp_inner)
            src_csv = os.path.join(tmp_inner, "bank-additional", "bank-additional-full.csv")
            if os.path.exists(src_csv):
                shutil.copy(src_csv, TARGET_CSV)
                print(f"[+] Successfully extracted '{TARGET_CSV}'.")
            shutil.rmtree(tmp_inner, ignore_errors=True)
            
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(zip_dest):
            os.remove(zip_dest)
            
    except Exception as e:
        print(f"[!] Automated download failed: {e}")
        print("[*] Please download 'bank-additional.zip' manually from:")
        print("    https://archive.ics.uci.edu/dataset/222/bank+marketing")
        print("    and extract 'bank-additional-full.csv' into 'data/'.")
        
    if os.path.exists(TARGET_CSV):
        df = pd.read_csv(TARGET_CSV, sep=";")
        print(f"[OK] Verification passed: {df.shape[0]:,} rows, {df.shape[1]} columns.")
        return TARGET_CSV
    else:
        raise FileNotFoundError(f"Failed to extract '{TARGET_CSV}'.")


if __name__ == "__main__":
    ensure_dataset()
