"""Download the UCI Bank Marketing dataset into data/raw/.

Source : https://archive.ics.uci.edu/dataset/222/bank+marketing
Licence: CC BY 4.0. Citation: Moro, Cortez & Rita (2014), "A Data-Driven
         Approach to Predict the Success of Bank Telemarketing", DSS 62.

Usage: python data/download_data.py [--out data/raw]
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
WANTED = "bank-full.csv"


def main() -> None:
    """Fetch the outer zip, open the nested bank.zip and extract bank-full.csv."""
    default_out = Path(__file__).resolve().parent / "raw"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=default_out)
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    (out / WANTED).write_bytes(inner.read(WANTED))
    print(f"Saved {out / WANTED}")


if __name__ == "__main__":
    main()
