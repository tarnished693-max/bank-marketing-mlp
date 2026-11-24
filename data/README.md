# Dataset Instructions: Bank Marketing (UCI)

## Overview
This project uses the **Bank Marketing Dataset** from the UCI Machine Learning Repository:
- **Source**: UCI Machine Learning Repository / S. Moro, P. Cortez, and P. Rita (2014)
- **Direct Link**: https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Task**: Binary classification predicting whether a bank client subscribes to a term deposit (`y` = `yes` / `no`).
- **Instances**: 45,211 rows.
- **Features**: 16 input features (age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome).
- **Target**: `y` (1 = yes, 0 = no). Class imbalance ratio: ~11.7% yes vs ~88.3% no.

## How to Download the Data
Do not commit raw CSV files to Git. Instead, run the provided automated script:

```bash
python data/download_data.py
```

This script downloads the official UCI zip archive, extracts `bank-full.csv`, and places it in this `data/` directory.

Alternatively, manually download `bank.zip` from UCI and extract `bank-full.csv` directly into `data/`.
