"""Experiment logging: one CSV row per run (spec section 3.5)."""
import csv
import os
from datetime import datetime

FIELDS = [
    "run_id", "date", "author", "study", "model", "hyperparameters", "seed",
    "train_time_s", "params", "epochs_run", "train_loss", "val_loss",
    "train_auc", "val_auc", "val_recall", "val_precision", "val_f1", "notes",
]


def next_run_id(path: str) -> str:
    """Return the next sequential run ID (R001, R002, ...) for the log at `path`."""
    if not os.path.exists(path):
        return "R001"
    with open(path, newline="") as f:
        n = sum(1 for _ in csv.DictReader(f))
    return f"R{n + 1:03d}"


def log_run(path: str, row: dict) -> str:
    """Append one run to the experiment CSV, creating it with a header if needed.

    Missing fields are left blank; `run_id` and `date` are filled automatically.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = dict(row)
    row.setdefault("run_id", next_run_id(path))
    row.setdefault("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return row["run_id"]
