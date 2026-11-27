"""Architecture and regularization ablations (Model Architect, Role 4).

Each ablation changes ONE factor from a fixed reference configuration and keeps
everything else fixed. Uses TRAIN and VALIDATION data only (never the test set).
Every run is logged to experiments/experiment_log.csv.

Usage:
    python -m src.run_ablations --synthetic                  # smoke test
    python -m src.run_ablations                              # real data, all ablations
    python -m src.run_ablations --ablation dropout           # one ablation only
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from src.experiment_tracker import log_run
from src.models import build_mlp
from src.plots import plot_learning_curves, plot_sweep
from src.run_experiment import (CURVE_DIR, LOG_PATH, get_optimizer, load_data,
                                set_seed)

# Fixed reference. Training settings are read from configs/reference.json when it exists.
DEFAULT_REF = {
    "model": "ablation_mlp",
    "hidden_units": [128, 64, 32],
    "dropout": 0.3,
    "activation": "relu",
    "use_batchnorm": False,
    "initializer": "he_normal",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 100,
    "patience": 10,
    "seed": 42,
    "use_class_weights": True,
}

# name -> (config key, values). One factor per ablation.
ABLATIONS = {
    "dropout": ("dropout", [0.0, 0.3, 0.5]),
    "activation": ("activation", ["relu", "tanh", "sigmoid"]),
    "batchnorm": ("use_batchnorm", [False, True]),
    "initializer": ("initializer", ["he_normal", "glorot_uniform"]),
    "depth": ("hidden_units", [[64], [128, 64], [128, 64, 32], [128, 64, 32, 16]]),
}


def load_reference(path: str) -> dict:
    """Return the reference config: defaults overridden by configs/reference.json."""
    ref = dict(DEFAULT_REF)
    if os.path.exists(path):
        with open(path) as f:
            file_cfg = json.load(f)
        file_cfg.pop("sweeps", None)
        ref.update(file_cfg)
    return ref


def run_one(cfg: dict, data: tuple, author: str, study: str, tag: str) -> dict:
    """Train one configuration, log it to the CSV, save its curves, return metrics."""
    X_tr, y_tr, X_val, y_val = data
    set_seed(cfg["seed"])
    model = build_mlp(
        X_tr.shape[1], hidden_units=cfg["hidden_units"], dropout=cfg["dropout"],
        activation=cfg["activation"], use_batchnorm=cfg["use_batchnorm"],
        initializer=cfg["initializer"], name="ablation_mlp")
    model.compile(
        optimizer=get_optimizer(cfg["optimizer"], cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")])

    cw = None
    if cfg.get("use_class_weights"):
        classes = np.unique(y_tr)
        w = compute_class_weight("balanced", classes=classes, y=y_tr)
        cw = {int(c): float(v) for c, v in zip(classes, w)}

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                     class_weight=cw, callbacks=[stop], verbose=0)
    elapsed = time.time() - t0

    pred = (model.predict(X_val, verbose=0).ravel() >= 0.5).astype(int)
    h = hist.history
    best = int(np.argmin(h["val_loss"]))
    row = {
        "author": author, "study": f"ablation_{study}", "model": cfg["model"],
        "hyperparameters": json.dumps(cfg), "seed": cfg["seed"],
        "train_time_s": round(elapsed, 1), "params": model.count_params(),
        "epochs_run": len(h["loss"]),
        "train_loss": round(h["loss"][best], 4), "val_loss": round(h["val_loss"][best], 4),
        "train_auc": round(h["auc"][best], 4), "val_auc": round(h["val_auc"][best], 4),
        "val_recall": round(recall_score(y_val, pred, zero_division=0), 4),
        "val_precision": round(precision_score(y_val, pred, zero_division=0), 4),
        "val_f1": round(f1_score(y_val, pred, zero_division=0), 4),
    }
    row["run_id"] = log_run(LOG_PATH, row)
    plot_learning_curves(h, f"{row['run_id']} {tag}", f"{CURVE_DIR}/{row['run_id']}_{tag}.png")
    return row


def run_ablation(name: str, ref: dict, data: tuple, author: str) -> pd.DataFrame:
    """Run one ablation: vary a single factor, keep the rest of `ref` fixed."""
    key, values = ABLATIONS[name]
    results = []
    for v in values:
        cfg = dict(ref)
        cfg[key] = v
        tag = f"{name}={v}".replace(" ", "")
        r = run_one(cfg, data, author, study=name, tag=tag)
        r[name] = str(v)
        results.append(r)
        print(f"{r['run_id']} {tag:<32} params={r['params']:<6} val_auc={r['val_auc']} "
              f"val_recall={r['val_recall']} val_f1={r['val_f1']} epochs={r['epochs_run']}")
    df = pd.DataFrame(results)
    plot_sweep(df, name, "val_auc", f"{CURVE_DIR}/ablation_{name}.png")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reference.json")
    p.add_argument("--author", default="Jessica")
    p.add_argument("--ablation", default=None, choices=list(ABLATIONS), help="run only this one")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()

    reference = load_reference(a.config)
    data = load_data(a.synthetic)
    for ablation_name in ABLATIONS:
        if a.ablation and ablation_name != a.ablation:
            continue
        run_ablation(ablation_name, reference, data, a.author)
