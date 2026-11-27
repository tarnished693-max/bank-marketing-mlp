"""Experiment runner: one-factor-at-a-time sweeps from a fixed reference config.

Uses TRAIN and VALIDATION data only. The test set is never loaded here.

Usage:
    python -m src.run_experiment --config configs/reference.json --author Ahmed
    python -m src.run_experiment --config configs/reference.json --author Ahmed --factor optimizer
    python -m src.run_experiment --synthetic          # smoke test, no real data
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from src.experiment_tracker import log_run
from src.plots import plot_learning_curves, plot_sweep

LOG_PATH = "experiments/experiment_log.csv"
CURVE_DIR = "experiments/curves"


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_optimizer(name: str, lr: float):
    """Return SGD, SGD with momentum (0.9) or Adam."""
    if name == "sgd":
        return tf.keras.optimizers.SGD(lr)
    if name == "sgd_momentum":
        return tf.keras.optimizers.SGD(lr, momentum=0.9)
    if name == "adam":
        return tf.keras.optimizers.Adam(lr)
    raise ValueError(f"Unknown optimizer: {name}")


def build_mlp(input_dim: int, cfg: dict) -> tf.keras.Model:
    """Build a binary-classification MLP: Dense(+Dropout) blocks, sigmoid output."""
    layers = [tf.keras.Input(shape=(input_dim,))]
    for u in cfg["hidden_units"]:
        layers.append(tf.keras.layers.Dense(u, activation=cfg["activation"]))
        if cfg["dropout"] > 0:
            layers.append(tf.keras.layers.Dropout(cfg["dropout"]))
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=get_optimizer(cfg["optimizer"], cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def run_one(cfg: dict, data: tuple, author: str, study: str, tag: str) -> dict:
    """Train one configuration, log it to the CSV, save its curves, return metrics."""
    X_tr, y_tr, X_val, y_val = data
    set_seed(cfg["seed"])
    model = build_mlp(X_tr.shape[1], cfg)

    cw = None
    if cfg.get("use_class_weights"):
        w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw = {int(c): float(v) for c, v in zip(np.unique(y_tr), w)}

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg["patience"], restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                     epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                     class_weight=cw, callbacks=[stop], verbose=0)
    elapsed = time.time() - t0

    prob = model.predict(X_val, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    h = hist.history
    best = int(np.argmin(h["val_loss"]))
    row = {
        "author": author, "study": study, "model": cfg["model"],
        "hyperparameters": json.dumps({k: v for k, v in cfg.items() if k != "sweeps"}),
        "seed": cfg["seed"], "train_time_s": round(elapsed, 1),
        "params": model.count_params(), "epochs_run": len(h["loss"]),
        "train_loss": round(h["loss"][best], 4), "val_loss": round(h["val_loss"][best], 4),
        "train_auc": round(h["auc"][best], 4), "val_auc": round(h["val_auc"][best], 4),
        "val_recall": round(recall_score(y_val, pred, zero_division=0), 4),
        "val_precision": round(precision_score(y_val, pred, zero_division=0), 4),
        "val_f1": round(f1_score(y_val, pred, zero_division=0), 4),
    }
    row["run_id"] = log_run(LOG_PATH, row)
    plot_learning_curves(h, f"{row['run_id']} {tag}", f"{CURVE_DIR}/{row['run_id']}_{tag}.png")
    return row


def sweep(ref: dict, data: tuple, author: str, factor: str | None = None) -> None:
    """Vary one factor at a time around the reference config and plot each study."""
    for name, values in ref["sweeps"].items():
        if factor and name != factor:
            continue
        results = []
        for v in values:
            cfg = {k: val for k, val in ref.items() if k != "sweeps"}
            cfg[name] = v
            tag = f"{name}={v}".replace(" ", "")
            r = run_one(cfg, data, author, study=name, tag=tag)
            r[name] = str(v)
            results.append(r)
            print(f"{r['run_id']} {tag:<28} val_auc={r['val_auc']} "
                  f"val_f1={r['val_f1']} epochs={r['epochs_run']}")
        plot_sweep(pd.DataFrame(results), name, "val_auc", f"{CURVE_DIR}/sweep_{name}.png")


def load_data(synthetic: bool) -> tuple:
    """Return (X_train, y_train, X_val, y_val). Never returns test data."""
    if synthetic:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(4000, 20)).astype("float32")
        y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=4000) > 1.6).astype(int)
        return X[:3000], y[:3000], X[3000:], y[3000:]
    from src.data import prepare_data
    d = prepare_data(save=False)
    return d.X_train, d.y_train, d.X_val, d.y_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/reference.json")
    p.add_argument("--author", default="Ahmed")
    p.add_argument("--factor", default=None, help="run only this sweep")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    with open(a.config) as f:
        ref = json.load(f)
    sweep(ref, load_data(a.synthetic), a.author, a.factor)
