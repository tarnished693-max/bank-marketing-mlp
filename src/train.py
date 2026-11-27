"""Training entry point and experiment runner for Bank Marketing neural models.

Allows training Model A, B, or C using a YAML config and random seed,
evaluating on the validation split only, logging to experiments/experiment_log.csv,
and saving the trained Keras model to models/.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.data import prepare_data, resolve_path
from src.experiment_tracker import log_run
from src.models import build_model_a, build_model_b, build_model_c
from src.plots import plot_learning_curves


MODEL_BUILDERS = {
    "model_a": build_model_a,
    "model_b": build_model_b,
    "model_c": build_model_c,
}


def set_seed(seed: int) -> None:
    """Fix Python, NumPy, and TensorFlow random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_optimizer(name: str, lr: float) -> tf.keras.optimizers.Optimizer:
    """Return configured Keras optimizer."""
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "sgd_momentum":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}")


def train_from_config(config_path: str, seed_override: int | None = None, save_model: bool = True) -> dict:
    """Train a configured model on TRAIN and evaluate on VALIDATION only."""
    cfg = load_config(config_path)
    seed = seed_override if seed_override is not None else int(cfg.get("random_seed", 42))
    set_seed(seed)

    d = prepare_data(save=True)
    input_dim = d.X_train.shape[1]

    m_cfg = cfg.get("model", {})
    arch = m_cfg.get("architecture", "model_a")
    if arch not in MODEL_BUILDERS:
        raise ValueError(f"Unknown architecture {arch!r}; expected one of {list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[arch](input_dim)

    t_cfg = cfg.get("training", {})
    lr = float(t_cfg.get("learning_rate", 1e-3))
    opt_name = t_cfg.get("optimizer", "adam")
    model.compile(
        optimizer=get_optimizer(opt_name, lr),
        loss=t_cfg.get("loss", "binary_crossentropy"),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    cw = d.class_weights if t_cfg.get("use_class_weights", True) else None
    es_cfg = t_cfg.get("early_stopping", {})
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            patience=int(es_cfg.get("patience", 10)),
            restore_best_weights=bool(es_cfg.get("restore_best_weights", True)),
        )
    ]

    t0 = time.time()
    hist = model.fit(
        d.X_train,
        d.y_train,
        validation_data=(d.X_val, d.y_val),
        epochs=int(t_cfg.get("epochs", 50)),
        batch_size=int(t_cfg.get("batch_size", 64)),
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    val_prob = model.predict(d.X_val, verbose=0).ravel()
    val_pred = (val_prob >= 0.5).astype(int)
    h = hist.history
    best_ep = int(np.argmin(h["val_loss"]))

    row = {
        "author": cfg.get("author", "Yousef"),
        "study": cfg.get("experiment_name", arch),
        "model": arch,
        "hyperparameters": json.dumps({"lr": lr, "optimizer": opt_name, "batch_size": t_cfg.get("batch_size", 64)}),
        "seed": seed,
        "train_time_s": round(elapsed, 1),
        "params": model.count_params(),
        "epochs_run": len(h["loss"]),
        "train_loss": round(float(h["loss"][best_ep]), 4),
        "val_loss": round(float(h["val_loss"][best_ep]), 4),
        "train_auc": round(float(h["auc"][best_ep]), 4),
        "val_auc": round(float(roc_auc_score(d.y_val, val_prob)), 4),
        "val_recall": round(float(recall_score(d.y_val, val_pred, zero_division=0)), 4),
        "val_precision": round(float(precision_score(d.y_val, val_pred, zero_division=0)), 4),
        "val_f1": round(float(f1_score(d.y_val, val_pred, zero_division=0)), 4),
    }

    log_path = str(resolve_path("experiments/experiment_log.csv"))
    row["run_id"] = log_run(log_path, row)
    curve_path = str(resolve_path(f"experiments/curves/{row['run_id']}_{arch}_seed{seed}.png"))
    plot_learning_curves(h, f"{row['run_id']} {arch} (seed={seed})", curve_path)

    if save_model:
        model_path = resolve_path(f"models/{arch}.keras")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        print(f"[*] Saved {arch} to {model_path}")

    print(
        f"[*] {row['run_id']} {arch}: val_auc={row['val_auc']} "
        f"val_pr_auc={average_precision_score(d.y_val, val_prob):.4f} "
        f"val_recall={row['val_recall']} val_f1={row['val_f1']}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Bank Marketing MLP models")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    train_from_config(args.config, seed_override=args.seed, save_model=not args.no_save)


if __name__ == "__main__":
    main()
