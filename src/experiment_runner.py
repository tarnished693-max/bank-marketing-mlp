"""
Modular Experiment Runner
Author: Experimentation Engineer
Course: AIE231 Neural Networks Project - Alamein International University

This script serves as the centralized runner for:
1. Running single experiment configurations from `configs/`.
2. Running the 3-value Learning Rate study (`--sweep lr`).
3. Running the 3-value Batch Size study (`--sweep batch_size`).
4. Running the 3-optimizer study (`--sweep optimizer`).
5. Running the 4 Ablation studies (`--sweep ablations`).
6. Running the 3-seed statistical robustness study (`--sweep multi_seed`).
"""

import os
import sys
import json
import argparse
import random
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Ensure src is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.experiment_tracker import ExperimentTracker, ExperimentTrackingCallback
from src.plot_curves import (
    plot_training_curves,
    plot_comparative_curves,
    plot_multi_seed_spread,
)


def set_seed(seed: int = 42):
    """Enforces strict deterministic behavior across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def load_data_or_mock(
    data_path: Optional[str] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads preprocessed tabular data, or produces mock tabular data matching
    the exact feature dimension of Bank Marketing (approx 48-52 encoded features)
    to enable end-to-end framework verification before the Data Engineer completes data.py.
    """
    if data_path and os.path.exists(data_path):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        print(f"[DATA] Loading real data from {data_path}...")
        df = pd.read_csv(data_path, sep=";")
        target_col = "y"
        y = (df[target_col].str.strip().str.lower() == "yes").astype(int).values
        X_df = df.drop(columns=[target_col])

        # Basic one-hot encoding for tabular pipeline
        X_encoded = pd.get_dummies(X_df, drop_first=True).astype(float).values

        # Stratified split: Train (70%), Val (15%), Test (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_encoded, y, test_size=(1.0 - train_ratio), random_state=seed, stratify=y
        )
        val_fraction_of_temp = val_ratio / (1.0 - train_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1.0 - val_fraction_of_temp), random_state=seed, stratify=y_temp
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test

    # Synthetic fallback matching Bank Marketing dimensions (~45,000 samples, ~48 features, 11.7% positives)
    print("[DATA] Using synthetic benchmark data matching Bank Marketing dimensions...")
    np.random.seed(seed)
    n_samples = 45211
    n_features = 48
    X = np.random.randn(n_samples, n_features)
    # 11.7% positive class imbalance
    y = (np.random.rand(n_samples) < 0.117).astype(int)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_mlp_model(input_dim: int, config: Dict[str, Any]):
    """
    Constructs an MLP model according to the experimental configuration.
    Fulfills course requirement: modular architecture, explicit activations, dropout, and optimizers.
    """
    import tensorflow as tf

    arch = config.get("architecture", {})
    hidden_units = arch.get("hidden_units", [128, 64])
    activation = arch.get("activation", "relu")
    dropout_rate = arch.get("dropout_rate", 0.3)
    use_batch_norm = arch.get("use_batch_norm", False)

    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 0.001)
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    momentum = train_cfg.get("momentum", 0.0)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for idx, units in enumerate(hidden_units):
        model.add(tf.keras.layers.Dense(units, use_bias=not use_batch_norm, name=f"dense_{idx+1}"))
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization(name=f"batch_norm_{idx+1}"))
        if activation == "leaky_relu":
            model.add(tf.keras.layers.LeakyReLU(negative_slope=0.1, name=f"leaky_relu_{idx+1}"))
        else:
            model.add(tf.keras.layers.Activation(activation, name=f"activation_{idx+1}"))
        if dropout_rate > 0.0:
            model.add(tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{idx+1}"))

    # Final binary output
    model.add(tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer"))

    # Select optimizer
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name in ["sgd", "sgd_momentum"]:
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Metrics: Primary metric is PR-AUC (Precision-Recall curve area)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)
    return model


def run_experiment(
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    csv_path: Optional[str] = None,
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a single experiment end-to-end:
      1. Sets random seed.
      2. Loads data (Train/Val/Test). Test set is strictly ignored.
      3. Builds and compiles model.
      4. Trains with early stopping and `ExperimentTrackingCallback`.
      5. Saves learning curves.
    """
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    seed = config.get("seed", 42)
    set_seed(seed)

    effective_run_id = run_id or config.get("run_id") or f"EXP-{int(time.time())}"
    print(f"\n{'='*70}\n[RUNNER] Launching Experiment: {effective_run_id}\n{'='*70}")

    # 1. Load Data
    X_train, y_train, X_val, y_val, _, _ = load_data_or_mock(
        data_path=data_path,
        seed=seed,
    )

    # 2. Build Model
    model = build_mlp_model(input_dim=X_train.shape[1], config=config)

    # 3. Class Weights
    class_weight_dict = None
    if config.get("training", {}).get("use_class_weights", False):
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        print(f"[RUNNER] Applied balanced class weights: {class_weight_dict}")

    # 4. Callbacks
    tracker = ExperimentTracker(csv_path=csv_path)
    tracking_callback = ExperimentTrackingCallback(
        run_id=effective_run_id,
        config=config,
        tracker=tracker,
        plot_curves=True,
    )

    patience = config.get("training", {}).get("early_stopping_patience", 10)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    batch_size = config.get("training", {}).get("batch_size", 64)
    epochs = config.get("training", {}).get("epochs", 40)

    # 5. Train Model (strictly on Train and Val sets!)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, tracking_callback],
        verbose=1,
    )

    return tracking_callback.history


def run_sweep(sweep_type: str, data_path: Optional[str] = None):
    """
    Executes a predefined hyperparameter study or ablation sweep,
    then automatically plots the comparative evaluation curves.
    """
    configs_dir = os.path.join(BASE_DIR, "configs")
    curves_dir = os.path.join(BASE_DIR, "experiments", "curves")
    os.makedirs(curves_dir, exist_ok=True)

    with open(os.path.join(configs_dir, "base_reference.json"), "r") as f:
        base_cfg = json.load(f)

    if sweep_type == "lr":
        config_file = os.path.join(configs_dir, "tuning_learning_rate.json")
        metric_title = "Learning Rate Sweep (1e-4 vs 1e-3 vs 1e-2)"
        save_plot_path = os.path.join(curves_dir, "comparison_learning_rate.png")
    elif sweep_type == "batch_size":
        config_file = os.path.join(configs_dir, "tuning_batch_size.json")
        metric_title = "Batch Size Sweep (32 vs 64 vs 128)"
        save_plot_path = os.path.join(curves_dir, "comparison_batch_size.png")
    elif sweep_type == "optimizer":
        config_file = os.path.join(configs_dir, "tuning_optimizer.json")
        metric_title = "Optimizer Study (Adam vs SGD vs SGD+Momentum)"
        save_plot_path = os.path.join(curves_dir, "comparison_optimizers.png")
    elif sweep_type == "ablations":
        print("\n[RUNNER] Running 4 Ablation Studies...")
        ablation_files = [
            ("ablation_dropout.json", "comparison_ablation_dropout.png", "Ablation: Dropout (0.0 vs 0.3)"),
            ("ablation_class_weights.json", "comparison_ablation_class_weights.png", "Ablation: Class Weights (None vs Balanced)"),
            ("ablation_activation.json", "comparison_ablation_activation.png", "Ablation: Activation (ReLU vs Tanh vs LeakyReLU)"),
            ("ablation_batch_norm.json", "comparison_ablation_batch_norm.png", "Ablation: Batch Normalization (Off vs On)"),
        ]
        for cfg_name, plot_name, title in ablation_files:
            with open(os.path.join(configs_dir, cfg_name), "r") as f:
                abl_data = json.load(f)
            histories = {}
            for run in abl_data["runs"]:
                merged_cfg = base_cfg.copy()
                merged_cfg["architecture"] = base_cfg["architecture"].copy()
                merged_cfg["training"] = base_cfg["training"].copy()
                for k, v in run.items():
                    if k in merged_cfg["architecture"]:
                        merged_cfg["architecture"][k] = v
                    elif k in merged_cfg["training"]:
                        merged_cfg["training"][k] = v
                    else:
                        merged_cfg[k] = v
                hist = run_experiment(merged_cfg, run_id=run["run_id"], data_path=data_path)
                histories[run["name"]] = hist

            plot_comparative_curves(
                runs_histories=histories,
                metric="val_loss",
                save_path=os.path.join(curves_dir, plot_name),
                title=title,
            )
        return

    elif sweep_type == "multi_seed":
        seed_cfg_file = os.path.join(configs_dir, "multi_seed.json")
        with open(seed_cfg_file, "r") as f:
            seed_data = json.load(f)
        histories = []
        for run in seed_data["runs"]:
            cfg = base_cfg.copy()
            cfg["seed"] = run["seed"]
            cfg["notes"] = run["notes"]
            hist = run_experiment(cfg, run_id=run["run_id"], data_path=data_path)
            histories.append(hist)

        plot_multi_seed_spread(
            seed_histories=histories,
            metric="val_pr_auc",
            save_path=os.path.join(curves_dir, "multi_seed_pr_auc_stability.png"),
            title="Validation PR-AUC Across 3 Random Seeds (42, 123, 999)",
        )
        return
    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    # Standard OFAT sweep execution
    with open(config_file, "r") as f:
        sweep_data = json.load(f)

    all_histories = {}
    for run in sweep_data["runs"]:
        merged_cfg = base_cfg.copy()
        merged_cfg["training"] = base_cfg["training"].copy()
        merged_cfg["architecture"] = base_cfg["architecture"].copy()
        for k, v in run.items():
            if k in merged_cfg["training"]:
                merged_cfg["training"][k] = v
            elif k in merged_cfg["architecture"]:
                merged_cfg["architecture"][k] = v
            else:
                merged_cfg[k] = v

        hist = run_experiment(merged_cfg, run_id=run["run_id"], data_path=data_path)
        all_histories[run["name"]] = hist

    # Generate comparative plot
    plot_comparative_curves(
        runs_histories=all_histories,
        metric="val_loss",
        save_path=save_plot_path,
        title=metric_title,
    )


def main():
    parser = argparse.ArgumentParser(description="Bank Marketing MLP Experiment Runner (AIE231)")
    parser.add_argument("--config", type=str, help="Path to experiment JSON configuration file")
    parser.add_argument("--run-id", type=str, help="Custom Run ID for logging")
    parser.add_argument(
        "--sweep",
        type=str,
        choices=["lr", "batch_size", "optimizer", "ablations", "multi_seed"],
        help="Run predefined hyperparameter sweep or ablation studies",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to bank-full.csv")

    args = parser.parse_args()

    # Default data path if exists
    default_csv = os.path.join(BASE_DIR, "data", "bank-full.csv")
    data_path = args.data_path or (default_csv if os.path.exists(default_csv) else None)

    if args.sweep:
        run_sweep(args.sweep, data_path=data_path)
    elif args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        run_experiment(cfg, run_id=args.run_id, data_path=data_path)
    else:
        # Default: Run base reference experiment
        ref_path = os.path.join(BASE_DIR, "configs", "base_reference.json")
        with open(ref_path, "r") as f:
            cfg = json.load(f)
        run_experiment(cfg, run_id="EXP-REF-DEMO", data_path=data_path)


if __name__ == "__main__":
    import time
    main()
