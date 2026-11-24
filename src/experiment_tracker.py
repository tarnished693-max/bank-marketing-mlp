"""
Experiment Tracking Module
Author: Experimentation Engineer
Course: AIE231 Neural Networks Project - Alamein International University

This module implements:
1. `ExperimentTracker`: Centralized manager for logging runs to `experiments/experiments.csv`.
2. `ExperimentTrackingCallback`: A Keras Callback that tracks training progress,
   logs epoch metrics, measures wall-clock time, and updates `experiments.csv`.
"""

import os
import csv
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd


class ExperimentTracker:
    """
    Central experiment logger managing `experiments/experiments.csv`.
    Ensures complete traceability, prevents manual copy-paste errors,
    and adheres to the course specification for experimental reporting.
    """

    DEFAULT_CSV_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "experiments.csv",
    )

    EXPECTED_COLUMNS = [
        "run_id",
        "date",
        "author",
        "model_name",
        "learning_rate",
        "batch_size",
        "optimizer",
        "hidden_units",
        "activation",
        "dropout_rate",
        "use_batch_norm",
        "use_class_weights",
        "random_seed",
        "epochs_trained",
        "training_time_sec",
        "total_params",
        "trainable_params",
        "train_loss",
        "val_loss",
        "train_pr_auc",
        "val_pr_auc",
        "val_roc_auc",
        "val_f1_score",
        "val_recall",
        "val_precision",
        "notes",
        "curve_path",
    ]

    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path or self.DEFAULT_CSV_PATH
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        """Create the CSV file with the standard header if it does not exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.EXPECTED_COLUMNS)

    def log_run(self, run_data: Dict[str, Any]) -> str:
        """
        Appends a run dictionary to `experiments.csv`.
        Fills missing fields with sensible defaults or empty strings.
        
        Args:
            run_data: Dictionary containing run metadata and performance metrics.
            
        Returns:
            The recorded `run_id`.
        """
        run_id = run_data.get("run_id") or f"EXP-{int(time.time())}"
        date_str = run_data.get("date") or datetime.now().strftime("%Y-%m-%d")

        row = []
        for col in self.EXPECTED_COLUMNS:
            if col == "run_id":
                row.append(run_id)
            elif col == "date":
                row.append(date_str)
            else:
                val = run_data.get(col, "")
                if isinstance(val, float):
                    row.append(round(val, 4))
                else:
                    row.append(str(val))

        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"[EXPERIMENT TRACKER] Run '{run_id}' logged successfully to '{self.csv_path}'.")
        return run_id

    def load_table(self) -> pd.DataFrame:
        """Loads all logged experiments as a pandas DataFrame."""
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=self.EXPECTED_COLUMNS)
        return pd.read_csv(self.csv_path)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata and metrics for a specific run ID."""
        df = self.load_table()
        match = df[df["run_id"] == run_id]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def summarize_sweeps(self, factor_column: str) -> pd.DataFrame:
        """
        Summarizes metrics grouped by a specific hyperparameter factor (e.g. learning_rate).
        Useful for One-Factor-At-A-Time report tables.
        """
        df = self.load_table()
        if factor_column not in df.columns:
            raise ValueError(f"Column '{factor_column}' not found in experiments table.")
        summary_cols = [
            factor_column,
            "train_loss",
            "val_loss",
            "val_pr_auc",
            "val_roc_auc",
            "val_f1_score",
            "training_time_sec",
        ]
        available_cols = [c for c in summary_cols if c in df.columns]
        return df[available_cols].groupby(factor_column).mean()


# Keras Callback for automatic training lifecycle tracking
try:
    import tensorflow as tf

    class ExperimentTrackingCallback(tf.keras.callbacks.Callback):
        """
        Keras Callback that logs training metrics epoch-by-epoch, measures elapsed
        training time, records model parameters, and automatically updates `experiments.csv`.
        """

        def __init__(
            self,
            run_id: str,
            config: Dict[str, Any],
            tracker: Optional[ExperimentTracker] = None,
            plot_curves: bool = True,
            curves_dir: Optional[str] = None,
        ):
            super().__init__()
            self.run_id = run_id
            self.config = config
            self.tracker = tracker or ExperimentTracker()
            self.plot_curves = plot_curves
            self.curves_dir = curves_dir or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "experiments",
                "curves",
            )
            os.makedirs(self.curves_dir, exist_ok=True)

            self.start_time = 0.0
            self.history = {}
            self.epoch_times = []

        def on_train_begin(self, logs=None):
            self.start_time = time.time()
            self.history = {}
            print(f"[EXPERIMENT TRACKER] Starting experiment '{self.run_id}'...")

        def on_epoch_begin(self, epoch, logs=None):
            self._epoch_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch_times.append(time.time() - self._epoch_start)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(float(v))

        def on_train_end(self, logs=None):
            total_duration = time.time() - self.start_time
            epochs_trained = len(self.epoch_times)

            # Compute model parameters
            total_params = self.model.count_params()
            trainable_params = sum(
                tf.size(w).numpy() for w in self.model.trainable_weights
            )

            # Extract final or best metrics
            train_loss = self.history.get("loss", [-1.0])[-1]
            val_loss = self.history.get("val_loss", [-1.0])[-1]

            # Primary metric: PR-AUC (or AUC)
            pr_auc_key = next((k for k in self.history if "pr_auc" in k and not k.startswith("val_")), None)
            val_pr_auc_key = next((k for k in self.history if "pr_auc" in k and k.startswith("val_")), None)
            roc_auc_key = next((k for k in self.history if "auc" in k and k.startswith("val_") and "pr" not in k), None)

            train_pr_auc = self.history.get(pr_auc_key, [0.0])[-1] if pr_auc_key else 0.0
            val_pr_auc = self.history.get(val_pr_auc_key, [0.0])[-1] if val_pr_auc_key else 0.0
            val_roc_auc = self.history.get(roc_auc_key, [0.0])[-1] if roc_auc_key else 0.0

            # Save curves if requested
            curve_relative_path = ""
            if self.plot_curves and epochs_trained > 0:
                try:
                    from src.plot_curves import plot_training_curves
                    curve_filename = f"{self.run_id}_curves.png"
                    curve_full_path = os.path.join(self.curves_dir, curve_filename)
                    plot_training_curves(
                        history=self.history,
                        save_path=curve_full_path,
                        title=f"Learning Curves: {self.run_id} ({self.config.get('name', '')})",
                    )
                    curve_relative_path = os.path.join("experiments", "curves", curve_filename).replace("\\", "/")
                except Exception as e:
                    print(f"[EXPERIMENT TRACKER] Could not plot curves: {e}")

            # Assemble log dictionary
            arch = self.config.get("architecture", {})
            train_cfg = self.config.get("training", {})

            entry = {
                "run_id": self.run_id,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "author": self.config.get("author", "ExperimentationEngineer"),
                "model_name": arch.get("model_type", "MLP"),
                "learning_rate": train_cfg.get("learning_rate", 0.001),
                "batch_size": train_cfg.get("batch_size", 64),
                "optimizer": train_cfg.get("optimizer", "adam"),
                "hidden_units": json.dumps(arch.get("hidden_units", [128, 64])),
                "activation": arch.get("activation", "relu"),
                "dropout_rate": arch.get("dropout_rate", 0.0),
                "use_batch_norm": arch.get("use_batch_norm", False),
                "use_class_weights": train_cfg.get("use_class_weights", False),
                "random_seed": self.config.get("seed", 42),
                "epochs_trained": epochs_trained,
                "training_time_sec": round(total_duration, 2),
                "total_params": total_params,
                "trainable_params": trainable_params,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_pr_auc": train_pr_auc,
                "val_pr_auc": val_pr_auc,
                "val_roc_auc": val_roc_auc,
                "val_f1_score": self.history.get("val_f1_score", [0.0])[-1],
                "val_recall": self.history.get("val_recall", [0.0])[-1],
                "val_precision": self.history.get("val_precision", [0.0])[-1],
                "notes": self.config.get("notes", ""),
                "curve_path": curve_relative_path,
            }

            self.tracker.log_run(entry)
            print(
                f"[EXPERIMENT TRACKER] Completed '{self.run_id}' in {total_duration:.2f}s. "
                f"Val Loss: {val_loss:.4f}, Val PR-AUC: {val_pr_auc:.4f}"
            )

except ImportError:
    # Graceful fallback if TensorFlow is not yet installed in the current environment
    ExperimentTrackingCallback = None
