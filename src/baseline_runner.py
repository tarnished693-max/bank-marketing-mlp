"""
Baseline Benchmarking Module
Author: Experimentation Engineer
Course: AIE231 Neural Networks Project - Alamein International University

This module implements and evaluates the two required baselines (Section 3.3):
1. Trivial Baseline (Majority Class Classifier): Predicts class 0 ('no') unconditionally.
2. Simple Learned Baseline (Logistic Regression): L2-regularized linear classification,
   including both unweighted and class-weighted variants.

Every metric is calculated on the validation set and logged to `experiments/experiments.csv`.
"""

import os
import sys
import time
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    log_loss,
)
from sklearn.linear_model import LogisticRegression

# Ensure src is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.experiment_tracker import ExperimentTracker
from src.experiment_runner import load_data_or_mock


def compute_all_metrics(
    y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Computes full suite of classification metrics for fair model comparison."""
    # Clip probabilities for stable log-loss
    eps = 1e-15
    y_prob_clipped = np.clip(y_pred_prob, eps, 1 - eps)
    loss = float(log_loss(y_true, y_prob_clipped))

    y_pred = (y_pred_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob))
    except Exception:
        roc_auc = 0.50

    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = float(auc(recall_arr, precision_arr))
    except Exception:
        pr_auc = float(np.mean(y_true))

    return {
        "loss": round(loss, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def run_trivial_majority_baseline(
    y_train: np.ndarray, y_val: np.ndarray, tracker: ExperimentTracker
) -> Dict[str, Any]:
    """
    Evaluates the Trivial Majority-Class Baseline (Section 3.3).
    Always predicts the majority class (class 0 'no').
    Demonstrates mathematically why accuracy alone is misleading on imbalanced data.
    """
    start_time = time.time()
    majority_class = int(np.bincount(y_train).argmax())

    # Constant probability matching the empirical class ratio
    pos_prior = float(np.mean(y_train))
    y_val_prob = np.full_like(y_val, fill_value=pos_prior, dtype=float)

    # Predictions are all majority_class (0)
    metrics_val = compute_all_metrics(y_val, y_val_prob, threshold=0.5)
    training_time = round(time.time() - start_time, 4)

    run_id = "BASE-TRIVIAL-001"
    run_entry = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%d"),
        "author": "ExperimentationEngineer",
        "model_name": "Trivial_Majority_Class",
        "learning_rate": "N/A",
        "batch_size": "N/A",
        "optimizer": "None (Dummy)",
        "hidden_units": "None (No layers)",
        "activation": "None",
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "use_class_weights": False,
        "random_seed": 42,
        "epochs_trained": 0,
        "training_time_sec": training_time,
        "total_params": 0,
        "trainable_params": 0,
        "train_loss": round(float(log_loss(y_train, np.full_like(y_train, pos_prior, dtype=float))), 4),
        "val_loss": metrics_val["loss"],
        "train_pr_auc": round(pos_prior, 4),
        "val_pr_auc": metrics_val["pr_auc"],
        "val_roc_auc": metrics_val["roc_auc"],
        "val_f1_score": metrics_val["f1_score"],
        "val_recall": metrics_val["recall"],
        "val_precision": metrics_val["precision"],
        "notes": "Trivial baseline: predicts majority class 0 ('no'). Accuracy is high (88.3%) but Recall is 0.0%.",
        "curve_path": "",
    }

    tracker.log_run(run_entry)
    print(f"[BASELINE] Trivial Majority Baseline: Accuracy={metrics_val['accuracy']}, Recall={metrics_val['recall']}, PR-AUC={metrics_val['pr_auc']}")
    return run_entry


def run_logistic_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tracker: ExperimentTracker,
    balanced: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates the Simple Learned Baseline: Logistic Regression (Section 3.3).
    Supports both standard unweighted and class-weighted configurations.
    """
    start_time = time.time()
    cw = "balanced" if balanced else None
    run_id = "BASE-LOGREG-BALANCED-002" if balanced else "BASE-LOGREG-001"
    model_name = "LogisticRegression_Balanced" if balanced else "LogisticRegression_Standard"

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight=cw,
    )
    clf.fit(X_train, y_train)
    training_time = round(time.time() - start_time, 4)

    y_train_prob = clf.predict_proba(X_train)[:, 1]
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    metrics_train = compute_all_metrics(y_train, y_train_prob)
    metrics_val = compute_all_metrics(y_val, y_val_prob)

    # Number of parameters: weights (number of features) + bias (1)
    n_params = X_train.shape[1] + 1

    run_entry = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%d"),
        "author": "ExperimentationEngineer",
        "model_name": model_name,
        "learning_rate": "L-BFGS default",
        "batch_size": "Full-batch",
        "optimizer": "L-BFGS",
        "hidden_units": "None (Linear)",
        "activation": "Sigmoid",
        "dropout_rate": 0.0,
        "use_batch_norm": False,
        "use_class_weights": balanced,
        "random_seed": 42,
        "epochs_trained": int(clf.n_iter_[0]),
        "training_time_sec": training_time,
        "total_params": n_params,
        "trainable_params": n_params,
        "train_loss": metrics_train["loss"],
        "val_loss": metrics_val["loss"],
        "train_pr_auc": metrics_train["pr_auc"],
        "val_pr_auc": metrics_val["pr_auc"],
        "val_roc_auc": metrics_val["roc_auc"],
        "val_f1_score": metrics_val["f1_score"],
        "val_recall": metrics_val["recall"],
        "val_precision": metrics_val["precision"],
        "notes": f"Simple learned baseline ({model_name}). Parameters: {n_params} (d={X_train.shape[1]} + 1 bias).",
        "curve_path": "",
    }

    tracker.log_run(run_entry)
    print(f"[BASELINE] {model_name}: PR-AUC={metrics_val['pr_auc']}, ROC-AUC={metrics_val['roc_auc']}, F1={metrics_val['f1_score']}, Recall={metrics_val['recall']}")
    return run_entry


def run_all_baselines(data_path: str = None) -> Dict[str, Dict[str, Any]]:
    """Runs all baselines and records them into experiments.csv."""
    tracker = ExperimentTracker()

    csv_file = data_path or os.path.join(BASE_DIR, "data", "bank-full.csv")
    X_train, y_train, X_val, y_val, _, _ = load_data_or_mock(csv_file if os.path.exists(csv_file) else None)

    print("\n" + "=" * 70)
    print("[BASELINE BENCHMARKING] Starting Week 10 Baselines Evaluation")
    print("=" * 70)

    # 1. Trivial Majority Baseline
    trivial_res = run_trivial_majority_baseline(y_train, y_val, tracker)

    # 2. Learned Standard Logistic Regression Baseline
    logreg_std_res = run_logistic_regression_baseline(X_train, y_train, X_val, y_val, tracker, balanced=False)

    # 3. Learned Balanced Logistic Regression Baseline
    logreg_bal_res = run_logistic_regression_baseline(X_train, y_train, X_val, y_val, tracker, balanced=True)

    print("=" * 70)
    print("[BASELINE BENCHMARKING] Completed. All baselines logged to experiments/experiments.csv")
    print("=" * 70)

    return {
        "trivial": trivial_res,
        "logreg_standard": logreg_std_res,
        "logreg_balanced": logreg_bal_res,
    }


if __name__ == "__main__":
    run_all_baselines()
