"""Baseline models and benchmark evaluation for the Bank Marketing task.

Part of Role 3 (Baseline and Analysis Engineer) deliverables:
1. Trivial baseline: Majority-class prediction.
2. Simple learned baseline: Logistic Regression (scikit-learn).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute standard classification metrics for imbalanced tabular benchmark.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted positive-class probabilities (optional).
        
    Returns:
        dict: Computed metrics dictionary.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(np.mean(y_true))
        
    return metrics


def evaluate_trivial_baseline(y_train, y_eval):
    """Evaluate trivial baseline: always predict the majority class of the training set.
    
    Args:
        y_train: Training labels (to identify majority class).
        y_eval: Evaluation labels (validation or test).
        
    Returns:
        dict: Metrics for trivial baseline.
    """
    majority_class = int(np.round(np.mean(y_train)))  # 0 for imbalanced "no"
    y_pred = np.full_like(y_eval, fill_value=majority_class)
    y_prob = np.full(shape=(len(y_eval),), fill_value=float(np.mean(y_train)))
    
    metrics = compute_metrics(y_eval, y_pred, y_prob)
    metrics["model_name"] = "Trivial Baseline (Majority Class)"
    metrics["majority_class"] = majority_class
    return metrics


def evaluate_logistic_baseline(X_train, y_train, X_eval, y_eval, class_weight="balanced", random_state=42):
    """Train and evaluate the simple learned baseline: Logistic Regression.
    
    Args:
        X_train: Preprocessed feature matrix for training.
        y_train: Training labels.
        X_eval: Preprocessed feature matrix for evaluation.
        y_eval: Evaluation labels.
        class_weight: 'balanced' or None (to address class imbalance).
        random_state: Reproducibility seed.
        
    Returns:
        tuple: (fitted_model, metrics_dict)
    """
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs"
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_eval)
    y_prob = clf.predict_proba(X_eval)[:, 1]
    
    metrics = compute_metrics(y_eval, y_pred, y_prob)
    metrics["model_name"] = f"Logistic Regression (class_weight={class_weight})"
    return clf, metrics


def print_metrics_summary(title, metrics):
    """Print formatted metrics summary to terminal."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Positive Recall   : {metrics['recall']:.4f}  <-- Primary target")
    print(f"Positive Precision: {metrics['precision']:.4f}")
    print(f"Positive F1-Score : {metrics['f1']:.4f}      <-- Primary metric")
    print(f"Macro F1-Score    : {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC           : {metrics.get('roc_auc', 0.0):.4f}")
    print(f"PR-AUC (Avg Prec) : {metrics.get('pr_auc', 0.0):.4f}")
    print(f"Confusion Matrix  : TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}")
    print(f"                    FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}")
    print(f"{'=' * 60}\n")
