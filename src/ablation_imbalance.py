"""Class Imbalance Ablation Study.

Owner / Author: Esraa Gamal (@esraagamal11mohamed-cell)
Role: Role 2 - Data Engineer
Course: Neural Networks (AIE231)
Milestone: Week 10 Deliverable (Imbalance Ablations & Effect on Recall)

Specification Reference:
    Section 3.2: "Class imbalance: measure it, then address it with at least
    one technique (class weights, oversampling, or augmentation of minority classes)
    and show the effect on per-class recall."
    Section 3.5: "Log every run in one experiment table (CSV in the repository)..."
    Section 4: Role 2 owner "Runs the imbalance and augmentation ablations".

This script benchmarks three controlled regimes:
1. Regime A: Unweighted (Standard empirical risk minimization without imbalance correction).
2. Regime B: Balanced Class Weighting (Cost-sensitive reweighting: w_0 ~ 0.56, w_1 ~ 4.44).
3. Regime C: Minority Random Oversampling (Balanced 50:50 distribution on train fold).
"""

import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure src package is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import prepare_data, oversample_minority
from src.baselines import compute_metrics, evaluate_trivial_baseline


EXPERIMENT_LOG_PATH = "experiments/experiment_log.csv"
SUMMARY_CSV_PATH = "experiments/imbalance_ablation_summary.csv"


def log_experiment_run(run_data: dict, log_path: str = EXPERIMENT_LOG_PATH):
    """Append a structured experiment run to the repository experiment log CSV.
    
    Args:
        run_data: Dictionary containing run metadata, hyperparameters, and metrics.
        log_path: Path to the CSV log file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    df_new = pd.DataFrame([run_data])
    
    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    df_combined.to_csv(log_path, index=False)
    print(f"Logged run '{run_data.get('run_id')}' to '{log_path}'.")


def run_imbalance_ablations(random_seed: int = 42):
    """Execute systematic class imbalance ablation study on leak-free data."""
    print("=" * 80)
    print(" AIE231 - CLASS IMBALANCE ABLATION STUDY (Esraa Gamal - Role 2)")
    print("=" * 80)
    
    # 1. Prepare leak-free data splits
    data = prepare_data(
        data_path="data/bank-additional-full.csv",
        save_preprocessor_path="models/preprocessor.joblib",
        random_state=random_seed,
    )
    
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    class_weights = data["class_weights"]
    
    print("\n--- Imbalance Characteristics in Partitions ---")
    print(f"Train samples: {len(y_train):,} | Negative: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%) | Positive: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
    print(f"Val samples  : {len(y_val):,} | Negative: {(y_val==0).sum():,} ({(y_val==0).mean()*100:.2f}%) | Positive: {(y_val==1).sum():,} ({(y_val==1).mean()*100:.2f}%)")
    print(f"Test samples : {len(y_test):,} | Negative: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%) | Positive: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")
    print(f"Computed Balanced Weights: w(0)={class_weights[0]:.4f}, w(1)={class_weights[1]:.4f} (Ratio: {class_weights[1]/class_weights[0]:.2f}:1)\n")
    
    results = []
    
    # -------------------------------------------------------------
    # 0. Trivial Baseline (Majority Class)
    # -------------------------------------------------------------
    triv_metrics = evaluate_trivial_baseline(y_train, y_val)
    triv_run = {
        "run_id": "ablation_imb_00_trivial",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "author": "Esraa Gamal",
        "model": "Trivial Baseline (Always No)",
        "imbalance_strategy": "None (Majority Predictor)",
        "class_weight": "None",
        "random_seed": random_seed,
        "training_time_sec": 0.001,
        "train_samples": len(y_train),
        "val_accuracy": round(triv_metrics["accuracy"], 4),
        "val_balanced_accuracy": round(triv_metrics["balanced_accuracy"], 4),
        "val_neg_recall": 1.0000,
        "val_pos_recall": round(triv_metrics["recall"], 4),
        "val_pos_precision": round(triv_metrics["precision"], 4),
        "val_pos_f1": round(triv_metrics["f1"], 4),
        "val_macro_f1": round(triv_metrics["macro_f1"], 4),
        "val_roc_auc": round(triv_metrics["roc_auc"], 4),
        "val_pr_auc": round(triv_metrics["pr_auc"], 4),
    }
    results.append(triv_run)
    log_experiment_run(triv_run)
    
    # -------------------------------------------------------------
    # 1. Regime A: Unweighted (Standard ERM)
    # -------------------------------------------------------------
    print("\n[1/3] Training Regime A: Unweighted Baseline (class_weight=None)...")
    clf_unweighted = LogisticRegression(max_iter=1000, class_weight=None, random_state=random_seed, solver="lbfgs")
    t0 = time.time()
    clf_unweighted.fit(X_train, y_train)
    t_unweighted = time.time() - t0
    
    y_val_pred_a = clf_unweighted.predict(X_val)
    y_val_prob_a = clf_unweighted.predict_proba(X_val)[:, 1]
    m_a = compute_metrics(y_val, y_val_pred_a, y_val_prob_a)
    
    # Per-class recall
    neg_recall_a = float(m_a["confusion_matrix"][0][0] / (m_a["confusion_matrix"][0][0] + m_a["confusion_matrix"][0][1]))
    pos_recall_a = m_a["recall"]
    
    run_a = {
        "run_id": "ablation_imb_01_unweighted",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "author": "Esraa Gamal",
        "model": "Learned Baseline (Logistic Regression)",
        "imbalance_strategy": "Unweighted (Standard ERM)",
        "class_weight": "None",
        "random_seed": random_seed,
        "training_time_sec": round(t_unweighted, 4),
        "train_samples": len(y_train),
        "val_accuracy": round(m_a["accuracy"], 4),
        "val_balanced_accuracy": round(m_a["balanced_accuracy"], 4),
        "val_neg_recall": round(neg_recall_a, 4),
        "val_pos_recall": round(pos_recall_a, 4),
        "val_pos_precision": round(m_a["precision"], 4),
        "val_pos_f1": round(m_a["f1"], 4),
        "val_macro_f1": round(m_a["macro_f1"], 4),
        "val_roc_auc": round(m_a["roc_auc"], 4),
        "val_pr_auc": round(m_a["pr_auc"], 4),
    }
    results.append(run_a)
    log_experiment_run(run_a)
    
    # -------------------------------------------------------------
    # 2. Regime B: Balanced Class Weighting
    # -------------------------------------------------------------
    print("\n[2/3] Training Regime B: Balanced Class Weighting (Cost-Sensitive)...")
    clf_weighted = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed, solver="lbfgs")
    t0 = time.time()
    clf_weighted.fit(X_train, y_train)
    t_weighted = time.time() - t0
    
    y_val_pred_b = clf_weighted.predict(X_val)
    y_val_prob_b = clf_weighted.predict_proba(X_val)[:, 1]
    m_b = compute_metrics(y_val, y_val_pred_b, y_val_prob_b)
    
    neg_recall_b = float(m_b["confusion_matrix"][0][0] / (m_b["confusion_matrix"][0][0] + m_b["confusion_matrix"][0][1]))
    pos_recall_b = m_b["recall"]
    
    run_b = {
        "run_id": "ablation_imb_02_class_weights",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "author": "Esraa Gamal",
        "model": "Learned Baseline (Logistic Regression)",
        "imbalance_strategy": "Balanced Class Weights",
        "class_weight": f"{{0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}}}",
        "random_seed": random_seed,
        "training_time_sec": round(t_weighted, 4),
        "train_samples": len(y_train),
        "val_accuracy": round(m_b["accuracy"], 4),
        "val_balanced_accuracy": round(m_b["balanced_accuracy"], 4),
        "val_neg_recall": round(neg_recall_b, 4),
        "val_pos_recall": round(pos_recall_b, 4),
        "val_pos_precision": round(m_b["precision"], 4),
        "val_pos_f1": round(m_b["f1"], 4),
        "val_macro_f1": round(m_b["macro_f1"], 4),
        "val_roc_auc": round(m_b["roc_auc"], 4),
        "val_pr_auc": round(m_b["pr_auc"], 4),
    }
    results.append(run_b)
    log_experiment_run(run_b)
    
    # -------------------------------------------------------------
    # 3. Regime C: Minority Class Oversampling
    # -------------------------------------------------------------
    print("\n[3/3] Training Regime C: Minority Class Oversampling (Train fold only)...")
    X_train_res, y_train_res = oversample_minority(X_train, y_train, random_state=random_seed)
    clf_oversampled = LogisticRegression(max_iter=1000, class_weight=None, random_state=random_seed, solver="lbfgs")
    t0 = time.time()
    clf_oversampled.fit(X_train_res, y_train_res)
    t_oversampled = time.time() - t0
    
    y_val_pred_c = clf_oversampled.predict(X_val)
    y_val_prob_c = clf_oversampled.predict_proba(X_val)[:, 1]
    m_c = compute_metrics(y_val, y_val_pred_c, y_val_prob_c)
    
    neg_recall_c = float(m_c["confusion_matrix"][0][0] / (m_c["confusion_matrix"][0][0] + m_c["confusion_matrix"][0][1]))
    pos_recall_c = m_c["recall"]
    
    run_c = {
        "run_id": "ablation_imb_03_oversampling",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "author": "Esraa Gamal",
        "model": "Learned Baseline (Logistic Regression)",
        "imbalance_strategy": "Random Minority Oversampling (50:50)",
        "class_weight": "None (Balanced via Resampling)",
        "random_seed": random_seed,
        "training_time_sec": round(t_oversampled, 4),
        "train_samples": len(y_train_res),
        "val_accuracy": round(m_c["accuracy"], 4),
        "val_balanced_accuracy": round(m_c["balanced_accuracy"], 4),
        "val_neg_recall": round(neg_recall_c, 4),
        "val_pos_recall": round(pos_recall_c, 4),
        "val_pos_precision": round(m_c["precision"], 4),
        "val_pos_f1": round(m_c["f1"], 4),
        "val_macro_f1": round(m_c["macro_f1"], 4),
        "val_roc_auc": round(m_c["roc_auc"], 4),
        "val_pr_auc": round(m_c["pr_auc"], 4),
    }
    results.append(run_c)
    log_experiment_run(run_c)
    
    # -------------------------------------------------------------
    # Comparison & Ablation Findings
    # -------------------------------------------------------------
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    
    print("\n" + "=" * 90)
    print(" CLASS IMBALANCE ABLATION BENCHMARK RESULTS")
    print("=" * 90)
    cols_to_show = [
        "run_id",
        "imbalance_strategy",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_pos_recall",
        "val_pos_precision",
        "val_pos_f1",
        "val_roc_auc",
        "val_pr_auc",
    ]
    print(summary_df[cols_to_show].to_string(index=False))
    print("=" * 90)
    
    print("\n--- Key Engineering Findings from Imbalance Ablations ---")
    print(f"1. Naive Unweighted model attains high accuracy ({run_a['val_accuracy']:.2%}) but fails miserably on minority recall ({run_a['val_pos_recall']:.2%}).")
    print(f"2. Balanced Class Weighting surges positive recall to {run_b['val_pos_recall']:.2%} (+{run_b['val_pos_recall'] - run_a['val_pos_recall']:.2%}), boosting positive F1 to {run_b['val_pos_f1']:.4f}.")
    print(f"3. Minority Oversampling achieves comparable positive recall ({run_c['val_pos_recall']:.2%}) and positive F1 ({run_c['val_pos_f1']:.4f}) with balanced accuracy ({run_c['val_balanced_accuracy']:.2%}).")
    print("4. Conclusion: Cost-sensitive class weighting is computationally optimal (zero storage overhead) and directly applicable to Keras MLP training.\n")
    
    return summary_df


if __name__ == "__main__":
    run_imbalance_ablations(random_seed=42)
