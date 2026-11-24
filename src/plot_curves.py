"""
Learning Curves and Experiment Visualization Module
Author: Experimentation Engineer
Course: AIE231 Neural Networks Project - Alamein International University

This module generates publication-grade visualizations for:
1. Per-run training vs validation curves (Loss, PR-AUC, ROC-AUC) with overfitting/underfitting diagnosis.
2. Comparative hyperparameter curves (comparing Learning Rates, Batch Sizes, and Optimizers).
3. Multi-seed stability plots with mean and shaded standard deviation (+/- 1 std).
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training and Validation Curves",
    primary_metric: str = "pr_auc",
) -> plt.Figure:
    """
    Plots two-panel learning curves:
      - Left: Loss (Train vs Validation) with minimum validation loss epoch marked.
      - Right: Primary Metric (Train vs Validation) with best validation epoch marked.
    Automatically detects and labels signs of overfitting or underfitting.

    Args:
        history: Dictionary containing per-epoch metric lists (from Keras history.history).
        save_path: Path to save the resulting high-res figure (.png).
        title: Main title of the figure.
        primary_metric: Name of primary evaluation metric (e.g. 'pr_auc', 'auc', 'accuracy').

    Returns:
        The matplotlib Figure object.
    """
    epochs = range(1, len(history.get("loss", [])) + 1)
    if not epochs:
        print("[WARNING] History is empty. Cannot plot learning curves.")
        return plt.figure()

    # Style configuration
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    # ---- 1. Loss Panel ----
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    if val_loss:
        ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2)
        best_val_loss_epoch = int(np.argmin(val_loss)) + 1
        min_val_loss = min(val_loss)
        ax1.axvline(x=best_val_loss_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best Val Epoch ({best_val_loss_epoch})")
        ax1.scatter([best_val_loss_epoch], [min_val_loss], color="red", s=50, zorder=5)

        # Diagnose Overfitting / Underfitting
        final_val_loss = val_loss[-1]
        loss_gap = final_val_loss - train_loss[-1]
        if loss_gap > 0.10 and final_val_loss > min_val_loss * 1.15:
            diagnosis = "Overfitting detected (Val Loss rising)"
            ax1.text(0.05, 0.05, diagnosis, transform=ax1.transAxes, fontsize=10, color="darkred",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffdddd", ec="red", lw=1))
        elif min_val_loss > 0.45:
            diagnosis = "Possible Underfitting (High loss plateau)"
            ax1.text(0.05, 0.05, diagnosis, transform=ax1.transAxes, fontsize=10, color="darkblue",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ddeeff", ec="blue", lw=1))
        else:
            diagnosis = "Good generalization / stable convergence"
            ax1.text(0.05, 0.05, diagnosis, transform=ax1.transAxes, fontsize=10, color="darkgreen",
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ddffdd", ec="green", lw=1))

    ax1.set_title("Loss Trajectory (Binary Cross-Entropy)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # ---- 2. Primary Metric Panel ----
    train_metric_key = next((k for k in history if primary_metric in k and not k.startswith("val_")), None)
    val_metric_key = next((k for k in history if primary_metric in k and k.startswith("val_")), None)

    if not train_metric_key and "accuracy" in history:
        train_metric_key = "accuracy"
        val_metric_key = "val_accuracy"

    if train_metric_key and train_metric_key in history:
        train_metric = history[train_metric_key]
        ax2.plot(epochs, train_metric, label=f"Train {train_metric_key.upper()}", color="#2ca02c", linewidth=2)
    if val_metric_key and val_metric_key in history:
        val_metric = history[val_metric_key]
        ax2.plot(epochs, val_metric, label=f"Val {val_metric_key.upper()}", color="#d62728", linewidth=2)
        best_metric_epoch = int(np.argmax(val_metric)) + 1
        max_val_metric = max(val_metric)
        ax2.axvline(x=best_metric_epoch, color="purple", linestyle="--", alpha=0.7, label=f"Max Val ({max_val_metric:.4f})")
        ax2.scatter([best_metric_epoch], [max_val_metric], color="purple", s=50, zorder=5)

    ax2.set_title(f"Primary Metric Trajectory ({primary_metric.upper()})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel(primary_metric.upper(), fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[PLOT] Learning curves saved to: {save_path}")

    return fig


def plot_comparative_curves(
    runs_histories: Dict[str, Dict[str, List[float]]],
    metric: str = "val_loss",
    save_path: Optional[str] = None,
    title: str = "Hyperparameter Comparison",
    ylabel: Optional[str] = None,
) -> plt.Figure:
    """
    Overlays multiple experiment runs on a single plot.
    Crucial for:
      - Learning Rate comparison (1e-4 vs 1e-3 vs 1e-2)
      - Batch Size comparison (32 vs 64 vs 128)
      - Optimizer comparison (Adam vs SGD vs SGD-Momentum)
      - Ablation studies (e.g. With Dropout vs Without Dropout)
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (label, hist) in enumerate(runs_histories.items()):
        vals = hist.get(metric, [])
        if vals:
            color = palette[idx % len(palette)]
            epochs = range(1, len(vals) + 1)
            ax.plot(epochs, vals, label=label, linewidth=2, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[PLOT] Comparative curves saved to: {save_path}")

    return fig


def plot_multi_seed_spread(
    seed_histories: List[Dict[str, List[float]]],
    metric: str = "val_pr_auc",
    save_path: Optional[str] = None,
    title: str = "Validation Stability Across 3 Random Seeds (Mean +/- Std)",
) -> plt.Figure:
    """
    Plots the mean curve with shaded +/- 1 standard deviation across multiple seeds.
    Fulfills course requirement: "Run final configuration with at least 3 seeds and report mean +/- std".
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Pad or truncate to the minimum common length
    min_len = min(len(h[metric]) for h in seed_histories if metric in h)
    all_runs = np.array([h[metric][:min_len] for h in seed_histories if metric in h])

    mean_curve = np.mean(all_runs, axis=0)
    std_curve = np.std(all_runs, axis=0)
    epochs = np.arange(1, min_len + 1)

    ax.plot(epochs, mean_curve, color="#1f77b4", linewidth=2.5, label="Mean Validation Score")
    ax.fill_between(
        epochs,
        mean_curve - std_curve,
        mean_curve + std_curve,
        color="#1f77b4",
        alpha=0.25,
        label=r"$\pm 1\ \sigma$ Standard Deviation",
    )

    final_mean = mean_curve[-1]
    final_std = std_curve[-1]
    annotation_text = f"Final Val Score: {final_mean:.4f} \u00b1 {final_std:.4f}"
    ax.text(
        0.05, 0.90, annotation_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#1f77b4", lw=1.5),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"[PLOT] Multi-seed stability plot saved to: {save_path}")

    return fig
