"""Learning-curve and sweep plots."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_learning_curves(history: dict, title: str, path: str) -> None:
    """Plot training vs validation loss and AUC from a Keras History.history dict."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axes, ["loss", "auc"]):
        if key in history:
            ax.plot(history[key], label="train")
            ax.plot(history.get(f"val_{key}", []), label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(df, factor: str, metric: str, path: str) -> None:
    """Bar plot of a validation metric across the values of one swept factor."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(df[factor].astype(str), df[metric])
    ax.set_xlabel(factor)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs {factor}")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
