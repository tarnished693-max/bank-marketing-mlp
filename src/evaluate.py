"""Evaluation utilities.

Framework-agnostic: every function works on ``y_true`` and predicted
probabilities, so the same code evaluates the baselines (scikit-learn) and
Models A/B/C (Keras).

Test-set discipline
-------------------
Thresholds and models are chosen on the VALIDATION split only. The test split
can only be scored through ``evaluate_split(..., split="test", final=True)``
and every such call is appended to ``experiments/test_access_log.txt`` so the
team can prove the test set was used once, at the end.

CLI
---
    python -m src.evaluate --model models/baseline_logreg.joblib --split val
    python -m src.evaluate --model models/final_mlp.keras --split test --final
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "experiments" / "eval"
TEST_LOG = ROOT / "experiments" / "test_access_log.txt"
CLASS_NAMES = ["no", "yes"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Full binary-classification metric set at a given decision threshold.

    Positive-class precision/recall/F1 are the primary metrics (the data is
    ~88.7 % "no"); accuracy is reported only for completeness.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_yes": precision_score(y_true, y_pred, zero_division=0),
        "recall_yes": recall_score(y_true, y_pred, zero_division=0),
        "f1_yes": f1_score(y_true, y_pred, zero_division=0),
        "recall_no": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_threshold(y_val, p_val, metric: str = "f1") -> float:
    """Pick the decision threshold that maximises F1 (or F2) on VALIDATION.

    Class-weighted training shifts probabilities, so 0.5 is rarely optimal.
    ``metric`` is ``"f1"`` or ``"f2"`` (recall-weighted, matching the cost
    argument that a missed subscriber is dearer than a wasted call).
    """
    beta = {"f1": 1.0, "f2": 2.0}[metric]
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta**2) * prec * rec / np.maximum(beta**2 * prec + rec, 1e-12)
    return float(thr[int(np.argmax(f))])


def per_class_report(y_true, y_prob, threshold: float = 0.5) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support as a DataFrame."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    rep = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return pd.DataFrame(rep).T.round(4)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_prob, threshold, title, path: Optional[Path] = None):
    """Counts + row-normalised confusion matrix side by side."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, data, fmt, sub in zip(axes, (cm, cmn), ("d", ".2f"), ("Counts", "Row-normalised")):
        ax.imshow(data, cmap="Blues")
        ax.set_xticks([0, 1], CLASS_NAMES)
        ax.set_yticks([0, 1], CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(sub)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
    fig.suptitle(f"{title} (threshold={threshold:.2f})")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(y_true, probs: Dict[str, np.ndarray], title: str, path: Optional[Path] = None):
    """Overlay ROC and precision-recall curves for several models."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        a2.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, p):.3f})")
    a1.plot([0, 1], [0, 1], "k--", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a2.axhline(np.mean(y_true), color="k", ls="--", lw=0.8, label="prevalence")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def categorise_error(row: pd.Series, threshold: float) -> str:
    """Assign one failure category to a misclassified raw-feature row.

    Rules are deliberately simple and explainable in the defense; they are
    evaluated in order and the first match wins.
    """
    p, y = row["p_yes"], row["y_true"]
    if abs(p - threshold) < 0.10:
        return "borderline (score within 0.10 of threshold)"
    if y == 1:  # false negative
        if row.get("pdays", -1) in (-1, 999) and row.get("poutcome") in ("unknown", "nonexistent"):
            return "FN: first-time contact, no history to signal interest"
        if row.get("balance", 0) < 0:
            return "FN: subscribed despite negative account balance (atypical)"
        return "FN: other (rare positive profile)"
    # false positive
    if row.get("poutcome") == "failure":
        return "FP: previous campaign failed but profile looks like a subscriber"
    if row.get("poutcome") == "success":
        return "FP: previous success but declined this time"
    return "FP: other (profile resembles subscribers)"


def all_errors(raw_df: pd.DataFrame, y_true, y_prob, threshold: float) -> pd.DataFrame:
    """Every misclassified row with probability, error type and category."""
    df = raw_df.reset_index(drop=True).copy()
    df["y_true"] = np.asarray(y_true).astype(int)
    df["p_yes"] = np.asarray(y_prob, dtype=float)
    df["y_pred"] = (df["p_yes"] >= threshold).astype(int)
    df["error_size"] = (df["p_yes"] - df["y_true"]).abs()
    wrong = df[df["y_pred"] != df["y_true"]].copy()
    wrong["error_type"] = np.where(wrong["y_true"] == 1, "false_negative", "false_positive")
    wrong["category"] = wrong.apply(categorise_error, axis=1, threshold=threshold)
    return wrong


def error_analysis(raw_df: pd.DataFrame, y_true, y_prob, threshold: float,
                   n: int = 20) -> pd.DataFrame:
    """Return ``n`` most confidently wrong samples, split ~evenly FN / FP.

    ``raw_df`` must be the UNSCALED rows (same order as ``y_true``) so the
    failures can be read by a human.
    """
    wrong = all_errors(raw_df, y_true, y_prob, threshold)
    fn_all = wrong[wrong.error_type == "false_negative"].sort_values("error_size", ascending=False)
    fp_all = wrong[wrong.error_type == "false_positive"].sort_values("error_size", ascending=False)
    k_fn = min(len(fn_all), n - min(len(fp_all), n // 2))
    k_fp = min(len(fp_all), n - k_fn)
    return pd.concat([fn_all.head(k_fn), fp_all.head(k_fp)]).sort_values("error_size", ascending=False)


def category_summary(wrong_all: pd.DataFrame) -> pd.DataFrame:
    """Count of all errors per failure category (for the report table)."""
    return (wrong_all.groupby(["error_type", "category"]).size()
            .rename("count").reset_index().sort_values("count", ascending=False))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_models(y_true, probs: Dict[str, np.ndarray], thresholds: Dict[str, float],
                    tag: str, out_dir: Path = EVAL_DIR) -> pd.DataFrame:
    """Metric table + confusion matrices + ROC/PR overlay for several models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, p in probs.items():
        thr = thresholds.get(name, 0.5)
        rows.append({"model": name, **compute_metrics(y_true, p, thr)})
        plot_confusion_matrix(y_true, p, thr, f"{name} [{tag}]",
                              out_dir / f"cm_{tag}_{name}.png")
    plot_roc_pr(y_true, probs, f"ROC / PR [{tag}]", out_dir / f"roc_pr_{tag}.png")
    table = pd.DataFrame(rows).round(4)
    table.to_csv(out_dir / f"metrics_{tag}.csv", index=False)
    return table


def log_test_access(model_name: str) -> None:
    """Append an audit line each time the test split is scored."""
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    with TEST_LOG.open("a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{model_name}\n")


def predict_proba_any(model, X) -> np.ndarray:
    """Positive-class probability from a Keras model or an sklearn estimator."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X, verbose=0)).ravel()


def load_model(path: Path):
    """Load a ``.keras`` / ``.h5`` Keras model or a ``.joblib`` sklearn model."""
    path = Path(path)
    if path.suffix == ".joblib":
        import joblib
        return joblib.load(path)
    import tensorflow as tf  # imported lazily: not needed for sklearn baselines
    return tf.keras.models.load_model(path)


def evaluate_split(model_path: Path, load_split: Callable[[str], tuple],
                   split: str = "val", final: bool = False,
                   threshold: Optional[float] = None) -> pd.DataFrame:
    """Evaluate one saved model on ``split``.

    ``load_split(name)`` must return ``(X_processed, y, raw_df)`` for the named
    split ("train" | "val" | "test"). This is the single integration point with
    ``src/data.py`` (see ``_default_load_split``).
    """
    if split == "test":
        if not final:
            raise PermissionError("Test split is locked. Pass final=True (CLI: --final) "
                                  "only for the frozen final model.")
        log_test_access(Path(model_path).name)
    model = load_model(model_path)
    name = Path(model_path).stem
    Xv, yv, _ = load_split("val")
    thr = threshold if threshold is not None else best_threshold(yv, predict_proba_any(model, Xv))
    X, y, raw = load_split(split)
    p = predict_proba_any(model, X)
    table = evaluate_models(y, {name: p}, {name: thr}, tag=f"{split}_{name}")
    print(table.T)
    print(per_class_report(y, p, thr))
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    error_analysis(raw, y, p, thr, n=20).to_csv(EVAL_DIR / f"errors_{split}_{name}.csv", index=False)
    category_summary(all_errors(raw, y, p, thr)).to_csv(
        EVAL_DIR / f"error_categories_{split}_{name}.csv", index=False)
    return table


def _default_load_split(name: str):
    """ADAPTER to the team's ``src/data.py``.

    Returns ``(X_processed, y, raw_df_unscaled)`` for ``"train"``, ``"val"``,
    or ``"test"``.
    """
    if name not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {name!r}; expected 'train', 'val', or 'test'.")
    from src import data as data_mod

    cfg = data_mod.load_config()
    raw = data_mod.load_raw(cfg)
    df, _ = (
        data_mod.drop_duplicate_rows(raw)
        if cfg["cleaning"]["drop_duplicates"]
        else (raw, 0)
    )
    splits = data_mod.split_data(df, cfg)
    raw_df = getattr(splits, f"X_{name}")
    y = getattr(splits, f"y_{name}").to_numpy()
    pre_path = data_mod.resolve_path(cfg["paths"]["preprocessor"])
    if pre_path.exists():
        preprocessor = data_mod.load_preprocessor(cfg)
    else:
        preprocessor = data_mod.prepare_data(cfg, save=False).preprocessor
    X_processed = data_mod.preprocess_raw(raw_df, preprocessor, cfg)
    return X_processed, y, raw_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--final", action="store_true", help="unlock the test split")
    ap.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()
    evaluate_split(a.model, _default_load_split, a.split, a.final, a.threshold)
