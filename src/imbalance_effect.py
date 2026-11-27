"""Measure how class weights / oversampling change per-class recall.

Uses a logistic-regression probe on the VALIDATION set (never the test set),
so the effect of each imbalance technique is visible before any neural net is
trained. Output: ``experiments/imbalance_logreg.csv``.

Run: ``python -m src.imbalance_effect``
"""
from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.data import (get_class_weights, load_config, oversample_minority,
                      prepare_data, resolve_path)


def run(seed: int = 42) -> pd.DataFrame:
    """Fit three probes (none / class weights / oversampling); return metrics."""
    cfg = load_config()
    d = prepare_data(cfg, save=False)
    variants = {
        "no_handling": (d.X_train, d.y_train, None),
        "class_weights": (d.X_train, d.y_train, get_class_weights(d.y_train)),
        "oversampling": (*oversample_minority(d.X_train, d.y_train,
                                              cfg["imbalance"]["oversample_seed"]), None),
    }
    rows = []
    for name, (X, y, cw) in variants.items():
        clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=seed)
        pred = clf.fit(X, y).predict(d.X_val)
        p, r, f, _ = precision_recall_fscore_support(d.y_val, pred, labels=[0, 1],
                                                     zero_division=0)
        rows.append({"variant": name, "recall_no": r[0], "recall_yes": r[1],
                     "precision_yes": p[1], "f1_yes": f[1],
                     "macro_f1": f1_score(d.y_val, pred, average="macro")})
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    table = run()
    out = resolve_path("experiments/imbalance_logreg.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
