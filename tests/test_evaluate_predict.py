"""Unit tests for src/evaluate.py and src/predict.py (synthetic data, no TensorFlow)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import evaluate as ev
from src.predict import Predictor

NUM = ["age", "campaign", "pdays", "previous", "emp.var.rate"]
CAT = ["job", "poutcome"]


def make_df(n=2000, seed=0):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": r.integers(18, 90, n), "campaign": r.integers(1, 20, n),
        "pdays": r.choice([999, 3, 6], n), "previous": r.integers(0, 4, n),
        "emp.var.rate": r.normal(0, 1.5, n),
        "job": r.choice(["admin.", "services", "retired"], n),
        "poutcome": r.choice(["nonexistent", "failure", "success"], n),
        "duration": r.integers(5, 900, n),
    })
    logit = -2.2 + 1.5 * (df.poutcome == "success") - 0.4 * df["emp.var.rate"]
    df["y"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


@pytest.fixture()
def fitted(tmp_path):
    df = make_df()
    pre = ColumnTransformer([("n", StandardScaler(), NUM),
                             ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    X = pre.fit_transform(df[NUM + CAT])
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(X, df.y)
    joblib.dump(pre, tmp_path / "pre.joblib")
    joblib.dump(clf, tmp_path / "m.joblib")
    return df, pre, clf, tmp_path


def test_metrics_perfect_and_keys():
    y = np.array([0, 0, 1, 1])
    m = ev.compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["f1_yes"] == 1 and m["roc_auc"] == 1 and m["fn"] == 0 and m["fp"] == 0


def test_majority_baseline_has_zero_recall():
    y = np.array([0] * 90 + [1] * 10)
    m = ev.compute_metrics(y, np.zeros(100) + np.linspace(0, 0.01, 100))
    assert m["accuracy"] == 0.9 and m["recall_yes"] == 0


def test_best_threshold_beats_default_on_skewed_scores():
    r = np.random.default_rng(1)
    y = (r.random(3000) < 0.1).astype(int)
    p = np.clip(0.05 + 0.2 * y + r.normal(0, 0.05, 3000), 0, 1)  # scores never reach 0.5
    thr = ev.best_threshold(y, p)
    assert ev.compute_metrics(y, p, thr)["f1_yes"] > ev.compute_metrics(y, p, 0.5)["f1_yes"]


def test_error_analysis_returns_20_with_categories():
    df = make_df(3000)
    p = np.random.default_rng(2).random(3000)
    out = ev.error_analysis(df, df.y, p, 0.5, n=20)
    assert len(out) == 20 and out["category"].notna().all()
    assert set(out.error_type) == {"false_negative", "false_positive"}


def test_test_split_is_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "TEST_LOG", tmp_path / "log.txt")
    with pytest.raises(PermissionError):
        ev.evaluate_split(tmp_path / "m.joblib", lambda s: None, split="test", final=False)
    assert not (tmp_path / "log.txt").exists()


def test_predict_raw_row_drops_leaky_and_returns_confidence(fitted, tmp_path):
    df, _, _, d = fitted
    pr = Predictor(d / "m.joblib", d / "pre.joblib", threshold_file=d / "none.json")
    row = df.iloc[0].drop("y").to_dict()  # still contains 'duration'
    res = pr.predict(row)[0]
    assert res["prediction"] in {"yes", "no"} and 0.5 <= res["confidence"] <= 1
    assert pr.predict(df.drop(columns="y").head(5)).__len__() == 5


def test_predict_matches_training_pipeline(fitted):
    df, pre, clf, d = fitted
    pr = Predictor(d / "m.joblib", d / "pre.joblib", threshold_file=d / "none.json")
    got = [r["p_subscribe"] for r in pr.predict(df.head(50))]
    want = clf.predict_proba(pre.transform(df.head(50)[NUM + CAT]))[:, 1]
    np.testing.assert_allclose(got, want, atol=1e-4)  # no train/inference skew


def test_predict_missing_field_raises(fitted):
    df, _, _, d = fitted
    pr = Predictor(d / "m.joblib", d / "pre.joblib", threshold_file=d / "none.json")
    with pytest.raises(ValueError):
        pr.predict(df.head(1).drop(columns=["age"]))


def test_threshold_file_is_used(fitted):
    _, _, _, d = fitted
    (d / "t.json").write_text(json.dumps({"threshold": 0.3}))
    assert Predictor(d / "m.joblib", d / "pre.joblib", threshold_file=d / "t.json").threshold == 0.3
