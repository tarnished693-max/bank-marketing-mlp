"""Unit tests for src/data.py (leakage, cleaning, imbalance)."""
import numpy as np
import pandas as pd
import pytest

from src import data as D


@pytest.fixture(scope="module")
def cfg():
    return D.load_config()


@pytest.fixture(scope="module")
def prepared(cfg):
    return D.prepare_data(cfg, save=False)


def test_split_sizes_and_stratification(prepared):
    n = sum(len(x) for x in (prepared.X_train, prepared.X_val, prepared.X_test))
    assert abs(len(prepared.X_train) / n - 0.70) < 0.01
    rates = [y.mean() for y in (prepared.y_train, prepared.y_val, prepared.y_test)]
    assert max(rates) - min(rates) < 0.005


def test_leakage_checks_pass(prepared):
    assert prepared.report["leakage_checks"]["scaler_fitted_on_train_only"]


def test_train_scaled_numeric_columns_are_standardised(prepared):
    i = prepared.feature_names.index("age")
    assert abs(prepared.X_train[:, i].mean()) < 1e-3
    assert abs(prepared.X_train[:, i].std() - 1) < 1e-2


def test_no_nan_and_same_width(prepared):
    for X in (prepared.X_train, prepared.X_val, prepared.X_test):
        assert np.isfinite(X).all() and X.shape[1] == len(prepared.feature_names)


def test_duration_and_target_removed(cfg):
    raw = pd.DataFrame({"age": [30], "duration": [100], "pdays": [-1], "balance": [-5],
                        "campaign": [1], "previous": [0], "y": ["no"]})
    out = D.clean_features(raw, cfg)
    assert "duration" not in out and "y" not in out
    assert out.loc[0, "pdays"] == 0 and out.loc[0, D.PREVIOUSLY_CONTACTED] == 0


def test_inference_path_matches_training_path(cfg, prepared):
    raw = D.load_raw(cfg).iloc[:5]
    again = D.preprocess_raw(raw, prepared.preprocessor, cfg)
    assert again.shape == (5, len(prepared.feature_names))


def test_unseen_category_does_not_crash(cfg, prepared):
    row = D.load_raw(cfg).iloc[[0]].copy()
    row["job"] = "astronaut"
    assert D.preprocess_raw(row, prepared.preprocessor, cfg).shape[0] == 1


def test_class_weights_favour_minority(prepared):
    w = prepared.class_weights
    assert w[1] > w[0]


def test_oversampling_balances_classes(prepared):
    X, y = D.oversample_minority(prepared.X_train, prepared.y_train, seed=1)
    assert y.sum() == len(y) - y.sum()
