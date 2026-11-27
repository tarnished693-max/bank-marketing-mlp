"""Unit tests for src/models.py (architectures, hand parameter counts, GradientTape)."""
import numpy as np

from src.models import (
    build_model_a,
    build_model_b,
    build_model_c,
    check_against_summary,
    gradient_tape_demo,
    hand_param_counts,
)


def test_models_output_probabilities():
    x = np.random.default_rng(0).normal(size=(8, 51)).astype("float32")
    for builder in (build_model_a, build_model_b, build_model_c):
        model = builder(51)
        preds = model.predict(x, verbose=0)
        assert preds.shape == (8, 1)
        assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_hand_param_counts_match_summary():
    cases = [
        (build_model_a(51), (32,), 0.0, False),
        (build_model_b(51), (128, 64, 32), 0.3, False),
        (build_model_c(51), (64, 32), 0.5, False),
        (build_model_b(51, use_batchnorm=True), (128, 64, 32), 0.3, True),
    ]
    for model, units, drop, bn in cases:
        rows = hand_param_counts(51, hidden_units=units, dropout=drop, use_batchnorm=bn)
        assert len(rows) == len(model.layers)
        assert check_against_summary(model, rows)


def test_gradient_tape_reduces_loss():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(128, 51)).astype("float32")
    y = (x[:, 0] > 0).astype("float32")
    losses = gradient_tape_demo(build_model_b(51), x, y, steps=30, lr=1e-2, seed=42)
    assert len(losses) == 30
    assert losses[-1] < losses[0]
