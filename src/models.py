"""Neural network architectures for the Bank Marketing term-deposit task.

Owner: Model Architect (Role 4). This module holds:
  * ``build_mlp``            - one flexible MLP builder (used by B, C and the ablations)
  * ``build_model_b``        - Model B: deeper MLP with Dropout regularization
  * ``build_model_c``        - Model C: best MLP after tuning/ablations
  * ``hand_param_counts``    - layer-by-layer parameter formulas to compare with model.summary()
  * ``gradient_tape_demo``   - one manual training loop with tf.GradientTape

``build_model_a`` (shallow MLP) is owned by the Baseline/Analysis Engineer and is kept unchanged.
"""
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_model_a(input_dim):
    """Shallow MLP baseline (Model A) for the Bank Marketing task.

    Architecture:
        - Input: (input_dim,)
        - Dense: 32 units, ReLU activation
        - Output: 1 unit, Sigmoid activation
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="model_a_shallow")
    return model


def build_mlp(
    input_dim: int,
    hidden_units: Sequence[int] = (128, 64, 32),
    dropout: float = 0.3,
    activation: str = "relu",
    use_batchnorm: bool = False,
    initializer: str = "he_normal",
    l2: float = 0.0,
    name: str = "mlp",
) -> keras.Model:
    """Build a binary-classification MLP.

    Args:
        input_dim: number of input features AFTER scaling and one-hot encoding.
        hidden_units: units of each hidden Dense layer, e.g. (128, 64, 32).
        dropout: Dropout rate after each hidden block (0 disables Dropout).
        activation: hidden activation ("relu", "tanh", "sigmoid", ...).
        use_batchnorm: add BatchNormalization after each Dense layer.
        initializer: kernel initializer ("he_normal" or "glorot_uniform").
        l2: L2 penalty on the Dense kernels (0 disables it).
        name: Keras model name.

    Returns:
        An uncompiled Keras model whose output is one sigmoid probability.
    """
    reg = regularizers.l2(l2) if l2 > 0 else None
    model_layers: List[layers.Layer] = [layers.Input(shape=(input_dim,))]
    for units in hidden_units:
        model_layers.append(
            layers.Dense(units, activation=activation,
                         kernel_initializer=initializer, kernel_regularizer=reg)
        )
        if use_batchnorm:
            model_layers.append(layers.BatchNormalization())
        if dropout > 0:
            model_layers.append(layers.Dropout(dropout))
    # sigmoid + binary cross-entropy: the target is a single yes/no probability
    model_layers.append(layers.Dense(1, activation="sigmoid"))
    return keras.Sequential(model_layers, name=name)


def build_model_b(input_dim: int, **overrides) -> keras.Model:
    """Model B: deeper MLP (128-64-32) with Dropout(0.3), ReLU and He init.

    Keyword overrides let the ablations change ONE factor at a time
    (e.g. ``dropout=0.0``, ``activation="tanh"``, ``hidden_units=(64, 32)``).
    """
    cfg = dict(hidden_units=(128, 64, 32), dropout=0.3, activation="relu",
               use_batchnorm=False, initializer="he_normal", l2=0.0, name="model_b")
    cfg.update(overrides)
    return build_mlp(input_dim, **cfg)


def build_model_c(input_dim: int, best_config: Dict = None) -> keras.Model:
    """Model C: best MLP from validation ablations (64-32, Dropout 0.5, ReLU, He init).

    Pass ``best_config`` to override any parameter after final tuning sweeps.
    """
    cfg = dict(hidden_units=(64, 32), dropout=0.5, activation="relu",
               use_batchnorm=False, initializer="he_normal", l2=0.0, name="model_c")
    if best_config:
        cfg.update(best_config)
    return build_mlp(input_dim, **cfg)


def hand_param_counts(
    input_dim: int,
    hidden_units: Sequence[int] = (128, 64, 32),
    dropout: float = 0.3,
    use_batchnorm: bool = False,
) -> List[Dict]:
    """Compute the parameter count of every layer with the textbook formulas.

    Dense = inputs * units + units
    BatchNorm = 4 * units (2 trainable: gamma, beta; 2 non-trainable: moving mean/var)
    Dropout = 0
    """
    rows, n_in = [], input_dim
    for units in hidden_units:
        rows.append({"layer": f"Dense({units})", "shape": (None, units),
                     "params": n_in * units + units})
        if use_batchnorm:
            rows.append({"layer": "BatchNorm", "shape": (None, units), "params": 4 * units})
        if dropout > 0:
            rows.append({"layer": "Dropout", "shape": (None, units), "params": 0})
        n_in = units
    rows.append({"layer": "Dense(1)", "shape": (None, 1), "params": n_in * 1 + 1})
    return rows


def check_against_summary(model: keras.Model, hand_rows: List[Dict]) -> bool:
    """Return True when the hand total equals ``model.count_params()``."""
    hand_total = sum(r["params"] for r in hand_rows)
    return hand_total == model.count_params()


def gradient_tape_demo(model: keras.Model, x: np.ndarray, y: np.ndarray,
                       steps: int = 50, lr: float = 1e-3, seed: int = 42) -> List[float]:
    """Train ``model`` manually with tf.GradientTape and return the loss per step.

    Each step is: forward pass -> loss -> gradients -> optimizer update.
    The loss should decrease, which shows the training mechanics are understood.
    """
    tf.random.set_seed(seed)
    loss_fn = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(lr)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            preds = model(xb, training=True)          # 1) forward pass (Dropout active)
            loss = loss_fn(yb, preds)                 # 2) loss
        grads = tape.gradient(loss, model.trainable_variables)             # 3) gradients
        optimizer.apply_gradients(zip(grads, model.trainable_variables))   # 4) update
        return loss

    losses = []
    for step in range(steps):
        losses.append(float(train_step(x, y)))
        if step % 10 == 0:
            print(f"step {step:3d}  loss = {losses[-1]:.4f}")
    return losses


if __name__ == "__main__":
    dim = 51
    for label, mdl, units, drop, bn in [
        ("Model A", build_model_a(dim), (32,), 0.0, False),
        ("Model B", build_model_b(dim), (128, 64, 32), 0.3, False),
        ("Model C", build_model_c(dim), (64, 32), 0.5, False),
    ]:
        rows = hand_param_counts(dim, hidden_units=units, dropout=drop, use_batchnorm=bn)
        assert check_against_summary(mdl, rows)
        print(f"{label}: params={mdl.count_params()} (hand check passed)")
