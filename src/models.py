"""Neural network architectures for the Bank Marketing task."""
from tensorflow import keras
from tensorflow.keras import layers


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


def build_model_b(input_dim):
    """Deeper MLP with Dropout for the Bank Marketing task."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model
