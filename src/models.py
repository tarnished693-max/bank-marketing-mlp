"""Neural network architectures for the Bank Marketing task."""
from tensorflow import keras
from tensorflow.keras import layers


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
