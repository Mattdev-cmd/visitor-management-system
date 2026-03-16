"""
Train a lightweight emotion-recognition CNN on the FER-2013 dataset and
export both a Keras (.h5) and TensorFlow Lite (.tflite) model.

Usage
-----
1.  Download the FER-2013 CSV from:
    https://www.kaggle.com/datasets/msambare/fer2013
2.  Place ``fer2013.csv`` in the project root (or pass --csv <path>).
3.  Run:
        python models/train_emotion_model.py
4.  The script saves:
        models/emotion_model.h5
        models/emotion_model.tflite
"""

import argparse
import os
import sys

import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def load_fer2013(csv_path: str):
    """Load the FER-2013 CSV and return (X_train, y_train), (X_test, y_test)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    pixels = df["pixels"].apply(lambda x: np.fromstring(x, sep=" ", dtype=np.float32))
    X = np.stack(pixels).reshape(-1, 48, 48, 1) / 255.0
    y = df["emotion"].values

    train_mask = df["Usage"] == "Training"
    return (X[train_mask], y[train_mask]), (X[~train_mask], y[~train_mask])


def build_model(num_classes: int = 7):
    """Build a small CNN suitable for Raspberry Pi."""
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_tflite(keras_path: str, tflite_path: str):
    import tensorflow as tf

    model = tf.keras.models.load_model(keras_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # quantise for Pi
    tflite_bytes = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[Train] TFLite model saved → {tflite_path}  ({len(tflite_bytes)/1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition model")
    parser.add_argument("--csv", default="fer2013.csv", help="Path to fer2013.csv")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"ERROR: {args.csv} not found. Download FER-2013 from Kaggle first.")
        sys.exit(1)

    (X_train, y_train), (X_test, y_test) = load_fer2013(args.csv)
    print(f"[Train] Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = build_model()
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
    )

    keras_path = config.EMOTION_MODEL_PATH.replace(".tflite", ".h5")
    model.save(keras_path)
    print(f"[Train] Keras model saved → {keras_path}")

    export_tflite(keras_path, config.EMOTION_MODEL_PATH)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Train] Test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
