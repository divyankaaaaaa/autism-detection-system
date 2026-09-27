"""
Trains the autism detection model using MobileNetV2 transfer learning.

Two-phase training:
  1. Freeze the MobileNetV2 base, train only the new classifier head.
  2. Unfreeze the top layers of the base and fine-tune at a low learning rate.

Run:
    python -m src.train
"""

import os
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving plots
import matplotlib.pyplot as plt

from src import tf_compat  # noqa: F401  — must be imported before tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src import config
from src.data_loader import get_generators


def build_model():
    base_model = MobileNetV2(
        input_shape=config.IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze for phase 1

    inputs = layers.Input(shape=config.IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history_list, save_path):
    """history_list can contain one or two keras History objects (phase 1 [+ phase 2])."""
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in history_list:
        acc += h.history["accuracy"]
        val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]
        val_loss += h.history["val_loss"]

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend(loc="lower right")
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend(loc="upper right")
    plt.title("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    train_gen, val_gen, _ = get_generators()

    model, base_model = build_model()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE_HEAD),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
        ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    print("\n=== Phase 1: training classifier head (base frozen) ===")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_HEAD,
        callbacks=callbacks,
    )

    print("\n=== Phase 2: fine-tuning top layers of MobileNetV2 ===")
    base_model.trainable = True
    for layer in base_model.layers[: config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.LEARNING_RATE_FINE_TUNE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_FINE_TUNE,
        callbacks=callbacks,
    )

    plot_history([history1, history2], os.path.join(config.REPORTS_DIR, "training_curves.png"))

    model.save(config.MODEL_PATH)
    print(f"\nFinal model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
