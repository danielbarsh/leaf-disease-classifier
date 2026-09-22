"""
train.py
--------
Trains a tomato leaf image classification model (healthy / 5 diseases) using
Transfer Learning on top of MobileNetV2 pretrained on ImageNet.

Why Transfer Learning?
Instead of training a new network from scratch (which requires a huge amount
of images and compute time), we take a network that already "knows how to
see" (trained on millions of images), freeze it, and add small classification
layers on top tailored specifically to our task. This is the standard
industry approach when you have a small-to-medium dataset.

Run:
    python src/train.py
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # no graphical window needed on a server
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_utils as mu

EPOCHS_HEAD = 8           # training the top layers (base frozen)
EPOCHS_FINE_TUNE = 4      # second stage: "unfreeze" part of the base and fine-tune at a slow pace
FINE_TUNE_AT_LAYER = 100  # from which layer in the base to start training (lower layers stay frozen)
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        mu.DATA_DIR / "train",
        image_size=mu.IMG_SIZE,
        batch_size=mu.BATCH_SIZE,
        seed=mu.SEED,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        mu.DATA_DIR / "val",
        image_size=mu.IMG_SIZE,
        batch_size=mu.BATCH_SIZE,
        seed=mu.SEED,
        shuffle=False,
    )
    class_names = train_ds.class_names

    # I/O optimization - prefetches the next batch while the model processes the current one
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def load_pretrained_base():
    """
    Loads MobileNetV2 with ImageNet weights.

    On some restricted/corporate networks, the normal download from
    storage.googleapis.com may be blocked. If that happens, we fall back to a
    public mirror of the exact same weights hosted on GitHub Releases, and
    place it in Keras's cache so that Keras's normal download "finds" it
    instead of trying the blocked URL again.
    """
    try:
        return tf.keras.applications.MobileNetV2(
            input_shape=mu.IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )
    except Exception as e:
        print(f"Normal download of ImageNet weights failed ({e}).")
        print("Trying alternative GitHub mirror...")

        import urllib.request
        cache_path = pathlib.Path.home() / ".keras" / "models" / (
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        mirror_url = (
            "https://github.com/JonathanCMitchell/mobilenet_v2_keras/"
            "releases/download/v1.1/"
            "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
        )
        urllib.request.urlretrieve(mirror_url, cache_path)
        print(f"Weights successfully downloaded from the alternative mirror to {cache_path}")

        return tf.keras.applications.MobileNetV2(
            input_shape=mu.IMG_SIZE + (3,),
            include_top=False,
            weights="imagenet",
        )


def build_model(num_classes: int):
    # Data augmentation layers - slightly vary the training images
    # (rotation, flip, zoom) so the model generalizes better and doesn't memorize.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # MobileNetV2 expects input in the range [-1, 1]
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = load_pretrained_base()
    base_model.trainable = False  # stage 1: freeze the base

    inputs = tf.keras.Input(shape=mu.IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def plot_history(history_head, history_fine, out_path: pathlib.Path):
    acc = history_head.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history_head.history["loss"] + history_fine.history["loss"]
    val_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]

    switch_epoch = len(history_head.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(acc, label="Train accuracy")
    axes[0].plot(val_acc, label="Validation accuracy")
    axes[0].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(loss, label="Train loss")
    axes[1].plot(val_loss, label="Validation loss")
    axes[1].axvline(switch_epoch - 0.5, color="gray", linestyle="--", label="Fine-tuning starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Training plot saved to {out_path}")


def main():
    mu.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, class_names = build_datasets()
    print(f"Classes: {class_names}")
    mu.save_class_names(class_names)

    print("\nBuilding model (MobileNetV2 + Transfer Learning)...")
    model, base_model = build_model(num_classes=len(class_names))
    model.summary()

    # --- Stage 1: train the top layers only ---
    print(f"\n=== Stage 1/2: Training top layers ({EPOCHS_HEAD} epochs, base frozen) ===")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD)

    # --- Stage 2: Fine-tuning - unfreeze part of the top layers in the base ---
    print(f"\n=== Stage 2/2: Fine-tuning ({EPOCHS_FINE_TUNE} epochs, low learning rate) ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD + EPOCHS_FINE_TUNE,
        initial_epoch=history_head.epoch[-1] + 1,
    )

    print(f"\nSaving model to {mu.MODEL_PATH}")
    model.save(mu.MODEL_PATH)

    plot_history(history_head, history_fine, mu.HISTORY_PLOT_PATH)

    final_val_acc = history_fine.history["val_accuracy"][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.2%}")
    print("Training completed successfully! You can now run python src/evaluate.py")


if __name__ == "__main__":
    main()
