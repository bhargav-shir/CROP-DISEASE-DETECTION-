"""
Crop Disease Detection - Local training script
----------------------------------------------
Runs on your own machine (no Google Colab / Google Drive needed).

Produces (in the output folder):
  - crop_disease_model.h5
  - class_names.json
which app.py then loads.

Setup:
  pip install tensorflow scikit-learn numpy
  # (optional) for GPU on Windows/Linux, install a TensorFlow build that matches your CUDA setup

Dataset:
  Option A (Kaggle CLI):
      pip install kaggle
      # put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\)
      kaggle datasets download -d vipoooool/new-plant-diseases-dataset -p data --unzip
      # the train images usually end up in a path like:
      #   data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train
  Option B: your own folder, one sub-folder per class:
      my_data/
        Tomato_healthy/   img1.jpg ...
        Tomato_blight/    img1.jpg ...

Usage:
  python train_local.py --data-dir "path/to/train"
  python train_local.py --data-dir "path/to/train" --save-dir ./output --epochs 15 --fine-tune-epochs 10
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------
def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Train crop disease classifier locally")
    p.add_argument("--data-dir", default=os.path.join(script_dir, "data", "train"),
                   help="Folder containing one sub-folder per class")
    p.add_argument("--save-dir", default=os.path.join(script_dir, "output"),
                   help="Where to save the model and class_names.json")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15, help="Stage 1 epochs (frozen base)")
    p.add_argument("--fine-tune-epochs", type=int, default=10, help="Stage 2 epochs")
    p.add_argument("--unfreeze-layers", type=int, default=30,
                   help="Number of top base-model layers to unfreeze in stage 2")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def setup_gpu():
    """Let TensorFlow grow GPU memory instead of grabbing it all (if a GPU exists)."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("GPU config warning:", e)
    print(f"GPUs available: {len(gpus)}")


def make_callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
    ]


def main():
    args = parse_args()
    img_size = (args.img_size, args.img_size)

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"Data folder not found: {args.data_dir}\n"
            "Pass the correct path with --data-dir (a folder with one sub-folder per class)."
        )

    os.makedirs(args.save_dir, exist_ok=True)
    setup_gpu()

    # ------------------------------------------------------------------------
    # Data pipelines
    # ------------------------------------------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=args.batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    with open(os.path.join(args.save_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ])

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    # ------------------------------------------------------------------------
    # Model (MobileNetV2 transfer learning)
    # ------------------------------------------------------------------------
    num_classes = len(class_names)

    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False  # freeze for stage 1

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = os.path.join(args.save_dir, "checkpoint.h5")

    # ------------------------------------------------------------------------
    # Stage 1: train the new head (frozen base)
    # ------------------------------------------------------------------------
    print("\n=== Stage 1: training classifier head ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=make_callbacks(checkpoint_path))

    # ------------------------------------------------------------------------
    # Stage 2: fine-tune the top layers of the base model
    # ------------------------------------------------------------------------
    print("\n=== Stage 2: fine-tuning top layers ===")
    base_model.trainable = True
    for layer in base_model.layers[:-args.unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
              callbacks=make_callbacks(checkpoint_path))

    # ------------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------------
    model_path = os.path.join(args.save_dir, "crop_disease_model.h5")
    model.save(model_path)
    print(f"\nSaved model and class_names.json to: {args.save_dir}")

    # ------------------------------------------------------------------------
    # Quick evaluation
    # ------------------------------------------------------------------------
    print("\n=== Evaluation on validation set ===")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print(classification_report(y_true, y_pred, target_names=class_names))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()