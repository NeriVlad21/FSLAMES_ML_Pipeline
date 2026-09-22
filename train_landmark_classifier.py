"""CLI entry point for the modular FSLAMES landmark training pipeline."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fslames_ml.bundle import create_mobile_bundle, write_training_metadata
from fslames_ml.augmentation import augment_landmarks
from fslames_ml.data import load_dataset
from fslames_ml.errors import fail
from fslames_ml.metrics import classification_metrics, confusion_matrix, write_csv_matrix
from fslames_ml.modeling import (
    build_model,
    class_weights,
    export_tflite,
    require_tensorflow,
    verify_tflite,
)
from fslames_ml.quality import validate_dataset_quality
from fslames_ml.splitting import make_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MediaPipe-landmark classifier and export an app bundle."
    )
    parser.add_argument("csv", type=Path, help="Input landmark CSV file.")
    parser.add_argument("--output-dir", type=Path, default=Path("training_output"))
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--source-column", default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--quantization", choices=("float16", "dynamic", "none"), default="float16"
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--augmentation-copies",
        type=int,
        default=4,
        help="Mild synthetic copies per real training row. Validation/test stay real.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.csv, args.label_column, args.source_column)
    quality_summary = validate_dataset_quality(dataset.frame)
    if "sign_type" in dataset.frame.columns:
        kinds = set(dataset.frame["sign_type"].astype(str).str.lower())
        if kinds != {"static"}:
            fail("This trainer accepts static signs only. Use train_sequence_classifier.py for dynamic signs.")
    splits = make_splits(
        dataset.frame,
        dataset.targets,
        dataset.source_column,
        args.validation_fraction,
        args.test_fraction,
        args.seed,
    )
    summary = {
        "csv": str(args.csv.resolve()),
        "rows": int(len(dataset.frame)),
        "features": len(dataset.feature_columns),
        "hands_represented": dataset.hand_count,
        "labels": dataset.labels,
        "class_counts": {
            label: int((dataset.targets == index).sum())
            for index, label in enumerate(dataset.labels)
        },
        "split": asdict(splits.summary),
        "warning": (
            None
            if splits.summary.grouped_by_source
            else "Frame-level split may overestimate generalization accuracy."
        ),
        "quality": quality_summary.to_dict(),
    }
    print(json.dumps(summary, indent=2))
    if args.inspect_only:
        return

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.output_dir / "fslames_landmark_classifier.keras"
    x_train, y_train = augment_landmarks(
        dataset.features[splits.training],
        dataset.targets[splits.training],
        copies=args.augmentation_copies,
        seed=args.seed,
    )
    model = build_model(
        tf, x_train, len(dataset.labels), args.seed
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            dataset.features[splits.validation],
            dataset.targets[splits.validation],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(
            y_train, len(dataset.labels)
        ),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=18, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                keras_path, monitor="val_loss", save_best_only=True
            ),
        ],
        verbose=2,
    )
    model = tf.keras.models.load_model(keras_path)
    probabilities = model.predict(dataset.features[splits.testing], verbose=0)
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    matrix = confusion_matrix(
        dataset.targets[splits.testing], predictions, len(dataset.labels)
    )
    metrics = classification_metrics(matrix, dataset.labels)

    tflite_path = args.output_dir / "fslames_landmark_classifier.tflite"
    export_tflite(tf, model, tflite_path, args.quantization)
    verification = verify_tflite(
        tf, model, tflite_path, dataset.features[splits.testing[0]]
    )
    pd.DataFrame(history.history).to_csv(
        args.output_dir / "training_history.csv", index=False
    )
    write_csv_matrix(
        args.output_dir / "confusion_matrix.csv", matrix, dataset.labels
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps({"labels": dataset.labels}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = write_training_metadata(
        args.output_dir / "model_metadata.json",
        dataset_summary=summary,
        feature_columns=dataset.feature_columns,
        quantization=args.quantization,
        metrics=metrics,
        verification=verification,
    )
    metadata["augmentation"] = {
        "training_only": True,
        "copies_per_real_sample": args.augmentation_copies,
        "rotation_radians": 0.12,
        "scale_range": [0.92, 1.08],
        "gaussian_noise_stddev": 0.004,
    }
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bundle = create_mobile_bundle(
        args.output_dir, tflite_path, dataset.labels, metadata
    )
    print(f"Saved Keras model: {keras_path}")
    print(f"Saved TensorFlow Lite model: {tflite_path}")
    print(f"Saved app-ready mobile bundle: {bundle}")
    print(f"Held-out accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("TFLite verification: PASSED")


if __name__ == "__main__":
    main()
