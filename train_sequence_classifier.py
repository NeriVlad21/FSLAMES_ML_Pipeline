"""Train dynamic one- or two-hand signs from participant-grouped videos."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from fslames_ml.augmentation import augment_landmarks
from fslames_ml.bundle import create_mobile_bundle, write_training_metadata
from fslames_ml.data import load_dataset
from fslames_ml.errors import fail
from fslames_ml.metrics import classification_metrics, confusion_matrix, write_csv_matrix
from fslames_ml.modeling import (
    build_sequence_model,
    class_weights,
    export_tflite,
    require_tensorflow,
    verify_tflite,
)
from fslames_ml.sequences import build_sequences
from fslames_ml.splitting import make_splits
from fslames_ml.quality import validate_dataset_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dynamic FSL sequence classifier.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("dynamic_training_output"))
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmentation-copies", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quantization", choices=("float16", "dynamic", "none"), default="float16")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = load_dataset(args.csv, "label", "participant_id")
    quality_summary = validate_dataset_quality(dataset.frame)
    if "sign_type" in dataset.frame.columns:
        kinds = set(dataset.frame["sign_type"].astype(str).str.lower())
        if kinds != {"dynamic"}:
            fail("The sequence trainer accepts dynamic-sign CSVs only.")
    sequences = build_sequences(dataset, args.sequence_length)
    splits = make_splits(
        sequences.frame, sequences.targets, "participant_id", 0.2, 0.2, args.seed
    )
    summary = {
        "csv": str(args.csv.resolve()),
        "source_sequences": int(len(sequences.frame)),
        "sequence_length": args.sequence_length,
        "features_per_frame": int(sequences.features.shape[-1]),
        "hands_represented": dataset.hand_count,
        "labels": dataset.labels,
        "split": asdict(splits.summary),
        "quality": quality_summary.to_dict(),
    }
    print(json.dumps(summary, indent=2))
    if args.inspect_only:
        return

    tf = require_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)
    x_train, y_train = augment_landmarks(
        sequences.features[splits.training],
        sequences.targets[splits.training],
        copies=args.augmentation_copies,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = args.output_dir / "fslames_sequence_classifier.keras"
    model = build_sequence_model(tf, x_train, len(dataset.labels), args.seed)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            sequences.features[splits.validation],
            sequences.targets[splits.validation],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights(y_train, len(dataset.labels)),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=24, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=9, factor=0.5, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(keras_path, monitor="val_loss", save_best_only=True),
        ],
        verbose=2,
    )
    model = tf.keras.models.load_model(keras_path)
    probabilities = model.predict(sequences.features[splits.testing], verbose=0)
    predictions = np.argmax(probabilities, axis=1).astype(np.int32)
    matrix = confusion_matrix(sequences.targets[splits.testing], predictions, len(dataset.labels))
    metrics = classification_metrics(matrix, dataset.labels)
    tflite_path = args.output_dir / "fslames_landmark_classifier.tflite"
    export_tflite(tf, model, tflite_path, args.quantization)
    verification = verify_tflite(tf, model, tflite_path, sequences.features[splits.testing[0]])
    pd.DataFrame(history.history).to_csv(args.output_dir / "training_history.csv", index=False)
    write_csv_matrix(args.output_dir / "confusion_matrix.csv", matrix, dataset.labels)
    model_kind = (
        "dynamic_single_hand_sequence_classifier"
        if dataset.hand_count == 1
        else "dynamic_two_hand_sequence_classifier"
    )
    metadata = write_training_metadata(
        args.output_dir / "model_metadata.json",
        dataset_summary=summary,
        feature_columns=dataset.feature_columns,
        quantization=args.quantization,
        metrics=metrics,
        verification=verification,
        model_kind=model_kind,
    )
    metadata["augmentation"] = {
        "training_only": True,
        "copies_per_real_sequence": args.augmentation_copies,
    }
    metadata["input_contract"]["sequence_length"] = args.sequence_length
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bundle = create_mobile_bundle(args.output_dir, tflite_path, dataset.labels, metadata)
    print(f"Saved dynamic model: {tflite_path}")
    print(f"Saved bundle: {bundle}")
    print(f"Held-out participant accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
