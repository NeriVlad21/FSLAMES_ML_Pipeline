from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence


BUNDLE_VERSION = 1
MODEL_FILENAME = "fslames_landmark_classifier.tflite"
MANIFEST_FILENAME = "fslames_classifier_manifest.json"


def write_training_metadata(
    path: Path,
    *,
    dataset_summary: dict,
    feature_columns: Sequence[str],
    quantization: str,
    metrics: dict,
    verification: dict,
    model_kind: str | None = None,
) -> dict:
    model_kind = model_kind or (
        "static_single_hand_frame_classifier"
        if len(feature_columns) == 63
        else "static_two_hand_frame_classifier"
    )
    preprocessing = (
        "wrist_relative_xyz"
        if len(feature_columns) == 63
        else "primary_wrist_relative_xyz"
    )
    metadata = {
        **dataset_summary,
        "bundle_version": BUNDLE_VERSION,
        "model_kind": model_kind,
        "input_contract": {
            "coordinate_order": list(feature_columns),
            "preprocessing": preprocessing,
            "feature_count": len(feature_columns),
            "dtype": "float32",
        },
        "feature_columns": list(feature_columns),
        "quantization": quantization,
        "metrics": metrics,
        "tflite_verification": verification,
        "important_limitations": [
            "Use held-out people and recordings for defensible accuracy metrics.",
            "This frame classifier must not grade dynamic signs.",
            "Dataset CVI and classifier accuracy are separate evaluations.",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def create_mobile_bundle(output_dir: Path, model_path: Path, labels: Sequence[str], metadata: dict) -> Path:
    bundle = output_dir / "mobile_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, bundle / MODEL_FILENAME)
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "model_file": MODEL_FILENAME,
        "model_kind": metadata["model_kind"],
        "labels": list(labels),
        "input_contract": metadata["input_contract"],
        "test_accuracy": metadata["metrics"]["accuracy"],
        "macro_f1": metadata["metrics"]["macro_f1"],
    }
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return bundle
