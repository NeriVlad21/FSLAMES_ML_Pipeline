from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

from .errors import fail
from .sequences import TRAJECTORY_COLUMNS


BUNDLE_VERSION = 2

# The four independent model slots. Each slot has its own file names so all
# four bundles can be installed side by side in the Flutter asset directory.
MODEL_SLOTS = {
    "static_single": {
        "model_kind": "static_single_hand_frame_classifier",
        "sign_type": "static",
        "required_hands": 1,
    },
    "static_both": {
        "model_kind": "static_two_hand_frame_classifier",
        "sign_type": "static",
        "required_hands": 2,
    },
    "dynamic_single": {
        "model_kind": "dynamic_single_hand_sequence_classifier",
        "sign_type": "dynamic",
        "required_hands": 1,
    },
    "dynamic_both": {
        "model_kind": "dynamic_two_hand_sequence_classifier",
        "sign_type": "dynamic",
        "required_hands": 2,
    },
}

# Two-hand rows store h0 then h1. The extractor puts MediaPipe "Left" before
# "Right"; when handedness is ambiguous (both hands given the same label) it
# falls back to the hand whose wrist has the smaller image x. The app must
# order hands the same way before encoding.
TWO_HAND_ORDERING = "mediapipe_left_then_right_else_smaller_wrist_x_first"


def slot_name(sign_type: str, required_hands: int) -> str:
    slot = f"{sign_type}_{'single' if required_hands == 1 else 'both'}"
    if slot not in MODEL_SLOTS:
        fail(f"Unsupported model slot: sign_type={sign_type}, hands={required_hands}")
    return slot


def model_filename(slot: str) -> str:
    return f"fslames_{slot}_classifier.tflite"


def manifest_filename(slot: str) -> str:
    return f"fslames_{slot}_manifest.json"


def build_input_contract(
    feature_columns: Sequence[str],
    *,
    sequence_length: int | None = None,
) -> dict:
    """Describe exactly what the exported TFLite input tensor contains."""
    landmark_count = len(feature_columns)
    if landmark_count not in (63, 126):
        fail("Only 63 (one-hand) or 126 (two-hand) landmark features are supported.")
    hands = landmark_count // 63
    contract = {
        "preprocessing": (
            "wrist_relative_xyz" if hands == 1 else "primary_wrist_relative_xyz"
        ),
        "landmark_feature_count": landmark_count,
        "dtype": "float32",
    }
    if hands == 2:
        contract["hand_ordering"] = TWO_HAND_ORDERING
    if sequence_length is None:
        contract.update(
            coordinate_order=list(feature_columns),
            feature_count=landmark_count,
            input_shape=[1, landmark_count],
        )
        return contract
    per_frame = landmark_count + len(TRAJECTORY_COLUMNS)
    contract.update(
        coordinate_order=list(feature_columns) + list(TRAJECTORY_COLUMNS),
        feature_count=per_frame,
        sequence_length=int(sequence_length),
        input_shape=[1, int(sequence_length), per_frame],
        temporal_resampling="uniform_nearest_index_over_detected_frames",
        trajectory={
            "columns": list(TRAJECTORY_COLUMNS),
            "center": "mean image x/y of all landmarks of all required hands",
            "scale": "median palm_size (wrist to middle-finger MCP of hand h0) "
                     "over the resampled sequence, minimum 0.001",
            "displacement": "(center[t] - center[0]) / scale",
            "velocity": "displacement[t] - displacement[t-1], 0 at t=0",
        },
    )
    return contract


def verify_contract_shape(contract: dict, verification: dict) -> None:
    """Fail when the exported TFLite input does not match the declared contract."""
    actual = [int(value) for value in verification["input_shape"]]
    if actual != contract["input_shape"]:
        fail(
            "TFLite input shape does not match the input contract: "
            f"{actual} != {contract['input_shape']}"
        )


def write_training_metadata(
    path: Path,
    *,
    dataset_summary: dict,
    feature_columns: Sequence[str],
    quantization: str,
    metrics: dict,
    verification: dict,
    model_kind: str | None = None,
    sequence_length: int | None = None,
    decision_policy: dict | None = None,
    cvi: dict | None = None,
    augmentation: dict | None = None,
) -> dict:
    hands = len(feature_columns) // 63
    sign_type = "static" if sequence_length is None else "dynamic"
    slot = slot_name(sign_type, hands)
    expected_kind = MODEL_SLOTS[slot]["model_kind"]
    if model_kind is not None and model_kind != expected_kind:
        fail(f"Model kind {model_kind} does not match slot {slot} ({expected_kind}).")
    contract = build_input_contract(feature_columns, sequence_length=sequence_length)
    verify_contract_shape(contract, verification)
    cvi = cvi or {}
    metadata = {
        **dataset_summary,
        "bundle_version": BUNDLE_VERSION,
        "slot": slot,
        "model_kind": expected_kind,
        "sign_type": sign_type,
        "required_hands": hands,
        "input_contract": contract,
        "feature_columns": list(feature_columns),
        "quantization": quantization,
        "metrics": metrics,
        "tflite_verification": verification,
        "decision_policy": decision_policy or {},
        "cvi": cvi,
        "release_ready": cvi.get("status") == "expert_validated",
        "augmentation": augmentation or {},
        "important_limitations": [
            "Use held-out people and recordings for defensible accuracy metrics.",
            (
                "This frame classifier must not grade dynamic signs."
                if sign_type == "static"
                else "This sequence classifier must not grade static signs."
            ),
            "Dataset CVI and classifier accuracy are separate evaluations.",
        ],
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def create_mobile_bundle(output_dir: Path, model_path: Path, labels: Sequence[str], metadata: dict) -> Path:
    slot = metadata["slot"]
    bundle = output_dir / "mobile_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, bundle / model_filename(slot))
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "slot": slot,
        "model_file": model_filename(slot),
        "model_kind": metadata["model_kind"],
        "sign_type": metadata["sign_type"],
        "required_hands": metadata["required_hands"],
        "labels": list(labels),
        "input_contract": metadata["input_contract"],
        "test_accuracy": metadata["metrics"]["accuracy"],
        "macro_f1": metadata["metrics"]["macro_f1"],
        "decision_policy": metadata.get("decision_policy", {}),
        "cvi": metadata.get("cvi", {}),
        "release_ready": bool(metadata.get("release_ready", False)),
    }
    (bundle / manifest_filename(slot)).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return bundle
