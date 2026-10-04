"""Validate and install a trained classifier bundle into Flutter assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fslames_ml.bundle import MANIFEST_FILENAME, MODEL_FILENAME
from fslames_ml.errors import fail


SLOTS = {
    "static_single_hand_frame_classifier": ("static_single", 63, "wrist_relative_xyz"),
    "static_two_hand_frame_classifier": ("static_both", 126, "primary_wrist_relative_xyz"),
    "dynamic_single_hand_sequence_classifier": (
        "dynamic_single", 67, "wrist_relative_xyz_plus_palm_normalized_trajectory"
    ),
    "dynamic_two_hand_sequence_classifier": (
        "dynamic_both", 130, "wrist_relative_xyz_plus_palm_normalized_trajectory"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install an app-compatible FSLAMES classifier bundle."
    )
    parser.add_argument("bundle", type=Path, help="training_output/mobile_bundle")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets/models"),
        help="Flutter model asset directory.",
    )
    return parser.parse_args()


def validate(bundle: Path) -> tuple[Path, Path, str]:
    model = bundle / MODEL_FILENAME
    manifest_path = bundle / MANIFEST_FILENAME
    if not model.is_file() or model.stat().st_size == 0:
        fail(f"Missing or empty model: {model}")
    if not manifest_path.is_file():
        fail(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != 2:
        fail("Only classifier bundle version 2 is supported by the current app.")
    kind = manifest.get("model_kind")
    if kind not in SLOTS:
        fail(f"Unsupported model kind: {kind}")
    slot, expected_features, expected_preprocessing = SLOTS[kind]
    contract = manifest.get("input_contract", {})
    if contract.get("feature_count") != expected_features:
        fail(f"{kind} requires exactly {expected_features} features.")
    if contract.get("preprocessing") != expected_preprocessing:
        fail("The model preprocessing does not match the app encoder.")
    if kind.startswith("dynamic_") and contract.get("sequence_length") != 24:
        fail("The app runtime currently requires 24-frame dynamic models.")
    labels = manifest.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        fail("The classifier manifest must contain at least two labels.")
    if manifest.get("cvi", {}).get("status") != "expert_validated":
        fail("App installation requires an expert-validated CVI manifest.")
    return model, manifest_path, slot


def main() -> None:
    args = parse_args()
    model, manifest, slot = validate(args.bundle)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    model_name = f"fslames_{slot}_classifier.tflite"
    manifest_name = f"fslames_{slot}_manifest.json"
    shutil.copy2(model, args.assets_dir / model_name)
    shutil.copy2(manifest, args.assets_dir / manifest_name)
    print(f"Installed model: {args.assets_dir / model_name}")
    print(f"Installed manifest: {args.assets_dir / manifest_name}")
    print("Run flutter analyze, flutter test, and flutter run before release.")


if __name__ == "__main__":
    main()
