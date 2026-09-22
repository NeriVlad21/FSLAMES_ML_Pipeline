"""Validate and install a trained classifier bundle into Flutter assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fslames_ml.bundle import MANIFEST_FILENAME, MODEL_FILENAME
from fslames_ml.errors import fail


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


def validate(bundle: Path) -> tuple[Path, Path]:
    model = bundle / MODEL_FILENAME
    manifest_path = bundle / MANIFEST_FILENAME
    if not model.is_file() or model.stat().st_size == 0:
        fail(f"Missing or empty model: {model}")
    if not manifest_path.is_file():
        fail(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != 1:
        fail("Only classifier bundle version 1 is supported by the current app.")
    if manifest.get("model_kind") != "static_single_hand_frame_classifier":
        fail("The current app bundle accepts only a static single-hand classifier.")
    contract = manifest.get("input_contract", {})
    if contract.get("feature_count") != 63:
        fail("The app requires exactly 63 landmark features.")
    if contract.get("preprocessing") != "wrist_relative_xyz":
        fail("The model preprocessing does not match the app encoder.")
    labels = manifest.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        fail("The classifier manifest must contain at least two labels.")
    return model, manifest_path


def main() -> None:
    args = parse_args()
    model, manifest = validate(args.bundle)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, args.assets_dir / MODEL_FILENAME)
    shutil.copy2(manifest, args.assets_dir / MANIFEST_FILENAME)
    print(f"Installed model: {args.assets_dir / MODEL_FILENAME}")
    print(f"Installed manifest: {args.assets_dir / MANIFEST_FILENAME}")
    print("Run flutter analyze, flutter test, and flutter run before release.")


if __name__ == "__main__":
    main()
