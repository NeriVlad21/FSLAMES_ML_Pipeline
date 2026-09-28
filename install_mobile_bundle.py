"""Validate and install a trained classifier bundle into Flutter assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from fslames_ml.bundle import (
    BUNDLE_VERSION,
    MODEL_SLOTS,
    build_input_contract,
    manifest_filename,
    model_filename,
)
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
    parser.add_argument(
        "--allow-development-bundle",
        action="store_true",
        help="Install a bundle trained without expert CVI (never for release builds).",
    )
    return parser.parse_args()


def _find_manifest(bundle: Path) -> Path:
    manifests = sorted(bundle.glob("fslames_*_manifest.json"))
    if len(manifests) != 1:
        fail(f"Expected exactly one fslames_<slot>_manifest.json in {bundle}.")
    return manifests[0]


def validate(bundle: Path, *, allow_development: bool = False) -> tuple[Path, Path, dict]:
    manifest_path = _find_manifest(bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        fail(f"Only classifier bundle version {BUNDLE_VERSION} is supported.")
    slot = manifest.get("slot")
    if slot not in MODEL_SLOTS:
        fail(f"Unknown model slot: {slot!r}")
    spec = MODEL_SLOTS[slot]
    if manifest_path.name != manifest_filename(slot):
        fail(f"Manifest name does not match slot {slot}.")
    for key in ("model_kind", "sign_type", "required_hands"):
        if manifest.get(key) != spec[key]:
            fail(f"Manifest {key}={manifest.get(key)!r} does not match slot {slot}.")
    if manifest.get("model_file") != model_filename(slot):
        fail(f"Manifest model_file must be {model_filename(slot)}.")
    model = bundle / model_filename(slot)
    if not model.is_file() or model.stat().st_size == 0:
        fail(f"Missing or empty model: {model}")

    contract = manifest.get("input_contract", {})
    order = contract.get("coordinate_order", [])
    landmark_count = 63 * spec["required_hands"]
    expected = build_input_contract(
        order[:landmark_count],
        sequence_length=contract.get("sequence_length") if spec["sign_type"] == "dynamic" else None,
    )
    for key in ("feature_count", "input_shape", "preprocessing", "coordinate_order"):
        if contract.get(key) != expected[key]:
            fail(f"Input contract {key} does not match the {slot} slot.")

    labels = manifest.get("labels")
    if not isinstance(labels, list) or len(labels) < 2:
        fail("The classifier manifest must contain at least two labels.")
    policy = manifest.get("decision_policy", {})
    if policy.get("mode") != "target_top1_with_margin" or not policy.get("require_target_is_top1"):
        fail("The manifest has no target-aware decision policy.")
    if not manifest.get("release_ready"):
        if not allow_development:
            fail(
                "This bundle has no expert CVI approval (release_ready=false). "
                "Use --allow-development-bundle only for non-release builds."
            )
        print("WARNING: installing a development-only bundle.")
    return model, manifest_path, manifest


def main() -> None:
    args = parse_args()
    model, manifest_path, manifest = validate(
        args.bundle, allow_development=args.allow_development_bundle
    )
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, args.assets_dir / model.name)
    shutil.copy2(manifest_path, args.assets_dir / manifest_path.name)
    print(f"Installed {manifest['slot']} model: {args.assets_dir / model.name}")
    print(f"Installed manifest: {args.assets_dir / manifest_path.name}")
    print("Run flutter analyze, flutter test, and flutter run before release.")


if __name__ == "__main__":
    main()
