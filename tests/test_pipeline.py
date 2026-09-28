from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from fslames_ml.augmentation import augment_landmarks
from fslames_ml.bundle import (
    MODEL_SLOTS,
    build_input_contract,
    create_mobile_bundle,
    manifest_filename,
    model_filename,
    write_training_metadata,
)
from fslames_ml.calibration import (
    DEFAULT_TARGET_THRESHOLD,
    accept_target,
    calibrate_decision_policy,
    evaluate_decision_policy,
)
from fslames_ml.data import load_dataset
from fslames_ml.extraction import order_two_hands
from fslames_ml.quality import validate_dataset_quality
from fslames_ml.sequences import TRAJECTORY_COLUMNS, build_sequences
from fslames_ml.splitting import make_splits
from fslames_ml.validation import require_expert_cvi

from synthetic import make_frame, write_csv


# --- dataset gate -----------------------------------------------------------

@pytest.mark.parametrize("hands,sign", [(1, "static"), (2, "static"), (1, "dynamic"), (2, "dynamic")])
def test_quality_gate_accepts_all_four_schemas(hands, sign):
    summary = validate_dataset_quality(make_frame(hands=hands, sign_type=sign))
    assert summary.required_hands == hands and summary.sign_type == sign


def test_quality_gate_requires_five_participants():
    with pytest.raises(SystemExit, match="5 participants"):
        validate_dataset_quality(make_frame(hands=1, sign_type="static", participants=4))


def test_quality_gate_rejects_hand_count_mismatch():
    frame = make_frame(hands=2, sign_type="static")
    frame["required_hands"] = 1
    with pytest.raises(SystemExit):
        validate_dataset_quality(frame)


# --- splitting --------------------------------------------------------------

@pytest.mark.parametrize("hands", [1, 2])
def test_static_split_holds_out_whole_participants(tmp_path, hands):
    dataset = load_dataset(write_csv(tmp_path, hands=hands, sign_type="static"), "label", None)
    assert dataset.source_column == "participant_id"
    splits = make_splits(dataset.frame, dataset.targets, dataset.source_column, 0.15, 0.15, 1)
    assert splits.summary.grouping == "participant"
    people = [set(dataset.frame.loc[idx, "participant_id"]) for idx in
              (splits.training, splits.validation, splits.testing)]
    assert not (people[0] & people[1] or people[0] & people[2] or people[1] & people[2])
    assert len(people[0] | people[1] | people[2]) == 5


# --- sequences: dimensions and trajectory ----------------------------------

@pytest.mark.parametrize("hands,width", [(1, 67), (2, 130)])
def test_sequence_feature_width_includes_trajectory(tmp_path, hands, width):
    dataset = load_dataset(write_csv(tmp_path, hands=hands, sign_type="dynamic"), "label", "participant_id")
    sequences = build_sequences(dataset, 16)
    assert sequences.features.shape[1:] == (16, width)
    trajectory = sequences.features[:, :, -4:]
    assert np.allclose(trajectory[:, 0, :], 0.0)  # starts at origin with zero velocity
    # Velocity is the per-step difference of displacement.
    assert np.allclose(np.diff(trajectory[:, :, 0], axis=1), trajectory[:, 1:, 2], atol=1e-6)
    # Movement is preserved: label C moves further than label A.
    final_dx = {label: [] for label in dataset.labels}
    for row, label in zip(sequences.features, sequences.frame["label"]):
        final_dx[label].append(row[-1, -4])
    assert np.mean(final_dx["C"]) > np.mean(final_dx["A"]) > 0


def test_sequence_split_is_participant_grouped(tmp_path):
    dataset = load_dataset(write_csv(tmp_path, hands=1, sign_type="dynamic"), "label", "participant_id")
    sequences = build_sequences(dataset, 16)
    splits = make_splits(sequences.frame, sequences.targets, "participant_id", 0.2, 0.2, 3)
    train = set(sequences.frame.loc[splits.training, "participant_id"])
    held = set(sequences.frame.loc[np.concatenate([splits.validation, splits.testing]), "participant_id"])
    assert train and held and not train & held


# --- augmentation -----------------------------------------------------------

@pytest.mark.parametrize("shape", [(10, 63), (10, 126), (4, 16, 67), (4, 16, 130)])
def test_augmentation_preserves_shape(shape):
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    y = np.arange(shape[0]) % 2
    xa, ya = augment_landmarks(x, y, copies=3, seed=1)
    assert xa.shape == (shape[0] * 4, *shape[1:])
    assert np.array_equal(xa[: shape[0]], x)  # real rows untouched
    assert len(ya) == len(xa)


def test_augmentation_rejects_unknown_width():
    with pytest.raises(ValueError):
        augment_landmarks(np.zeros((2, 66)), np.zeros(2), copies=1, seed=0)


# --- decision policy --------------------------------------------------------

def test_default_threshold_is_thirty_percent():
    assert DEFAULT_TARGET_THRESHOLD == 0.30


def test_policy_accepts_top1_target_above_threshold_with_margin():
    assert accept_target([0.35, 0.25, 0.20, 0.20], 0, target_threshold=0.30, min_margin=0.06)


def test_policy_rejects_recognizable_wrong_sign_even_above_threshold():
    # Target has 35% (> 30%) but another known sign is top-1.
    assert not accept_target([0.35, 0.60, 0.05], 0)


def test_policy_rejects_insufficient_margin_and_low_probability():
    assert not accept_target([0.33, 0.30, 0.37 - 0.0], 2, min_margin=0.10)
    assert not accept_target([0.29, 0.28, 0.27, 0.16], 0)


def test_calibration_reports_false_acceptance():
    probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    targets = np.array([0, 1, 2])
    policy = calibrate_decision_policy(probs, targets, ["A", "B", "C"])
    assert policy["target_threshold"] == 0.30 and policy["require_target_is_top1"]
    stats = evaluate_decision_policy(probs, targets, ["A", "B", "C"], policy)
    assert stats["correct_sign_acceptance_rate"] == pytest.approx(2 / 3)
    # Row 3 (true C) would be accepted if the learner had been asked for B.
    assert stats["wrong_sign_false_acceptance_rate"] == pytest.approx(1 / 6)


# --- CVI gate ---------------------------------------------------------------

def _manifest(tmp_path, items):
    path = tmp_path / "cvi.json"
    path.write_text(json.dumps({"experts": ["E1", "E2"], "items": items}))
    return path


def test_cvi_requires_manifest_unless_development_bypass():
    with pytest.raises(SystemExit):
        require_expert_cvi(None, ["A"], allow_unvalidated=False)
    assert require_expert_cvi(None, ["A"], allow_unvalidated=True)["status"] == "unvalidated_development_only"


def test_cvi_requires_both_experts_per_label(tmp_path):
    ok = [{"label": l, "expert_id": e, "approved": True} for l in "AB" for e in ("E1", "E2")]
    assert require_expert_cvi(_manifest(tmp_path, ok), ["A", "B"], allow_unvalidated=False)["status"] == "expert_validated"
    with pytest.raises(SystemExit, match="B"):
        require_expert_cvi(_manifest(tmp_path, ok[:3]), ["A", "B"], allow_unvalidated=False)


# --- two-hand ordering ------------------------------------------------------

def _hand(x):
    return [SimpleNamespace(x=x, y=0.5, z=0.0)] * 21


def test_two_hand_ordering_is_stable():
    left, right = (_hand(0.7), "Left", 0.6), (_hand(0.3), "Right", 0.9)
    assert [d[1] for d in order_two_hands([right, left])] == ["Left", "Right"]
    a, b = (_hand(0.7), "Left", 0.9), (_hand(0.3), "Left", 0.6)
    assert order_two_hands([a, b])[0] is b and order_two_hands([b, a])[0] is b


# --- contract, bundle, installer -------------------------------------------

def test_static_two_hand_contract_is_not_temporal():
    contract = build_input_contract([f"h{h}_{a}{i}" for h in range(2) for i in range(21) for a in "xyz"])
    assert contract["input_shape"] == [1, 126] and "sequence_length" not in contract


@pytest.mark.parametrize("hands", [1, 2])
def test_dynamic_contract_matches_sequence_width(tmp_path, hands):
    dataset = load_dataset(write_csv(tmp_path, hands=hands, sign_type="dynamic"), "label", "participant_id")
    sequences = build_sequences(dataset, 20)
    contract = build_input_contract(dataset.feature_columns, sequence_length=20)
    assert contract["input_shape"] == [1, *sequences.features.shape[1:]]
    assert contract["coordinate_order"][-4:] == list(TRAJECTORY_COLUMNS)
    assert len(contract["coordinate_order"]) == contract["feature_count"]


def _bundle(tmp_path, hands, sequence_length, cvi_status):
    columns = [(f"h{h}_" if hands == 2 else "") + f"{a}{i}" for h in range(hands) for i in range(21) for a in "xyz"]
    contract = build_input_contract(columns, sequence_length=sequence_length)
    policy = calibrate_decision_policy(np.array([[0.8, 0.2], [0.3, 0.7]]), np.array([0, 1]), ["A", "B"])
    metadata = write_training_metadata(
        tmp_path / "model_metadata.json", dataset_summary={}, feature_columns=columns,
        quantization="float16", metrics={"accuracy": 0.0, "macro_f1": 0.0},
        verification={"input_shape": contract["input_shape"]},
        sequence_length=sequence_length, decision_policy=policy, cvi={"status": cvi_status},
    )
    model = tmp_path / "m.tflite"
    model.write_bytes(b"x")
    return create_mobile_bundle(tmp_path, model, ["A", "B"], metadata), metadata


@pytest.mark.parametrize("hands,length,slot", [
    (1, None, "static_single"), (2, None, "static_both"),
    (1, 24, "dynamic_single"), (2, 24, "dynamic_both"),
])
def test_bundles_fill_four_independent_slots(tmp_path, hands, length, slot):
    import install_mobile_bundle

    bundle, metadata = _bundle(tmp_path, hands, length, "expert_validated")
    assert metadata["slot"] == slot and metadata["model_kind"] == MODEL_SLOTS[slot]["model_kind"]
    assert (bundle / model_filename(slot)).is_file() and (bundle / manifest_filename(slot)).is_file()
    _, _, manifest = install_mobile_bundle.validate(bundle)
    assert manifest["release_ready"] is True


def test_installer_rejects_development_bundle_by_default(tmp_path):
    import install_mobile_bundle

    bundle, _ = _bundle(tmp_path, 1, None, "unvalidated_development_only")
    with pytest.raises(SystemExit, match="release_ready"):
        install_mobile_bundle.validate(bundle)
    install_mobile_bundle.validate(bundle, allow_development=True)


def test_metadata_rejects_shape_mismatch(tmp_path):
    columns = [f"{a}{i}" for i in range(21) for a in "xyz"]
    with pytest.raises(SystemExit, match="input shape"):
        write_training_metadata(
            tmp_path / "m.json", dataset_summary={}, feature_columns=columns,
            quantization="none", metrics={"accuracy": 0, "macro_f1": 0},
            verification={"input_shape": [1, 24, 63]}, sequence_length=24,
        )
