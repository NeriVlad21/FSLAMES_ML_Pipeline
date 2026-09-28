"""Target-aware acceptance policy for "perform sign X" practice attempts.

An attempt at a requested (target) sign is accepted only when all hold:

1. the target sign is the model's top-1 prediction,
2. the target probability is at least ``target_threshold`` (default 0.30),
3. the target probability beats the nearest competing known sign by at
   least ``min_margin``.

A recognizable *wrong* sign is therefore rejected even when the target still
receives more than 30% probability.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_TARGET_THRESHOLD = 0.30
DEFAULT_MIN_MARGIN = 0.06
POLICY_MODE = "target_top1_with_margin"


def _validate(threshold: float, margin: float) -> None:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("target_threshold must be in (0, 1].")
    if not 0.0 <= margin < 1.0:
        raise ValueError("min_margin must be in [0, 1).")


def target_decisions(
    probabilities: np.ndarray,
    target_indices: np.ndarray,
    *,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> np.ndarray:
    """Return one accept/reject flag per row for the requested target sign."""
    _validate(target_threshold, min_margin)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected probabilities shaped [rows, classes>=2].")
    if target_indices.shape != (probabilities.shape[0],):
        raise ValueError("Expected one target index per probability row.")
    rows = np.arange(len(target_indices))
    target = probabilities[rows, target_indices]
    competitors = probabilities.copy()
    competitors[rows, target_indices] = -np.inf
    nearest = competitors.max(axis=1)
    is_top1 = np.argmax(probabilities, axis=1) == target_indices
    return is_top1 & (target >= target_threshold) & (target - nearest >= min_margin)


def accept_target(
    probabilities: Sequence[float],
    target_index: int,
    *,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> bool:
    """Single-attempt form of :func:`target_decisions` (mirrors the app logic)."""
    return bool(
        target_decisions(
            np.asarray(probabilities)[None, :],
            np.asarray([target_index]),
            target_threshold=target_threshold,
            min_margin=min_margin,
        )[0]
    )


def evaluate_decision_policy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: Sequence[str],
    policy: dict,
) -> dict:
    """Measure correct-sign acceptance and wrong-sign false acceptance.

    Every row is scored once against its true label, and once against every
    other known label as if the learner had been asked for that sign instead.
    """
    threshold = float(policy["target_threshold"])
    margin = float(policy["minimum_top1_margin"])
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    class_count = len(labels)
    if len(targets) == 0:
        return {"rows": 0}
    correct = target_decisions(
        probabilities, targets, target_threshold=threshold, min_margin=margin
    )
    wrong_accepts = 0
    wrong_trials = 0
    for offset in range(1, class_count):
        requested = (targets + offset) % class_count
        wrong_accepts += int(
            target_decisions(
                probabilities, requested,
                target_threshold=threshold, min_margin=margin,
            ).sum()
        )
        wrong_trials += len(targets)
    per_class = {}
    for index, label in enumerate(labels):
        mask = targets == index
        if not mask.any():
            continue
        per_class[label] = {
            "rows": int(mask.sum()),
            "correct_sign_acceptance_rate": float(correct[mask].mean()),
            "median_target_probability": float(
                np.median(probabilities[mask, index])
            ),
        }
    return {
        "rows": int(len(targets)),
        "correct_sign_acceptance_rate": float(correct.mean()),
        "wrong_sign_false_acceptance_rate": float(
            wrong_accepts / max(1, wrong_trials)
        ),
        "per_class": per_class,
    }


def calibrate_decision_policy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    *,
    target_threshold: float = DEFAULT_TARGET_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> dict:
    """Build the shipped policy and record its behaviour on the validation split.

    The threshold is fixed by policy (30% by default); calibration reports how
    that threshold behaves on held-out validation participants instead of
    silently lowering it.
    """
    _validate(target_threshold, min_margin)
    policy = {
        "mode": POLICY_MODE,
        "target_threshold": float(target_threshold),
        "minimum_top1_margin": float(min_margin),
        "require_target_is_top1": True,
        "reject_if_predicted_label_differs_from_target": True,
    }
    policy["validation_calibration"] = evaluate_decision_policy(
        probabilities, targets, labels, policy
    )
    return policy
