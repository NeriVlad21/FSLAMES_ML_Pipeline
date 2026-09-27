from __future__ import annotations

import numpy as np


def calibrate_decision_policy(
    probabilities: np.ndarray,
    targets: np.ndarray,
    labels: list[str],
    *,
    leniency: float = 0.10,
    wrong_sign_margin: float = 0.06,
) -> dict:
    """Lower correct-sign thresholds without weakening wrong-class rejection."""
    thresholds: dict[str, float] = {}
    for index, label in enumerate(labels):
        class_rows = probabilities[targets == index]
        own = class_rows[:, index]
        correct = own[np.argmax(class_rows, axis=1) == index]
        baseline = float(np.quantile(correct, 0.10)) if len(correct) else 0.50
        thresholds[label] = round(max(0.05, baseline - leniency), 4)
    return {
        "mode": "target_top1_with_margin",
        "per_class_thresholds": thresholds,
        "leniency_percentage_points": int(round(leniency * 100)),
        "minimum_top1_margin": wrong_sign_margin,
        "reject_if_predicted_label_differs_from_target": True,
    }
