from __future__ import annotations

import numpy as np


def augment_landmarks(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    copies: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create mild, label-preserving landmark variations for training only.

    One transform is applied consistently to all hands in a sample. Validation
    and test samples must never be passed to this function.
    """
    if copies <= 0:
        return features, targets
    if features.ndim not in (2, 3) or features.shape[-1] % 63 != 0:
        raise ValueError("Expected [rows, features] or [rows, time, features].")
    rng = np.random.default_rng(seed)
    batches = [features.astype(np.float32, copy=False)]
    labels = [targets]
    for _ in range(copies):
        augmented = features.astype(np.float32, copy=True)
        sample_count = augmented.shape[0]
        angles = rng.uniform(-0.12, 0.12, size=sample_count)
        scales = rng.uniform(0.92, 1.08, size=sample_count)
        cosines = np.cos(angles) * scales
        sines = np.sin(angles) * scales
        flat = augmented.reshape(sample_count, -1, augmented.shape[-1])
        for sample in range(sample_count):
            row = flat[sample].reshape(-1, 3)
            x = row[:, 0].copy()
            y = row[:, 1].copy()
            row[:, 0] = x * cosines[sample] - y * sines[sample]
            row[:, 1] = x * sines[sample] + y * cosines[sample]
        augmented += rng.normal(0.0, 0.004, size=augmented.shape).astype(np.float32)
        batches.append(augmented)
        labels.append(targets.copy())
    return np.concatenate(batches, axis=0), np.concatenate(labels, axis=0)
