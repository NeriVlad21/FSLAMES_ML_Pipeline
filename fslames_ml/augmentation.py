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
    if features.ndim not in (2, 3):
        raise ValueError("Expected [rows, features] or [rows, time, features].")
    landmark_width = (features.shape[-1] // 63) * 63
    trajectory_width = features.shape[-1] - landmark_width
    if landmark_width not in (63, 126) or trajectory_width not in (0, 4):
        raise ValueError(
            "Expected 63/126 landmarks, optionally followed by dx,dy,vx,vy."
        )
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
            for offset in range(0, landmark_width, 3):
                x = flat[sample, :, offset].copy()
                y = flat[sample, :, offset + 1].copy()
                flat[sample, :, offset] = (
                    x * cosines[sample] - y * sines[sample]
                )
                flat[sample, :, offset + 1] = (
                    x * sines[sample] + y * cosines[sample]
                )
            if trajectory_width:
                trajectory = flat[sample, :, landmark_width:]
                for offset in (0, 2):
                    tx = trajectory[:, offset].copy()
                    ty = trajectory[:, offset + 1].copy()
                    trajectory[:, offset] = (
                        tx * cosines[sample] - ty * sines[sample]
                    )
                    trajectory[:, offset + 1] = (
                        tx * sines[sample] + ty * cosines[sample]
                    )
        noise = rng.normal(0.0, 0.004, size=augmented.shape).astype(np.float32)
        if trajectory_width:
            noise[..., landmark_width:] *= 0.5
        augmented += noise
        batches.append(augmented)
        labels.append(targets.copy())
    return np.concatenate(batches, axis=0), np.concatenate(labels, axis=0)
