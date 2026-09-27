from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import LoadedDataset
from .errors import fail


@dataclass(frozen=True)
class SequenceDataset:
    frame: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("Cannot resample an empty sequence.")
    indices = np.linspace(0, len(values) - 1, length).round().astype(np.int64)
    return values[indices]


def build_sequences(dataset: LoadedDataset, length: int) -> SequenceDataset:
    if length < 4:
        fail("Sequence length must be at least four frames.")
    required = {"source_file", "participant_id", "frame_index"}
    missing = required - set(dataset.frame.columns)
    if missing:
        fail("Dynamic training requires CSV columns: " + ", ".join(sorted(missing)))

    sequences: list[np.ndarray] = []
    targets: list[int] = []
    records: list[dict] = []
    grouped = dataset.frame.groupby("source_file", sort=True)
    for source_file, group in grouped:
        indices = group.sort_values("frame_index").index.to_numpy()
        group_targets = np.unique(dataset.targets[indices])
        if len(group_targets) != 1:
            fail(f"Source {source_file} contains multiple labels.")
        landmarks = _resample(dataset.features[indices], length)
        ordered = group.sort_values("frame_index")
        for column in ("hand_center_x", "hand_center_y", "palm_size"):
            if column not in ordered.columns:
                fail(
                    "Dynamic training requires extractor metadata columns: "
                    "hand_center_x, hand_center_y, palm_size."
                )
        centers = ordered[["hand_center_x", "hand_center_y"]].to_numpy(
            dtype=np.float32
        )
        palms = ordered["palm_size"].to_numpy(dtype=np.float32)
        centers = _resample(centers, length)
        palms = _resample(palms[:, None], length)[:, 0]
        scale = max(float(np.median(palms)), 1e-3)
        displacement = (centers - centers[0]) / scale
        velocity = np.vstack(
            [np.zeros((1, 2), dtype=np.float32), np.diff(displacement, axis=0)]
        )
        trajectory = np.concatenate([displacement, velocity], axis=1)
        sequences.append(
            np.concatenate([landmarks, trajectory], axis=1).astype(np.float32)
        )
        targets.append(int(group_targets[0]))
        records.append(
            {
                "source_file": source_file,
                "participant_id": str(group.iloc[0]["participant_id"]),
                "label": dataset.labels[int(group_targets[0])],
            }
        )
    frame = pd.DataFrame(records)
    target_array = np.asarray(targets, dtype=np.int32)
    counts = np.bincount(target_array, minlength=len(dataset.labels))
    too_small = [dataset.labels[i] for i, count in enumerate(counts) if count < 5]
    if too_small:
        fail(
            "Dynamic training requires at least five source videos per label. "
            "Insufficient labels: " + ", ".join(too_small)
        )
    return SequenceDataset(
        frame=frame,
        features=np.stack(sequences).astype(np.float32),
        targets=target_array,
    )
