from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import fail


FEATURE_PATTERN = re.compile(
    r"^(?:h(?P<hand>\d+)_)?(?P<axis>[xyz])(?P<landmark>\d+)$",
    re.IGNORECASE,
)
SOURCE_CANDIDATES = (
    "participant_id", "participant", "subject_id", "signer_id",
    "source_id", "source", "source_file", "video", "video_name",
    "filename", "file", "sample_id",
)


@dataclass(frozen=True)
class LoadedDataset:
    frame: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray
    labels: list[str]
    feature_columns: list[str]
    source_column: str | None
    hand_count: int


def _feature_key(name: str) -> tuple[int, int, int]:
    match = FEATURE_PATTERN.fullmatch(name.strip())
    if match is None:
        raise ValueError(name)
    hand = int(match.group("hand") or 0)
    landmark = int(match.group("landmark"))
    axis = {"x": 0, "y": 1, "z": 2}[match.group("axis").lower()]
    return hand, landmark, axis


def discover_feature_columns(columns: Sequence[str]) -> list[str]:
    """Return landmarks in the exact order used by the mobile encoder."""
    features = [name for name in columns if FEATURE_PATTERN.fullmatch(name.strip())]
    if not features:
        fail("No landmark columns were found. Expected x0,y0,z0,...,x20,y20,z20.")
    if len(set(features)) != len(features):
        fail("Duplicate landmark columns were found in the CSV header.")
    features.sort(key=_feature_key)
    keys = [_feature_key(name) for name in features]
    hands = sorted({key[0] for key in keys})
    if hands != list(range(len(hands))):
        fail("Hand prefixes must be contiguous (h0_, h1_).")
    for hand in hands:
        expected = {(hand, landmark, axis) for landmark in range(21) for axis in range(3)}
        actual = {key for key in keys if key[0] == hand}
        if actual != expected:
            fail(f"Hand {hand} must contain x/y/z for landmarks 0 through 20.")
    if len(hands) not in (1, 2):
        fail("Only one-hand (63 features) and two-hand (126 features) CSVs are supported.")
    return features


def detect_source_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            fail(f"Source column '{requested}' does not exist in the CSV.")
        return requested
    lowered = {column.lower(): column for column in frame.columns}
    for candidate in SOURCE_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_dataset(
    csv_path: Path,
    label_column: str,
    requested_source_column: str | None,
) -> LoadedDataset:
    if not csv_path.is_file():
        fail(f"CSV file does not exist: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        fail("The CSV contains no data rows.")
    if label_column not in frame.columns:
        fail(f"Label column '{label_column}' does not exist in the CSV.")
    feature_columns = discover_feature_columns(list(frame.columns))
    source_column = detect_source_column(frame, requested_source_column)
    label_series = frame[label_column].astype(str).str.strip()
    if label_series.eq("").any():
        fail("One or more rows have an empty label.")
    numeric = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    invalid_rows = ~np.isfinite(values).all(axis=1)
    if invalid_rows.any():
        examples = list(frame.index[invalid_rows][:10])
        fail(f"Landmarks contain missing/non-numeric values at row indices: {examples}")
    class_counts = label_series.value_counts().sort_index()
    if len(class_counts) < 2:
        fail("At least two gesture labels are required for classification.")
    too_small = class_counts[class_counts < 5]
    if not too_small.empty:
        fail("Every class needs at least five rows: " + ", ".join(
            f"{label}={count}" for label, count in too_small.items()
        ))
    classes = sorted(class_counts.index.tolist(), key=str.casefold)
    class_to_index = {label: index for index, label in enumerate(classes)}
    targets = label_series.map(class_to_index).to_numpy(dtype=np.int32)
    return LoadedDataset(
        frame=frame,
        features=values,
        targets=targets,
        labels=classes,
        feature_columns=feature_columns,
        source_column=source_column,
        hand_count=len(feature_columns) // 63,
    )
