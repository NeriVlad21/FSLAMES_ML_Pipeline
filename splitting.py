from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import fail


@dataclass(frozen=True)
class SplitSummary:
    training_rows: int
    validation_rows: int
    testing_rows: int
    grouped_by_source: bool
    source_column: str | None
    grouping: str


@dataclass(frozen=True)
class DatasetSplits:
    training: np.ndarray
    validation: np.ndarray
    testing: np.ndarray
    summary: SplitSummary


def _split_counts(count: int, val_fraction: float, test_fraction: float) -> tuple[int, int]:
    if count < 5:
        fail("Each class or class-specific source group needs at least five samples.")
    test_count = max(1, int(round(count * test_fraction)))
    val_count = max(1, int(round(count * val_fraction)))
    while count - test_count - val_count < 2:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            fail("Not enough samples remain for a training split.")
    return val_count, test_count


def stratified_row_split(
    targets: np.ndarray, val_fraction: float, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for class_index in np.unique(targets):
        indices = np.flatnonzero(targets == class_index)
        rng.shuffle(indices)
        val_count, test_count = _split_counts(len(indices), val_fraction, test_fraction)
        test.extend(indices[:test_count])
        validation.extend(indices[test_count:test_count + val_count])
        train.extend(indices[test_count + val_count:])
    for values in (train, validation, test):
        rng.shuffle(values)
    return tuple(np.asarray(v, dtype=np.int64) for v in (train, validation, test))


def stratified_source_split(
    frame: pd.DataFrame,
    targets: np.ndarray,
    source_column: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng(seed)
    sources = frame[source_column].astype(str).to_numpy()
    group_keys = np.asarray([
        f"{int(class_index)}::{source}"
        for class_index, source in zip(targets, sources)
    ])
    train_sources: set[str] = set()
    val_sources: set[str] = set()
    test_sources: set[str] = set()
    for class_index in np.unique(targets):
        class_sources = np.unique(group_keys[targets == class_index])
        if len(class_sources) < 5:
            return None
        rng.shuffle(class_sources)
        val_count, test_count = _split_counts(len(class_sources), val_fraction, test_fraction)
        test_sources.update(class_sources[:test_count])
        val_sources.update(class_sources[test_count:test_count + val_count])
        train_sources.update(class_sources[test_count + val_count:])
    result = (
        np.flatnonzero(np.isin(group_keys, list(train_sources))),
        np.flatnonzero(np.isin(group_keys, list(val_sources))),
        np.flatnonzero(np.isin(group_keys, list(test_sources))),
    )
    train, validation, test = result
    if set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test):
        fail("Internal error: grouped dataset splits overlap.")
    return result


def participant_split(
    frame: pd.DataFrame,
    targets: np.ndarray,
    participant_column: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out entire people so testing measures unseen-signer performance."""
    participants = frame[participant_column].astype(str).to_numpy()
    unique = np.unique(participants)
    if len(unique) < 5:
        fail(
            "Participant-held-out evaluation requires at least five participants. "
            f"Found {len(unique)}."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    val_count, test_count = _split_counts(len(unique), val_fraction, test_fraction)
    test_people = unique[:test_count]
    val_people = unique[test_count:test_count + val_count]
    train_people = unique[test_count + val_count:]
    result = (
        np.flatnonzero(np.isin(participants, train_people)),
        np.flatnonzero(np.isin(participants, val_people)),
        np.flatnonzero(np.isin(participants, test_people)),
    )
    expected = set(np.unique(targets))
    for name, indices in zip(("training", "validation", "testing"), result):
        missing = expected - set(np.unique(targets[indices]))
        if missing:
            fail(
                f"The {name} participant split is missing {len(missing)} label(s). "
                "Every participant should perform every supported sign."
            )
    return result


def make_splits(
    frame: pd.DataFrame,
    targets: np.ndarray,
    source_column: str | None,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> DatasetSplits:
    if val_fraction <= 0 or test_fraction <= 0:
        fail("Validation and test fractions must both be greater than zero.")
    if val_fraction + test_fraction >= 0.8:
        fail("Validation and test fractions leave too little training data.")
    grouped = False
    grouping = "frame"
    split = None
    if source_column:
        if source_column.lower() in {
            "participant_id", "participant", "subject_id", "signer_id"
        }:
            split = participant_split(
                frame, targets, source_column, val_fraction, test_fraction, seed
            )
            grouped = True
            grouping = "participant"
        else:
            split = stratified_source_split(
                frame, targets, source_column, val_fraction, test_fraction, seed
            )
            grouped = split is not None
            if grouped:
                grouping = "source"
    if split is None:
        print(
            "WARNING: using a frame-level split. Add a source_file/sample_id column "
            "and at least five recordings per label for defensible evaluation.",
            file=sys.stderr,
        )
        split = stratified_row_split(targets, val_fraction, test_fraction, seed)
    train, validation, test = split
    return DatasetSplits(
        training=train,
        validation=validation,
        testing=test,
        summary=SplitSummary(
            training_rows=len(train),
            validation_rows=len(validation),
            testing_rows=len(test),
            grouped_by_source=grouped,
            source_column=source_column if grouped else None,
            grouping=grouping,
        ),
    )
