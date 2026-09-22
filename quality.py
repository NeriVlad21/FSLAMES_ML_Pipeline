from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from .errors import fail


REQUIRED_COLUMNS = {
    "label", "participant_id", "source_file", "frame_index", "sign_type",
    "required_hands", "handedness", "detection_score", "expected_hand",
    "mean_luma", "hand_center_x", "hand_center_y", "palm_size",
}


@dataclass(frozen=True)
class QualitySummary:
    participants: int
    labels: int
    sources: int
    rows: int
    sign_type: str
    required_hands: int
    source_quality_floor: float
    handedness_match_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataset_quality(
    frame: pd.DataFrame,
    *,
    minimum_participants: int = 5,
    minimum_source_quality: float = 0.80,
    minimum_dynamic_frames: int = 8,
) -> QualitySummary:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        fail(
            "Dataset is missing consistency columns. Re-extract it with the updated "
            "extract_landmarks.py. Missing: " + ", ".join(sorted(missing))
        )
    participants = sorted(frame["participant_id"].astype(str).unique())
    labels = sorted(frame["label"].astype(str).unique())
    if len(participants) < minimum_participants:
        fail(f"At least {minimum_participants} participants are required.")

    sign_types = set(frame["sign_type"].astype(str).str.lower())
    hand_counts = set(pd.to_numeric(frame["required_hands"], errors="coerce"))
    if len(sign_types) != 1 or sign_types - {"static", "dynamic"}:
        fail("Each CSV must contain exactly one valid sign_type: static or dynamic.")
    if len(hand_counts) != 1 or hand_counts - {1, 2}:
        fail("Each CSV must contain exactly one required_hands value: 1 or 2.")
    sign_type = next(iter(sign_types))
    required_hands = int(next(iter(hand_counts)))

    source_meta = frame.groupby("source_file").agg(
        labels=("label", "nunique"),
        participants=("participant_id", "nunique"),
        roles=("expected_hand", "nunique"),
        frames=("frame_index", "count"),
    )
    if (source_meta[["labels", "participants", "roles"]] != 1).any().any():
        fail("Every source file must belong to one label, participant, and hand role.")
    if sign_type == "dynamic" and (source_meta["frames"] < minimum_dynamic_frames).any():
        bad = list(source_meta.index[source_meta["frames"] < minimum_dynamic_frames][:10])
        fail(
            f"Dynamic videos need at least {minimum_dynamic_frames} detected frames. "
            f"Insufficient sources: {bad}"
        )

    sources = frame[["label", "participant_id", "source_file", "expected_hand"]].drop_duplicates()
    expected_roles = {"left", "right"} if required_hands == 1 else {"both"}
    for label in labels:
        for participant in participants:
            subset = sources[
                (sources["label"].astype(str) == label)
                & (sources["participant_id"].astype(str) == participant)
            ]
            roles = list(subset["expected_hand"].astype(str).str.lower())
            if set(roles) != expected_roles or len(roles) != len(expected_roles):
                fail(
                    f"{label}/{participant} must contain exactly "
                    f"{sorted(expected_roles)}; found {sorted(roles)}."
                )

    numeric = frame.copy()
    for column in (
        "detection_score", "mean_luma", "hand_center_x", "hand_center_y", "palm_size"
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    quality = (
        numeric["detection_score"].ge(0.50)
        & numeric["mean_luma"].between(45.0, 225.0)
        & numeric["hand_center_x"].between(0.15, 0.85)
        & numeric["hand_center_y"].between(0.15, 0.85)
        & numeric["palm_size"].between(0.04, 0.45)
    )
    rates = quality.groupby(frame["source_file"]).mean()
    if (rates < minimum_source_quality).any():
        bad = {
            source: round(float(rate), 3)
            for source, rate in rates[rates < minimum_source_quality].head(10).items()
        }
        fail(
            "At least 80% of sampled frames in every source must satisfy controlled "
            f"lighting, centering, distance, and confidence checks. Failed: {bad}"
        )

    single = frame["expected_hand"].astype(str).str.lower().isin({"left", "right"})
    if single.any():
        detected = frame.loc[single, "handedness"].astype(str).str.lower()
        expected = frame.loc[single, "expected_hand"].astype(str).str.lower()
        handedness_match = float(
            sum(e in d for e, d in zip(expected, detected)) / max(1, len(expected))
        )
    else:
        handedness_match = 1.0
    return QualitySummary(
        participants=len(participants),
        labels=len(labels),
        sources=len(sources),
        rows=len(frame),
        sign_type=sign_type,
        required_hands=required_hands,
        source_quality_floor=float(rates.min()),
        handedness_match_rate=handedness_match,
    )
