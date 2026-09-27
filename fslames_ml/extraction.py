from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class MediaSample:
    path: Path
    label: str
    participant_id: str
    source_file: str
    expected_hand: str


def infer_expected_hand(path: Path, num_hands: int, label: str = "") -> str:
    stem = path.stem.lower().replace("-", "_")
    # Drop the label prefix so labels like RIGHT or NO_LEFT_TURN aren't read as roles.
    prefix = label.lower().replace("-", "_") + "_"
    if label and stem.startswith(prefix):
        stem = stem[len(prefix):]
    tokens = stem.split("_")
    matches = [name for name in ("left", "right", "both") if name in tokens]
    if len(matches) != 1:
        raise SystemExit(
            f"ERROR: filename must contain exactly one hand role (left/right/both): {path}"
        )
    expected = matches[0]
    if num_hands == 1 and expected not in {"left", "right"}:
        raise SystemExit(f"ERROR: one-hand extraction requires left/right filename: {path}")
    if num_hands == 2 and expected != "both":
        raise SystemExit(f"ERROR: two-hand extraction requires a 'both' filename: {path}")
    return expected


def discover_samples(input_dir: Path, layout: str, num_hands: int) -> list[MediaSample]:
    """Find media stored as label/participant or participant/label."""
    if not input_dir.is_dir():
        raise SystemExit(f"ERROR: input directory does not exist: {input_dir}")
    samples: list[MediaSample] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(input_dir)
        if len(relative.parts) < 3:
            raise SystemExit(
                "ERROR: every file must be inside two folders: label/participant/file "
                "or participant/label/file."
            )
        if layout == "label/participant":
            label, participant = relative.parts[0], relative.parts[1]
        else:
            participant, label = relative.parts[0], relative.parts[1]
        samples.append(MediaSample(
            path,
            label.strip(),
            participant.strip(),
            relative.as_posix(),
            infer_expected_hand(path, num_hands, label.strip()),
        ))
    if not samples:
        raise SystemExit("ERROR: no supported videos or images were found.")
    if any(not sample.label or not sample.participant_id for sample in samples):
        raise SystemExit("ERROR: a sample has an empty label or participant identifier.")
    return samples


def csv_header(num_hands: int) -> list[str]:
    header = [
        "label", "participant_id", "source_file", "frame_index", "timestamp_ms",
        "sign_type", "required_hands", "handedness", "detection_score",
        "expected_hand", "mean_luma", "hand_center_x", "hand_center_y", "palm_size",
    ]
    for hand in range(num_hands):
        prefix = "" if num_hands == 1 else f"h{hand}_"
        for landmark in range(21):
            header.extend(
                (f"{prefix}x{landmark}", f"{prefix}y{landmark}", f"{prefix}z{landmark}")
            )
    return header


def wrist_relative_features(hands: Sequence[Sequence[object]]) -> list[float]:
    """Normalize every hand against the first hand's wrist.

    For two-hand samples this preserves the relative position between hands.
    """
    if not hands or any(len(hand) != 21 for hand in hands):
        raise ValueError("Each detected hand must contain exactly 21 landmarks.")
    origin = hands[0][0]
    output: list[float] = []
    for hand in hands:
        for point in hand:
            output.extend(
                (
                    float(point.x - origin.x),
                    float(point.y - origin.y),
                    float(point.z - origin.z),
                )
            )
    return output


def selected_detections(result, num_hands: int):
    if len(result.hand_landmarks) < num_hands:
        return None
    detected = []
    for index, landmarks in enumerate(result.hand_landmarks):
        categories = result.handedness[index] if index < len(result.handedness) else []
        category = categories[0] if categories else None
        name = category.category_name if category else "Unknown"
        score = float(category.score) if category else 0.0
        detected.append((landmarks, name, score))
    detected.sort(
        key=lambda item: ({"Left": 0, "Right": 1}.get(item[1], 2), -item[2])
    )
    if num_hands == 1:
        detected.sort(key=lambda item: item[2], reverse=True)
    return detected[:num_hands]


def capture_quality(cv2, frame, detections) -> tuple[float, float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_luma = float(gray.mean())
    points = [point for detection in detections for point in detection[0]]
    center_x = float(sum(point.x for point in points) / len(points))
    center_y = float(sum(point.y for point in points) / len(points))
    primary = detections[0][0]
    dx = float(primary[9].x - primary[0].x)
    dy = float(primary[9].y - primary[0].y)
    palm_size = float((dx * dx + dy * dy) ** 0.5)
    return mean_luma, center_x, center_y, palm_size


def iter_video_frames(cv2, path: Path, frame_skip: int) -> Iterable[tuple[int, int, object]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % frame_skip != 0:
                continue
            timestamp_ms = int(round(frame_index * 1000.0 / fps))
            yield frame_index, timestamp_ms, frame
    finally:
        capture.release()
