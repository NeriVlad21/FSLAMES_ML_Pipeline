"""Extract app-compatible wrist-relative MediaPipe landmarks from media."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fslames_ml.extraction import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    csv_header,
    capture_quality,
    discover_samples,
    iter_video_frames,
    selected_detections,
    wrist_relative_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract static one-hand landmarks from labeled videos/images."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--model", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--output", type=Path, default=Path("fslames_landmarks.csv"))
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--num-hands", type=int, choices=(1, 2), required=True)
    parser.add_argument("--sign-type", choices=("static", "dynamic"), required=True)
    parser.add_argument(
        "--layout",
        choices=("label/participant", "participant/label"),
        default="label/participant",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"ERROR: MediaPipe model does not exist: {args.model}")
    if args.frame_skip < 1:
        raise SystemExit("ERROR: --frame-skip must be at least 1.")
    if not 0 < args.min_confidence <= 1:
        raise SystemExit("ERROR: --min-confidence must be in (0, 1].")

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise SystemExit(
            "ERROR: install tools/ml/requirements.txt before extraction."
        ) from exc

    samples = discover_samples(args.input_dir, args.layout, args.num_hands)
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )
    written = 0
    missing = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with args.output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(csv_header(args.num_hands))
            for sample in samples:
                if sample.path.suffix.lower() in VIDEO_EXTENSIONS:
                    frames = iter_video_frames(cv2, sample.path, args.frame_skip)
                elif sample.path.suffix.lower() in IMAGE_EXTENSIONS:
                    image = cv2.imread(str(sample.path))
                    if image is None:
                        print(f"WARNING: could not read image: {sample.path}")
                        continue
                    frames = [(0, 0, image)]
                else:
                    continue
                for frame_index, timestamp_ms, frame in frames:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    detections = selected_detections(
                        landmarker.detect(mp_image), args.num_hands
                    )
                    if detections is None:
                        missing += 1
                        continue
                    landmarks = [item[0] for item in detections]
                    handedness = "+".join(item[1] for item in detections)
                    score = min(item[2] for item in detections)
                    mean_luma, center_x, center_y, palm_size = capture_quality(
                        cv2, frame, detections
                    )
                    writer.writerow([
                        sample.label,
                        sample.participant_id,
                        sample.source_file,
                        frame_index,
                        timestamp_ms,
                        args.sign_type,
                        args.num_hands,
                        handedness,
                        score,
                        sample.expected_hand,
                        mean_luma,
                        center_x,
                        center_y,
                        palm_size,
                        *wrist_relative_features(landmarks),
                    ])
                    written += 1

    print(f"Wrote {written} landmark rows to {args.output}")
    print(f"Skipped {missing} sampled frames with no detected hand")
    print(
        f"Schema: {args.sign_type}, {args.num_hands} hand(s), "
        f"layout={args.layout}"
    )


if __name__ == "__main__":
    main()
