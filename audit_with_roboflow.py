"""Optional: compare folder labels with the Roboflow model fsl-fnlzs/3.

Writes a review report only. Mismatches are for a human to inspect; this tool
never relabels or modifies the source dataset. Requires ROBOFLOW_API_KEY in
the environment and `pip install -r requirements-roboflow.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from fslames_ml.errors import fail
from fslames_ml.extraction import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, discover_samples
from fslames_ml.roboflow_audit import (
    ROBOFLOW_API_URL,
    ROBOFLOW_MODEL_ID,
    make_client,
    normalize_label,
    top_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--num-hands", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--layout", choices=("label/participant", "participant/label"),
        default="label/participant",
    )
    parser.add_argument("--output", type=Path, default=Path("roboflow_audit.csv"))
    parser.add_argument("--frames-per-video", type=int, default=3)
    parser.add_argument("--model-id", default=ROBOFLOW_MODEL_ID)
    parser.add_argument("--api-url", default=ROBOFLOW_API_URL)
    parser.add_argument(
        "--label-map", type=Path, default=None,
        help="Optional JSON {dataset_label: roboflow_class} when names differ.",
    )
    return parser.parse_args()


def sampled_frames(cv2, path: Path, count: int):
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image = cv2.imread(str(path))
        return [] if image is None else [(0, image)]
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        positions = sorted({int(total * (i + 1) / (count + 1)) for i in range(count)})
        frames = []
        for position in positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if ok:
                frames.append((position, frame))
        return frames
    finally:
        capture.release()


def main() -> None:
    args = parse_args()
    input_root = args.input_dir.resolve()
    output = args.output.resolve()
    if input_root == output.parent or input_root in output.parents:
        fail("Write the audit report outside the source dataset folder.")
    if args.frames_per_video < 1:
        fail("--frames-per-video must be at least 1.")
    label_map = {}
    if args.label_map:
        label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    try:
        import cv2
    except ImportError:
        fail("opencv-python is required for the audit.")

    samples = discover_samples(args.input_dir, args.layout, args.num_hands)
    client = make_client(args.api_url)
    rows = []
    for sample in samples:
        if sample.path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        expected = normalize_label(label_map.get(sample.label, sample.label))
        for frame_index, frame in sampled_frames(cv2, sample.path, args.frames_per_video):
            try:
                predicted, confidence = top_prediction(
                    client.infer(frame, model_id=args.model_id)
                )
                error = ""
            except Exception as exc:  # network/API issues are reported, not fatal
                predicted, confidence, error = None, None, str(exc)[:200]
            rows.append({
                "source_file": sample.source_file,
                "label": sample.label,
                "participant_id": sample.participant_id,
                "frame_index": frame_index,
                "roboflow_top_class": predicted or "",
                "roboflow_confidence": "" if confidence is None else round(confidence, 4),
                "label_match": (
                    "" if predicted is None else normalize_label(predicted) == expected
                ),
                "error": error,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["source_file"])
        writer.writeheader()
        writer.writerows(rows)
    scored = [row for row in rows if row["label_match"] != ""]
    mismatches = sum(1 for row in scored if row["label_match"] is False)
    print(json.dumps({
        "model_id": args.model_id,
        "frames_checked": len(rows),
        "frames_with_prediction": len(scored),
        "label_mismatches_for_human_review": mismatches,
        "report": str(output),
    }, indent=2))
    print("Source dataset was not modified. Review mismatches manually.")


if __name__ == "__main__":
    main()
