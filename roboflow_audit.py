"""Optional Roboflow audit that never mutates or relabels the dataset."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
from inference_sdk import InferenceConfiguration, InferenceHTTPClient


def prediction(result: object) -> tuple[str | None, float | None]:
    if not isinstance(result, dict):
        return None, None
    predictions = result.get("predictions")
    if isinstance(predictions, list) and predictions:
        best = max(
            (item for item in predictions if isinstance(item, dict)),
            key=lambda item: float(item.get("confidence", 0)),
            default=None,
        )
        if best is not None:
            label = best.get("class") or best.get("label")
            confidence = best.get("confidence")
            return (
                str(label) if label is not None else None,
                float(confidence) if confidence is not None else None,
            )
    return None, None


def sampled_frames(path: Path, count: int):
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        yield path, None
        return
    capture = cv2.VideoCapture(str(path))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = sorted({round(i * (total - 1) / max(1, count - 1)) for i in range(count)})
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            yield frame, index
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw FSL media with Roboflow.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("roboflow_audit.json"))
    parser.add_argument("--model-id", default="fsl-fnlzs/3")
    parser.add_argument("--video-samples", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set ROBOFLOW_API_KEY; never commit the key.")
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com", api_key=api_key
    ).configure(InferenceConfiguration(api_key_transport="header"))
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi"}
    records = []
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        for path in sorted(p for p in args.root.rglob("*") if p.suffix.lower() in suffixes):
            expected = path.relative_to(args.root).parts[0]
            for image, frame_index in sampled_frames(path, args.video_samples):
                request_path = image
                if frame_index is not None:
                    request_path = temp / f"frame_{len(records)}.jpg"
                    cv2.imwrite(str(request_path), image)
                label, confidence = prediction(
                    client.infer(str(request_path), model_id=args.model_id)
                )
                records.append({
                    "source_file": str(path),
                    "frame_index": frame_index,
                    "expected_label": expected,
                    "roboflow_label": label,
                    "confidence": confidence,
                    "agreement": label is not None and label.casefold() == expected.casefold(),
                    "review_required": label is None or label.casefold() != expected.casefold(),
                })
    args.output.write_text(json.dumps({
        "purpose": "teacher_audit_only_no_automatic_relabeling",
        "model_id": args.model_id,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} audit rows to {args.output}")


if __name__ == "__main__":
    main()
