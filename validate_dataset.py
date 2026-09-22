"""Run the FSLAMES participant, pairing, and capture-consistency gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fslames_ml.quality import validate_dataset_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an extracted FSLAMES CSV.")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"ERROR: CSV file does not exist: {args.csv}")
    summary = validate_dataset_quality(pd.read_csv(args.csv))
    print(json.dumps(summary.to_dict(), indent=2))
    if summary.handedness_match_rate < 0.8:
        print(
            "WARNING: MediaPipe handedness disagrees with filenames frequently. "
            "Check whether the source camera is mirrored; filename roles remain authoritative."
        )
    print("Dataset consistency validation: PASSED")


if __name__ == "__main__":
    main()
