"""Synthetic CSVs that follow the extractor schema exactly (no real data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fslames_ml.extraction import csv_header


def make_frame(
    *,
    hands: int,
    sign_type: str,
    labels=("A", "B", "C"),
    participants=5,
    frames_per_source=None,
    seed=0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames_per_source = frames_per_source or (12 if sign_type == "dynamic" else 3)
    roles = ["right", "left"] if hands == 1 else ["both"]
    header = csv_header(hands)
    rows = []
    for label_index, label in enumerate(labels):
        for p in range(1, participants + 1):
            participant = f"P{p:02d}"
            for role in roles:
                source = f"{label}/{participant}/{label}_{role}.mp4"
                base = rng.normal(label_index, 0.05, size=63 * hands)
                for frame in range(frames_per_source):
                    progress = frame / max(1, frames_per_source - 1)
                    rows.append([
                        label, participant, source, frame + 1, frame * 33,
                        sign_type, hands,
                        "Right" if role == "right" else "Left" if role == "left" else "Left+Right",
                        0.95, role, 120.0,
                        0.4 + 0.1 * progress * (label_index + 1), 0.5, 0.12,
                        *(base + rng.normal(0, 0.01, size=63 * hands)),
                    ])
    return pd.DataFrame(rows, columns=header)


def write_csv(tmp_path: Path, **kwargs) -> Path:
    path = tmp_path / "data.csv"
    make_frame(**kwargs).to_csv(path, index=False)
    return path
