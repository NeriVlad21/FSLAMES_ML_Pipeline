from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fslames_ml.augmentation import augment_landmarks
from fslames_ml.calibration import calibrate_decision_policy
from fslames_ml.data import LoadedDataset
from fslames_ml.sequences import build_sequences
from fslames_ml.validation import require_expert_cvi


class PipelineContractTests(unittest.TestCase):
    def test_dynamic_sequence_keeps_trajectory(self) -> None:
        rows, values, targets = [], [], []
        for label_index, label in enumerate(("HELLO", "THANKS")):
            for participant in range(5):
                source = f"{label}_{participant}.mp4"
                for frame in range(6):
                    rows.append({
                        "label": label,
                        "participant_id": f"p{participant}",
                        "source_file": source,
                        "frame_index": frame,
                        "hand_center_x": 0.45 + frame * 0.01,
                        "hand_center_y": 0.50,
                        "palm_size": 0.15,
                    })
                    values.append(np.full(63, frame, dtype=np.float32))
                    targets.append(label_index)
        dataset = LoadedDataset(
            frame=pd.DataFrame(rows),
            features=np.stack(values),
            targets=np.asarray(targets, dtype=np.int32),
            labels=["HELLO", "THANKS"],
            feature_columns=[f"{axis}{index}" for index in range(21) for axis in "xyz"],
            source_column="participant_id",
            hand_count=1,
        )
        result = build_sequences(dataset, 4)
        self.assertEqual(result.features.shape, (10, 4, 67))
        self.assertGreater(result.features[0, -1, -4], 0)

    def test_sequence_augmentation_accepts_trajectory_features(self) -> None:
        features = np.zeros((2, 24, 67), dtype=np.float32)
        augmented, labels = augment_landmarks(
            features, np.asarray([0, 1]), copies=1, seed=2026
        )
        self.assertEqual(augmented.shape, (4, 24, 67))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])

    def test_calibration_is_lenient_but_target_aware(self) -> None:
        policy = calibrate_decision_policy(
            np.asarray([[0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.2, 0.8]]),
            np.asarray([0, 0, 1, 1]),
            ["A", "B"],
        )
        self.assertEqual(policy["leniency_percentage_points"], 10)
        self.assertTrue(policy["reject_if_predicted_label_differs_from_target"])

    def test_cvi_requires_two_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cvi.json"
            path.write_text(json.dumps({
                "experts": ["E1", "E2"],
                "items": [
                    {"label": "A", "expert_id": "E1", "approved": True},
                    {"label": "A", "expert_id": "E2", "approved": True},
                ],
            }), encoding="utf-8")
            result = require_expert_cvi(path, ["A"], allow_unvalidated=False)
            self.assertEqual(result["status"], "expert_validated")


if __name__ == "__main__":
    unittest.main()
