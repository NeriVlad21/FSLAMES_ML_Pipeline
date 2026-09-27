from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_metrics(matrix: np.ndarray, labels: Sequence[str]) -> dict:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "accuracy": float(true_positive.sum() / max(1, matrix.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {"label": label, "support": int(actual[i]), "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(labels)
        ],
    }


def write_csv_matrix(path: Path, matrix: np.ndarray, labels: Sequence[str]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path, index_label="actual\\predicted")
