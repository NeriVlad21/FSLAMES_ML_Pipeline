"""Optional Roboflow second-opinion audit of the raw dataset.

This is a quality-review aid only. It never relabels, moves, or edits source
media, is never used by training, and is never needed by the mobile app.
"""

from __future__ import annotations

import os
import re
from typing import Any

from .errors import fail


ROBOFLOW_MODEL_ID = "fsl-fnlzs/3"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
API_KEY_ENV = "ROBOFLOW_API_KEY"


def api_key_from_environment() -> str:
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        fail(f"Set {API_KEY_ENV} in the environment to run the optional Roboflow audit.")
    return key


def make_client(api_url: str = ROBOFLOW_API_URL):
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        fail("Roboflow audit is optional; install it with: pip install -r requirements-roboflow.txt")
    return InferenceHTTPClient(api_url=api_url, api_key=api_key_from_environment())


def top_prediction(result: Any) -> tuple[str | None, float | None]:
    """Extract (class, confidence) from a classification or detection response."""
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        return None, None
    if result.get("top"):
        return str(result["top"]), float(result.get("confidence", 0.0))
    predictions = result.get("predictions")
    if isinstance(predictions, dict):  # multi-label classification
        scored = [
            (name, float(value.get("confidence", 0.0)))
            for name, value in predictions.items()
            if isinstance(value, dict)
        ]
    elif isinstance(predictions, list):  # single-label classification / detection
        scored = [
            (str(item.get("class")), float(item.get("confidence", 0.0)))
            for item in predictions
            if isinstance(item, dict) and item.get("class") is not None
        ]
    else:
        scored = []
    if not scored:
        return None, None
    return max(scored, key=lambda pair: pair[1])


def normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
