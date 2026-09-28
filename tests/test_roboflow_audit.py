import pytest

from fslames_ml import roboflow_audit


def test_top_prediction_handles_classification_and_detection():
    assert roboflow_audit.top_prediction({"top": "A", "confidence": 0.8}) == ("A", 0.8)
    assert roboflow_audit.top_prediction(
        {"predictions": [{"class": "B", "confidence": 0.4}, {"class": "C", "confidence": 0.7}]}
    ) == ("C", 0.7)
    assert roboflow_audit.top_prediction({"predictions": {"D": {"confidence": 0.9}}}) == ("D", 0.9)
    assert roboflow_audit.top_prediction({"predictions": []}) == (None, None)


def test_api_key_only_from_environment(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ROBOFLOW_API_KEY"):
        roboflow_audit.api_key_from_environment()
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-key")
    assert roboflow_audit.api_key_from_environment() == "test-key"


def test_model_id():
    assert roboflow_audit.ROBOFLOW_MODEL_ID == "fsl-fnlzs/3"


def test_core_pipeline_does_not_import_roboflow():
    import subprocess, sys
    code = (
        "import sys, fslames_ml.bundle, fslames_ml.calibration, fslames_ml.sequences, "
        "fslames_ml.validation; assert 'inference_sdk' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
