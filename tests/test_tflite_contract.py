"""Export-shape smoke test on random tensors (not training, no real data)."""

import numpy as np
import pytest

from fslames_ml.bundle import build_input_contract, verify_contract_shape
from fslames_ml.modeling import build_model, build_sequence_model, export_tflite, verify_tflite

tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("hands,length", [(1, None), (2, None), (1, 24), (2, 24)])
def test_exported_tflite_input_matches_contract(tmp_path, hands, length):
    columns = [(f"h{h}_" if hands == 2 else "") + f"{a}{i}" for h in range(hands) for i in range(21) for a in "xyz"]
    contract = build_input_contract(columns, sequence_length=length)
    x = np.random.default_rng(0).normal(size=(8, *contract["input_shape"][1:])).astype(np.float32)
    builder = build_model if length is None else build_sequence_model
    model = builder(tf, x, 3, 0)
    path = tmp_path / "m.tflite"
    export_tflite(tf, model, path, "float16")
    verification = verify_tflite(tf, model, path, x[0])
    verify_contract_shape(contract, verification)
    assert verification["output_shape"] == [1, 3]
