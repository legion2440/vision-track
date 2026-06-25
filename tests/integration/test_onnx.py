from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vision_track.detector import OnnxRuntimeBackend
from vision_track.device import DeviceInfo


pytestmark = pytest.mark.integration


def test_onnx_runtime_inference(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    model_path = tmp_path / "constant_detection.onnx"
    input_info = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    output_info = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 1, 6]
    )
    constant = numpy_helper.from_array(
        np.array([[[10, 10, 20, 20, 0.9, 0]]], dtype=np.float32),
        name="detections",
    )
    node = helper.make_node("Constant", inputs=[], outputs=["output0"], value=constant)
    graph = helper.make_graph([node], "constant_detector", [input_info], [output_info])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.save(model, model_path)

    backend = OnnxRuntimeBackend(
        model_path,
        DeviceInfo("cpu", "cpu", "CPU", "ONNX Runtime CPU"),
        image_size=64,
        confidence=0.35,
        iou=0.5,
    )
    result = backend.infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(result.detections) == 1
    assert result.device == "cpu"

