from __future__ import annotations

import numpy as np

from vision_track.detector import _postprocess_output
from vision_track.preprocessing import LetterboxInfo


def _info() -> LetterboxInfo:
    return LetterboxInfo(
        original_shape=(64, 64),
        input_shape=(64, 64),
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
    )


def test_onnx_raw_yolo_channels_by_anchors_layout_is_parsed() -> None:
    raw = np.array(
        [
            [
                [20.0, 40.0],
                [20.0, 40.0],
                [10.0, 10.0],
                [10.0, 10.0],
                [0.9, 0.2],
            ]
        ],
        dtype=np.float32,
    )

    detections = _postprocess_output(raw, _info(), confidence=0.35, iou=0.5, person_class_id=0)

    assert len(detections) == 1
    np.testing.assert_allclose(detections.xyxy[0], [15.0, 15.0, 25.0, 25.0])


def test_onnx_raw_yolo_anchors_by_channels_layout_is_parsed() -> None:
    raw = np.array(
        [
            [
                [20.0, 20.0, 10.0, 10.0, 0.9],
                [40.0, 40.0, 10.0, 10.0, 0.2],
            ]
        ],
        dtype=np.float32,
    )

    detections = _postprocess_output(raw, _info(), confidence=0.35, iou=0.5, person_class_id=0)

    assert len(detections) == 1
    np.testing.assert_allclose(detections.xyxy[0], [15.0, 15.0, 25.0, 25.0])


def test_onnx_postprocess_applies_nms_after_confidence_filtering() -> None:
    raw = np.array(
        [
            [
                [20.0, 21.0],
                [20.0, 21.0],
                [10.0, 10.0],
                [10.0, 10.0],
                [0.9, 0.8],
            ]
        ],
        dtype=np.float32,
    )

    detections = _postprocess_output(raw, _info(), confidence=0.35, iou=0.5, person_class_id=0)

    assert len(detections) == 1
    assert detections.confidence[0] == np.float32(0.9)


def test_onnx_postprocess_keeps_person_class_only() -> None:
    raw = np.array(
        [
            [
                [20.0, 40.0],
                [20.0, 40.0],
                [10.0, 10.0],
                [10.0, 10.0],
                [0.9, 0.1],
                [0.2, 0.95],
            ]
        ],
        dtype=np.float32,
    )

    detections = _postprocess_output(raw, _info(), confidence=0.35, iou=0.5, person_class_id=0)

    assert len(detections) == 1
    np.testing.assert_allclose(detections.xyxy[0], [15.0, 15.0, 25.0, 25.0])
