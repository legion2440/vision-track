from __future__ import annotations

import numpy as np

from vision_track.detections import Detections, composite_track_id
from vision_track.detector import nms_detections


def test_filtering_and_unified_format() -> None:
    detections = Detections(
        [[1, 2, 10, 20], [2, 3, 11, 21]],
        [0.9, 0.2],
        [0, 1],
    )
    filtered = detections.filter(class_id=0, confidence=0.5)
    assert len(filtered) == 1
    assert filtered.xyxy.dtype == np.float32
    assert filtered.class_id.dtype == np.int32


def test_composite_ids_are_unique_between_streams() -> None:
    assert composite_track_id("a", 1) != composite_track_id("b", 1)


def test_nms_uses_xyxy_geometry() -> None:
    detections = Detections(
        [[10, 10, 30, 30], [11, 11, 31, 31], [50, 50, 60, 60]],
        [0.9, 0.8, 0.7],
        [0, 0, 0],
    )
    kept = nms_detections(detections, 0.5)
    assert len(kept) == 2
    assert np.allclose(kept.confidence, [0.9, 0.7])
