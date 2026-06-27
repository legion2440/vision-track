from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_track.configuration import load_config
from vision_track.detections import Detections
from vision_track.detector import DetectorBackend, InferenceResult
from vision_track.device import DeviceInfo
from vision_track.engine import ProcessingEngine


class FakeDetector(DetectorBackend):
    name = "fake"

    def __init__(self) -> None:
        super().__init__(
            "fake",
            DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
            image_size=64,
            confidence=0.1,
            iou=0.5,
        )
        self.loaded = False
        self.calls = 0

    def load(self) -> None:
        self.loaded = True

    def warmup(self) -> None:
        self.loaded = True

    def infer_batch(self, frames):
        self.calls += 1
        results = []
        for frame in frames:
            height, width = frame.shape[:2]
            offset = min(self.calls * 2, height // 2)
            detections = Detections(
                [[width * 0.35, offset, width * 0.65, min(height, offset + height * 0.35)]],
                [0.95],
                [0],
            )
            results.append(InferenceResult(detections, 1.0, self.name, "cpu"))
        return results


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (160, 120)
    )
    assert writer.isOpened()
    for index in range(30):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, index * 2), (100, min(119, index * 2 + 45)), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def fake_engine(tmp_path: Path) -> ProcessingEngine:
    config = replace(load_config(), log_file=str(tmp_path / "app_errors.log"))
    engine = ProcessingEngine(
        config,
        device=DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
        detector=FakeDetector(),
    )
    yield engine
    engine.shutdown()


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False
