from __future__ import annotations

import numpy as np

from vision_track.counting import CrossingGuard, ZoneCounter, ZoneGeometry
from vision_track.detections import Detections
from vision_track.tracking import ByteTrackSettings, StreamTracker


def test_trackers_have_independent_state() -> None:
    settings = ByteTrackSettings(minimum_consecutive_frames=1)
    first = StreamTracker(settings)
    second = StreamTracker(settings)
    detection = Detections([[10, 10, 30, 40]], [0.9], [0])
    first.update(detection)
    assert first is not second
    assert first.trajectories is not second.trajectories


def test_counters_are_independent() -> None:
    first = ZoneCounter(ZoneGeometry())
    second = ZoneCounter(ZoneGeometry())
    first.in_count = 3
    assert second.in_count == 0


def test_duplicate_counting_protection() -> None:
    guard = CrossingGuard()
    assert guard.record(7, "in")
    assert not guard.record(7, "in")
    assert guard.record(7, "out")


def test_line_crossing_is_counted_once() -> None:
    counter = ZoneCounter(ZoneGeometry(line_start=(0.1, 0.5), line_end=(0.9, 0.5)))
    for y in [70, 68, 65, 35, 32, 30, 28]:
        detections = Detections(
            np.array([[40, y - 20, 70, y]], dtype=np.float32),
            [0.9],
            [0],
            [1],
        )
        counter.update(detections, (100, 100))
    assert counter.in_count + counter.out_count == 1

