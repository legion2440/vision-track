from __future__ import annotations

import sys
import types

import numpy as np

from vision_track.counting import ZoneCounter, ZoneGeometry
from vision_track.detections import Detections
from vision_track.tracking import ByteTrackSettings, StreamTracker


def install_fake_supervision(monkeypatch, events=None, occupancy=0) -> None:
    event_queue = list(events or [])

    class FakeDetections:
        def __init__(self, xyxy, confidence=None, class_id=None, tracker_id=None):
            self.xyxy = np.asarray(xyxy, dtype=np.float32)
            self.confidence = confidence
            self.class_id = class_id
            self.tracker_id = tracker_id

        def __len__(self):
            return len(self.xyxy)

    class FakeByteTrack:
        def __init__(self, **_):
            self.reset_called = False

        def update_with_detections(self, detections):
            if detections.tracker_id is None:
                detections.tracker_id = np.arange(1, len(detections) + 1)
            return detections

        def reset(self):
            self.reset_called = True

    class FakePoint:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    class FakeLineZone:
        def __init__(self, **_):
            pass

        def trigger(self, detections):
            if event_queue:
                crossed_in, crossed_out = event_queue.pop(0)
            else:
                crossed_in = [False] * len(detections)
                crossed_out = [False] * len(detections)
            return np.asarray(crossed_in, dtype=bool), np.asarray(crossed_out, dtype=bool)

    class FakePolygonZone:
        def __init__(self, **_):
            self.current_count = 0

        def trigger(self, detections):
            self.current_count = occupancy if occupancy is not None else len(detections)
            return np.ones(len(detections), dtype=bool)

    fake_supervision = types.SimpleNamespace(
        Detections=FakeDetections,
        ByteTrack=FakeByteTrack,
        Point=FakePoint,
        LineZone=FakeLineZone,
        PolygonZone=FakePolygonZone,
    )
    monkeypatch.setitem(sys.modules, "supervision", fake_supervision)


def test_trackers_have_independent_state(monkeypatch) -> None:
    install_fake_supervision(monkeypatch)
    settings = ByteTrackSettings(minimum_consecutive_frames=1)
    first = StreamTracker(settings)
    second = StreamTracker(settings)
    detection = Detections([[10, 10, 30, 40]], [0.9], [0])
    first.update(detection)
    assert first is not second
    assert first.trajectories is not second.trajectories


def _tracked_detection(tracker_id: int = 7) -> Detections:
    return Detections([[10, 10, 30, 40]], [0.9], [0], [tracker_id])


def test_trajectory_survives_within_lost_track_buffer(monkeypatch) -> None:
    install_fake_supervision(monkeypatch)
    tracker = StreamTracker(ByteTrackSettings(lost_track_buffer=3))

    tracker.update(_tracked_detection())
    for _ in range(3):
        tracker.update(Detections.empty())

    assert 7 in tracker.trajectories
    assert tracker._trajectory_last_seen[7] == 1


def test_trajectory_is_removed_after_lost_track_buffer(monkeypatch) -> None:
    install_fake_supervision(monkeypatch)
    tracker = StreamTracker(ByteTrackSettings(lost_track_buffer=2))

    tracker.update(_tracked_detection())
    tracker.update(Detections.empty())
    tracker.update(Detections.empty())
    assert 7 in tracker.trajectories

    tracker.update(Detections.empty())

    assert 7 not in tracker.trajectories
    assert 7 not in tracker._trajectory_last_seen


def test_empty_detections_expire_stale_trajectory(monkeypatch) -> None:
    install_fake_supervision(monkeypatch)
    tracker = StreamTracker(ByteTrackSettings(lost_track_buffer=0))

    tracker.update(_tracked_detection())
    tracker.update(Detections.empty())

    assert tracker.trajectories == {}
    assert tracker._trajectory_last_seen == {}


def test_tracker_reset_clears_all_trajectory_state(monkeypatch) -> None:
    install_fake_supervision(monkeypatch)
    tracker = StreamTracker(ByteTrackSettings(lost_track_buffer=3))
    tracker.update(_tracked_detection())

    tracker.reset()

    assert tracker.trajectories == {}
    assert tracker._trajectory_last_seen == {}
    assert tracker._trajectory_frame == 0


def test_counters_are_independent() -> None:
    first = ZoneCounter(ZoneGeometry())
    second = ZoneCounter(ZoneGeometry())
    first.in_count = 3
    assert second.in_count == 0


def test_one_entry_is_counted(monkeypatch) -> None:
    install_fake_supervision(monkeypatch, events=[([True], [False])])
    counter = ZoneCounter(ZoneGeometry(line_start=(0.1, 0.5), line_end=(0.9, 0.5)))
    detections = Detections([[40, 20, 70, 50]], [0.9], [0], [1])
    counter.update(detections, (100, 100))
    assert counter.in_count == 1
    assert counter.out_count == 0


def test_one_exit_is_counted(monkeypatch) -> None:
    install_fake_supervision(monkeypatch, events=[([False], [True])])
    counter = ZoneCounter(ZoneGeometry(line_start=(0.1, 0.5), line_end=(0.9, 0.5)))
    detections = Detections([[40, 60, 70, 90]], [0.9], [0], [1])
    counter.update(detections, (100, 100))
    assert counter.in_count == 0
    assert counter.out_count == 1


def test_same_tracker_can_enter_exit_and_enter_again(monkeypatch) -> None:
    install_fake_supervision(
        monkeypatch,
        events=[
            ([True], [False]),
            ([False], [True]),
            ([True], [False]),
        ],
    )
    counter = ZoneCounter(ZoneGeometry(line_start=(0.1, 0.5), line_end=(0.9, 0.5)))
    detections = Detections([[40, 20, 70, 50]], [0.9], [0], [7])
    counter.update(detections, (100, 100))
    counter.update(detections, (100, 100))
    counter.update(detections, (100, 100))
    assert counter.in_count == 2
    assert counter.out_count == 1


def test_repeated_frames_without_linezone_event_do_not_duplicate_counts(monkeypatch) -> None:
    install_fake_supervision(
        monkeypatch,
        events=[
            ([True], [False]),
            ([False], [False]),
            ([False], [False]),
        ],
    )
    counter = ZoneCounter(ZoneGeometry(line_start=(0.1, 0.5), line_end=(0.9, 0.5)))
    detections = Detections([[40, 20, 70, 50]], [0.9], [0], [1])
    for _ in range(3):
        counter.update(detections, (100, 100))
    assert counter.in_count == 1
    assert counter.out_count == 0


def test_polygon_occupancy_uses_polygon_zone_current_count(monkeypatch) -> None:
    install_fake_supervision(monkeypatch, events=[([False], [False])], occupancy=4)
    counter = ZoneCounter(ZoneGeometry())
    counter.update(Detections([[1, 1, 5, 5]], [0.9], [0], [1]), (100, 100))
    assert counter.occupancy == 4


def test_counter_reset_clears_totals(monkeypatch) -> None:
    install_fake_supervision(monkeypatch, events=[([True], [False])])
    counter = ZoneCounter(ZoneGeometry())
    counter.update(Detections([[1, 1, 5, 5]], [0.9], [0], [1]), (100, 100))
    counter.reset()
    assert counter.in_count == 0
    assert counter.out_count == 0
    assert counter.occupancy == 0
