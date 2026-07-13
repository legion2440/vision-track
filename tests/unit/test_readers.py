from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np

from vision_track.lifecycle import StreamState
from vision_track.readers import (
    VideoReader,
    _local_read_is_eof,
    _webcam_connection_is_stable,
)
from vision_track.sources import VideoSource
from vision_track.webcams import OpenedWebcam


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStopEvent:
    def __init__(self, clock: FakeClock, *, interrupt_on_wait: bool = False) -> None:
        self.clock = clock
        self.interrupt_on_wait = interrupt_on_wait
        self.waits: list[float] = []
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        if self.interrupt_on_wait:
            self._set = True
            return True
        self.clock.advance(timeout)
        return False


class FakeCapture:
    def __init__(
        self,
        *,
        frames: int,
        timestamps: list[float],
        fps: float,
        frame_count: float | None = None,
        failure_position: float | None = None,
        avi_ratio: float = 0.0,
        transient_failures: dict[int, int] | None = None,
    ) -> None:
        self.frames = frames
        self.timestamps = timestamps
        self.fps = fps
        self.frame_count = float(frames) if frame_count is None else frame_count
        self.failure_position = failure_position
        self.avi_ratio = avi_ratio
        self.transient_failures = dict(transient_failures or {})
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True

    def read(self):
        if self.transient_failures.get(self.index, 0) > 0:
            self.transient_failures[self.index] -= 1
            return False, None
        if self.failure_position is not None and self.index == self.failure_position:
            return False, None
        if self.index >= self.frames:
            return False, None
        value = np.full((4, 4, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, value

    def get(self, prop: int) -> float:
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.index)
        if prop == cv2.CAP_PROP_POS_MSEC:
            if self.index == 0:
                return 0.0
            return self.timestamps[min(self.index - 1, len(self.timestamps) - 1)]
        if prop == cv2.CAP_PROP_POS_AVI_RATIO:
            return self.avi_ratio
        return 0.0


class RecordingQueue:
    def __init__(self) -> None:
        self.packets = []

    def put(self, packet) -> None:
        self.packets.append(packet)


def _run_reader(
    monkeypatch,
    tmp_path: Path,
    capture: FakeCapture,
    *,
    interrupt_on_wait: bool = False,
    initial_processing_delay: float | None = None,
    wait_for_initial_processing: bool = False,
    logger: logging.Logger | None = None,
):
    import vision_track.readers as readers

    source_path = tmp_path / "video.mp4"
    source_path.write_bytes(b"placeholder")
    monkeypatch.setattr(readers.cv2, "VideoCapture", lambda _: capture)
    states: list[StreamState] = []
    errors: list[str | None] = []
    queue = RecordingQueue()
    clock = FakeClock()

    def packet_callback(packet) -> bool:
        queue.put(packet)
        if (
            packet.processing_complete is not None
            and initial_processing_delay is not None
        ):
            clock.advance(initial_processing_delay)
            packet.processing_complete.set()
        return True

    reader = VideoReader(
        stream_id="stream-1",
        source=VideoSource.from_uri(str(source_path)),
        frame_queue=queue,
        state_callback=states.append,
        error_callback=errors.append,
        logger=logger or logging.getLogger("test"),
        packet_callback=packet_callback,
        wait_for_initial_processing=wait_for_initial_processing,
        clock=clock,
    )
    stop_event = FakeStopEvent(clock, interrupt_on_wait=interrupt_on_wait)
    reader._stop_event = stop_event
    reader._run()
    return queue, states, errors, stop_event


def test_local_video_playback_is_paced_from_media_timestamps(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(frames=3, timestamps=[0.0, 500.0, 1000.0], fps=60.0)
    queue, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert errors == [None]
    assert len(queue.packets) == 3
    assert queue.packets[0].captured_at == 10.0
    assert queue.packets[1].captured_at == 10.5
    assert queue.packets[2].captured_at == 11.0
    assert queue.packets[2].source_timestamp_ms == 1000.0
    assert stop_event.waits


def test_local_playback_reanchors_after_initial_frame_processing(
    monkeypatch,
    tmp_path,
) -> None:
    capture = FakeCapture(frames=3, timestamps=[0.0, 500.0, 1000.0], fps=60.0)
    queue, states, errors, _ = _run_reader(
        monkeypatch,
        tmp_path,
        capture,
        initial_processing_delay=2.0,
        wait_for_initial_processing=True,
    )

    assert states[-1] is StreamState.EOF
    assert errors == [None]
    assert [packet.frame_index for packet in queue.packets] == [0, 1, 2]
    assert [packet.captured_at for packet in queue.packets] == [10.0, 12.5, 13.0]
    assert queue.packets[0].processing_complete is not None
    assert all(packet.processing_complete is None for packet in queue.packets[1:])


def test_stop_while_waiting_for_initial_processing_is_interruptible(
    monkeypatch,
    tmp_path,
) -> None:
    capture = FakeCapture(frames=3, timestamps=[0.0, 500.0, 1000.0], fps=60.0)
    queue, states, _, _ = _run_reader(
        monkeypatch,
        tmp_path,
        capture,
        interrupt_on_wait=True,
        wait_for_initial_processing=True,
    )

    assert len(queue.packets) == 1
    assert states[-1] is StreamState.STOPPED


def test_invalid_local_fps_falls_back_to_30fps(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(frames=3, timestamps=[0.0, 0.0, 0.0], fps=float("nan"))
    queue, states, _, _ = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert len(queue.packets) == 3
    assert queue.packets[1].source_timestamp_ms is None
    assert queue.packets[1].captured_at == 10.0 + 1 / 30.0
    assert queue.packets[2].captured_at == 10.0 + 2 / 30.0


def test_two_frame_video_eof_requires_position_two() -> None:
    capture = FakeCapture(
        frames=2,
        timestamps=[0.0, 50.0],
        fps=20.0,
        frame_count=2,
    )

    capture.index = 1
    assert not _local_read_is_eof(capture, frames_read=1)

    capture.index = 2
    assert _local_read_is_eof(capture, frames_read=2)


def test_failure_at_frame_count_minus_one_retries_instead_of_eof(
    monkeypatch,
    tmp_path,
) -> None:
    capture = FakeCapture(
        frames=2,
        timestamps=[0.0, 50.0],
        fps=20.0,
        frame_count=2,
        transient_failures={1: 1},
    )
    queue, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert errors == [None]
    assert len(queue.packets) == 2
    assert stop_event.waits


def test_known_frame_count_at_end_produces_eof(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=2,
        timestamps=[0.0, 50.0],
        fps=20.0,
        frame_count=2,
    )
    _, states, errors, _ = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert errors == [None]


def test_local_decode_failure_before_eof_sets_failed(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=5,
        failure_position=1,
    )
    _, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.FAILED
    assert "Decode failure before end" in (errors[-1] or "")
    assert len(stop_event.waits) == 3


def test_current_local_decode_failure_writes_error_log(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "reader-failure.log"
    logger = logging.getLogger(f"reader-failure-{id(log_path)}")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("state=%(state)s | %(message)s"))
    logger.addHandler(handler)
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=5,
        failure_position=1,
    )

    try:
        _run_reader(monkeypatch, tmp_path, capture, logger=logger)
    finally:
        handler.close()
        logger.handlers.clear()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Decode failure before end of local video" in log_text
    assert "state=FAILED" in log_text


def test_unknown_frame_count_without_ratio_retries_then_fails(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=0,
        avi_ratio=math.nan,
    )
    _, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.FAILED
    assert "Decode failure before end" in (errors[-1] or "")
    assert len(stop_event.waits) == 3


def test_unknown_frame_count_with_complete_ratio_produces_eof(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=0,
        avi_ratio=0.99,
    )
    _, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert errors == [None]
    assert stop_event.waits == []


def test_transient_local_read_failure_retries_and_continues(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=3,
        timestamps=[0.0, 50.0, 100.0],
        fps=20.0,
        frame_count=3,
        transient_failures={1: 1},
    )
    queue, states, errors, stop_event = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert errors == [None]
    assert len(queue.packets) == 3
    assert stop_event.waits


def test_stop_during_local_retry_wait_exits_without_failed(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=0,
        avi_ratio=math.nan,
    )
    _, states, _, _ = _run_reader(
        monkeypatch,
        tmp_path,
        capture,
        interrupt_on_wait=True,
    )

    assert states[-1] is StreamState.STOPPED
    assert StreamState.FAILED not in states


def test_normal_eof_does_not_write_error_log(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "reader.log"
    logger = logging.getLogger(f"reader-eof-{id(log_path)}")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logger.addHandler(handler)
    capture = FakeCapture(
        frames=1,
        timestamps=[0.0],
        fps=20.0,
        frame_count=1,
    )

    _run_reader(monkeypatch, tmp_path, capture, logger=logger)
    logger.removeHandler(handler)
    handler.close()

    assert log_path.read_text(encoding="utf-8") == ""


def test_stop_during_local_pacing_wait_is_interruptible(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(frames=2, timestamps=[0.0, 10_000.0], fps=30.0)
    queue, states, _, _ = _run_reader(
        monkeypatch,
        tmp_path,
        capture,
        interrupt_on_wait=True,
    )

    assert len(queue.packets) == 1
    assert states[-1] is StreamState.STOPPED


def _run_webcam_reader(
    monkeypatch,
    opened_results: list[OpenedWebcam | Exception],
    *,
    reconnect_attempts: int,
    wait_for_initial_processing: bool = False,
    clock: FakeClock | None = None,
    backend_preferences: tuple[int, ...] = (10, 20, 30),
):
    import vision_track.readers as readers

    states: list[StreamState] = []
    errors: list[str | None] = []
    queue = RecordingQueue()
    clock = clock or FakeClock()
    open_calls: list[int] = []
    backend_calls: list[tuple[int, ...]] = []
    pending = list(opened_results)

    def fake_open_webcam(index: int, **kwargs) -> OpenedWebcam:
        open_calls.append(index)
        backend_calls.append(tuple(kwargs["backends"]))
        result = pending.pop(0)
        if isinstance(result, Exception):
            raise result
        kwargs["capture_callback"](result.capture)
        return result

    monkeypatch.setattr(readers, "open_webcam", fake_open_webcam)
    monkeypatch.setattr(
        readers,
        "webcam_backend_preferences",
        lambda: backend_preferences,
    )
    reader = VideoReader(
        stream_id="camera-stream",
        source=VideoSource.webcam(2),
        frame_queue=queue,
        state_callback=states.append,
        error_callback=errors.append,
        logger=logging.getLogger("test-webcam-reader"),
        reconnect_attempts=reconnect_attempts,
        reconnect_backoff_seconds=0.0,
        wait_for_initial_processing=wait_for_initial_processing,
        clock=clock,
    )
    stop_event = FakeStopEvent(clock)
    reader._stop_event = stop_event
    reader._run()
    return queue, states, errors, stop_event, open_calls, backend_calls


def _opened_webcam(
    capture: FakeCapture,
    *,
    value: int,
    captured_at: float,
    backend: int = 10,
) -> OpenedWebcam:
    return OpenedWebcam(
        capture=capture,
        first_frame=np.full((4, 4, 3), value, dtype=np.uint8),
        captured_at=captured_at,
        backend=backend,
    )


class ScriptedWebcamCapture:
    def __init__(
        self,
        clock: FakeClock,
        events: list[tuple[bool, float]],
    ) -> None:
        self.clock = clock
        self.events = list(events)
        self.released = False
        self.read_calls = 0

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        self.read_calls += 1
        if not self.events:
            return False, None
        ok, advance_seconds = self.events.pop(0)
        self.clock.advance(advance_seconds)
        if not ok:
            return False, None
        return True, np.full((4, 4, 3), self.read_calls, dtype=np.uint8)

    def get(self, _prop: int) -> float:
        return 30.0

    def release(self) -> None:
        self.released = True


def test_webcam_preserves_validated_first_frame_and_has_no_local_barrier(
    monkeypatch,
) -> None:
    capture = FakeCapture(frames=1, timestamps=[0.0], fps=30.0)
    capture.index = 1
    queue, states, errors, stop_event, open_calls, _ = _run_webcam_reader(
        monkeypatch,
        [_opened_webcam(capture, value=9, captured_at=7.5)],
        reconnect_attempts=0,
        wait_for_initial_processing=True,
    )

    assert open_calls == [2]
    assert len(queue.packets) == 1
    assert np.all(queue.packets[0].frame == 9)
    assert queue.packets[0].captured_at == 7.5
    assert queue.packets[0].source_timestamp_ms is None
    assert queue.packets[0].processing_complete is None
    assert StreamState.EOF not in states
    assert states[-1] is StreamState.FAILED
    assert errors[0] is None
    assert len(stop_event.waits) == 2
    assert capture.released


def test_unstable_msmf_reconnect_starts_with_dshow(
    monkeypatch,
) -> None:
    first = FakeCapture(frames=1, timestamps=[0.0], fps=30.0)
    first.index = 1
    second = FakeCapture(frames=1, timestamps=[0.0], fps=30.0)
    second.index = 1

    queue, states, _, stop_event, open_calls, backend_calls = _run_webcam_reader(
        monkeypatch,
        [
            _opened_webcam(
                first,
                value=1,
                captured_at=10.0,
                backend=cv2.CAP_MSMF,
            ),
            _opened_webcam(
                second,
                value=2,
                captured_at=10.5,
                backend=cv2.CAP_DSHOW,
            ),
        ],
        reconnect_attempts=1,
        backend_preferences=(cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY),
    )

    assert open_calls == [2, 2]
    assert backend_calls == [
        (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY),
        (cv2.CAP_DSHOW, cv2.CAP_ANY),
    ]
    assert [int(packet.frame[0, 0, 0]) for packet in queue.packets] == [1, 2]
    assert states.count(StreamState.RECONNECTING) == 1
    assert states[-1] is StreamState.FAILED
    assert len(stop_event.waits) == 5
    assert first.released and second.released


def test_webcam_open_failures_exhaust_bounded_reconnect_budget(monkeypatch) -> None:
    _, states, errors, _, open_calls, backend_calls = _run_webcam_reader(
        monkeypatch,
        [OSError("one"), OSError("two"), OSError("three")],
        reconnect_attempts=2,
    )

    assert open_calls == [2, 2, 2]
    assert backend_calls == [(10, 20, 30)] * 3
    assert states == [
        StreamState.CONNECTING,
        StreamState.RECONNECTING,
        StreamState.RECONNECTING,
        StreamState.FAILED,
    ]
    assert errors == ["one", "two", "three"]


def test_webcam_reconnect_budget_resets_only_after_frames_and_time() -> None:
    assert not _webcam_connection_is_stable(29, 10.0, 13.0)
    assert not _webcam_connection_is_stable(30, 10.0, 12.999)
    assert _webcam_connection_is_stable(30, 10.0, 13.0)
    assert not _webcam_connection_is_stable(30, None, 13.0)


def test_webcam_stable_connection_resets_budget_in_reader_loop(monkeypatch) -> None:
    clock = FakeClock()
    stable_capture = ScriptedWebcamCapture(
        clock,
        [(True, 3.0 / 29)] * 29 + [(False, 0.0)] * 3,
    )
    final_capture = FakeCapture(frames=1, timestamps=[0.0], fps=30.0)
    final_capture.index = 1

    queue, states, _, _, open_calls, backend_calls = _run_webcam_reader(
        monkeypatch,
        [
            OSError("initial open failed"),
            OpenedWebcam(
                capture=stable_capture,
                first_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                captured_at=10.0,
                backend=10,
            ),
            _opened_webcam(final_capture, value=7, captured_at=13.02),
        ],
        reconnect_attempts=1,
        clock=clock,
    )

    assert open_calls == [2, 2, 2]
    assert backend_calls == [(10, 20, 30)] * 3
    assert states.count(StreamState.RECONNECTING) == 2
    assert states[-1] is StreamState.FAILED
    assert len(queue.packets) == 31
    assert stable_capture.released and final_capture.released


def test_stable_fallback_backend_restores_standard_order(monkeypatch) -> None:
    clock = FakeClock()
    unstable_msmf = ScriptedWebcamCapture(
        clock,
        [(False, 0.0)] * 3,
    )
    stable_dshow = ScriptedWebcamCapture(
        clock,
        [(True, 3.0 / 29)] * 29 + [(False, 0.0)] * 3,
    )
    final_msmf = ScriptedWebcamCapture(
        clock,
        [(False, 0.0)] * 3,
    )
    preferences = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)

    queue, states, _, _, open_calls, backend_calls = _run_webcam_reader(
        monkeypatch,
        [
            OpenedWebcam(
                capture=unstable_msmf,
                first_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                captured_at=10.0,
                backend=cv2.CAP_MSMF,
            ),
            OpenedWebcam(
                capture=stable_dshow,
                first_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                captured_at=10.02,
                backend=cv2.CAP_DSHOW,
            ),
            OpenedWebcam(
                capture=final_msmf,
                first_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                captured_at=13.04,
                backend=cv2.CAP_MSMF,
            ),
        ],
        reconnect_attempts=1,
        clock=clock,
        backend_preferences=preferences,
    )

    assert open_calls == [2, 2, 2]
    assert backend_calls == [
        preferences,
        (cv2.CAP_DSHOW, cv2.CAP_ANY),
        preferences,
    ]
    assert states.count(StreamState.RECONNECTING) == 2
    assert states[-1] is StreamState.FAILED
    assert len(queue.packets) == 32
    assert unstable_msmf.released
    assert stable_dshow.released
    assert final_msmf.released


def test_webcam_transient_read_failure_restarts_stability_window(
    monkeypatch,
) -> None:
    clock = FakeClock()
    capture = ScriptedWebcamCapture(
        clock,
        [(True, 3.0 / 28)] * 28
        + [(False, 0.0), (True, 0.0)]
        + [(False, 0.0)] * 3,
    )

    _, states, _, _, open_calls, backend_calls = _run_webcam_reader(
        monkeypatch,
        [
            OSError("initial open failed"),
            OpenedWebcam(
                capture=capture,
                first_frame=np.zeros((4, 4, 3), dtype=np.uint8),
                captured_at=10.0,
                backend=10,
            ),
            AssertionError("unstable connection must not regain reconnect budget"),
        ],
        reconnect_attempts=1,
        clock=clock,
    )

    assert open_calls == [2, 2]
    assert backend_calls == [(10, 20, 30)] * 2
    assert states.count(StreamState.RECONNECTING) == 1
    assert states[-1] is StreamState.FAILED
    assert capture.released
