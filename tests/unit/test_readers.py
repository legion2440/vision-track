from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vision_track.lifecycle import StreamState
from vision_track.readers import VideoReader
from vision_track.sources import VideoSource


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
    ) -> None:
        self.frames = frames
        self.timestamps = timestamps
        self.fps = fps
        self.frame_count = float(frames) if frame_count is None else frame_count
        self.failure_position = failure_position
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        self.released = True

    def read(self):
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
            if self.index >= self.frames and self.failure_position is not None:
                return self.failure_position
            return float(self.index)
        if prop == cv2.CAP_PROP_POS_MSEC:
            if self.index == 0:
                return 0.0
            return self.timestamps[self.index - 1]
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
):
    import vision_track.readers as readers

    source_path = tmp_path / "video.mp4"
    source_path.write_bytes(b"placeholder")
    monkeypatch.setattr(readers.cv2, "VideoCapture", lambda _: capture)
    states: list[StreamState] = []
    errors: list[str | None] = []
    queue = RecordingQueue()
    clock = FakeClock()
    reader = VideoReader(
        stream_id="stream-1",
        source=VideoSource.from_uri(str(source_path)),
        frame_queue=queue,
        state_callback=states.append,
        error_callback=errors.append,
        logger=logging.getLogger("test"),
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


def test_invalid_local_fps_falls_back_to_30fps(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(frames=3, timestamps=[0.0, 0.0, 0.0], fps=float("nan"))
    queue, states, _, _ = _run_reader(monkeypatch, tmp_path, capture)

    assert states[-1] is StreamState.EOF
    assert len(queue.packets) == 3
    assert queue.packets[1].source_timestamp_ms is None
    assert queue.packets[1].captured_at == 10.0 + 1 / 30.0
    assert queue.packets[2].captured_at == 10.0 + 2 / 30.0


def test_normal_local_eof_is_not_an_error(monkeypatch, tmp_path) -> None:
    capture = FakeCapture(frames=2, timestamps=[0.0, 50.0], fps=20.0, frame_count=2)
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
