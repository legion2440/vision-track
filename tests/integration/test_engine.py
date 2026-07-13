from __future__ import annotations

import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from vision_track.lifecycle import StreamState
from vision_track.queues import FramePacket
from vision_track.sources import VideoSource
from vision_track.webcams import OpenedWebcam


pytestmark = pytest.mark.integration


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


class FakeWebcamCapture:
    def __init__(self, *, successful_reads: int | None = None) -> None:
        self.successful_reads = successful_reads
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        time.sleep(0.01)
        if self.released:
            return False, None
        if self.successful_reads is not None and self.read_count >= self.successful_reads:
            return False, None
        value = self.read_count % 255
        self.read_count += 1
        return True, np.full((120, 160, 3), value, dtype=np.uint8)

    def get(self, _prop: int) -> float:
        return 30.0

    def release(self) -> None:
        self.released = True


class FakeWebcamFactory:
    def __init__(self, limits: dict[int, list[int | None]] | None = None) -> None:
        self.limits = {index: list(values) for index, values in (limits or {}).items()}
        self.captures: dict[int, list[FakeWebcamCapture]] = defaultdict(list)

    def __call__(self, index: int, **kwargs) -> OpenedWebcam:
        limits = self.limits.get(index, [])
        successful_reads = limits.pop(0) if limits else None
        capture = FakeWebcamCapture(successful_reads=successful_reads)
        self.captures[index].append(capture)
        kwargs["capture_callback"](capture)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            kwargs["capture_callback"](None)
            raise OSError(f"Unable to open webcam device {index}")
        return OpenedWebcam(
            capture=capture,
            first_frame=frame,
            captured_at=kwargs["clock"](),
            backend=0,
        )


def test_single_local_video_reaches_eof(fake_engine, synthetic_video: Path) -> None:
    stream_id = fake_engine.add_stream(str(synthetic_video))
    fake_engine.start(stream_id)
    context = fake_engine.get(stream_id)
    assert wait_until(lambda: context.state is StreamState.EOF)
    assert wait_until(lambda: context.metrics.processed_frames > 0)
    assert context.error is None


def test_two_local_videos_are_independent(fake_engine, synthetic_video: Path) -> None:
    first = fake_engine.add_stream(str(synthetic_video))
    second = fake_engine.add_stream(str(synthetic_video))
    fake_engine.start_all()
    assert wait_until(lambda: fake_engine.get(first).metrics.processed_frames > 0)
    assert wait_until(lambda: fake_engine.get(second).metrics.processed_frames > 0)
    assert fake_engine.get(first).tracker is not fake_engine.get(second).tracker
    assert fake_engine.get(first).counter is not fake_engine.get(second).counter


def test_webcam_and_local_video_lifecycle_release_every_capture(
    fake_engine,
    synthetic_video: Path,
    monkeypatch,
) -> None:
    import vision_track.readers as readers

    factory = FakeWebcamFactory()
    monkeypatch.setattr(readers, "open_webcam", factory)
    camera = fake_engine.add_stream(VideoSource.webcam(0))
    local = fake_engine.add_stream(str(synthetic_video))

    fake_engine.start_all()
    assert wait_until(lambda: fake_engine.get(camera).metrics.processed_frames >= 5)
    assert wait_until(lambda: fake_engine.get(local).metrics.processed_frames >= 1)
    first_latency = fake_engine.get(camera).metrics.end_to_end_latency_ms
    first_capture = factory.captures[0][0]
    first_reader = fake_engine.get(camera).reader

    time.sleep(0.5)
    camera_context = fake_engine.get(camera)
    assert camera_context.metrics.processed_frames >= 10
    assert camera_context.metrics.end_to_end_latency_ms < max(1_000.0, first_latency * 5)

    fake_engine.stop(camera)
    assert wait_until(lambda: first_capture.released)
    assert not first_reader.running

    fake_engine.restart(camera)
    assert wait_until(lambda: len(factory.captures[0]) == 2)
    assert wait_until(lambda: fake_engine.get(camera).metrics.processed_frames >= 3)
    restarted_capture = factory.captures[0][1]
    restarted_reader = fake_engine.get(camera).reader

    fake_engine.remove(camera)
    assert wait_until(lambda: restarted_capture.released)
    assert not restarted_reader.running
    assert camera not in {context.stream_id for context in fake_engine.contexts()}

    assert wait_until(lambda: fake_engine.get(local).state is StreamState.EOF)
    fake_engine.remove(local)

    shutdown_camera = fake_engine.add_stream(VideoSource.webcam(1))
    fake_engine.start(shutdown_camera)
    assert wait_until(lambda: fake_engine.get(shutdown_camera).metrics.processed_frames >= 1)
    shutdown_capture = factory.captures[1][0]
    shutdown_reader = fake_engine.get(shutdown_camera).reader
    fake_engine.shutdown()
    assert wait_until(lambda: shutdown_capture.released)
    assert not shutdown_reader.running


def test_webcam_reconnects_after_three_read_failures(
    fake_engine,
    monkeypatch,
) -> None:
    import vision_track.readers as readers

    factory = FakeWebcamFactory({0: [3, None]})
    monkeypatch.setattr(readers, "open_webcam", factory)
    camera = fake_engine.add_stream(VideoSource.webcam(0))

    fake_engine.start(camera)

    assert wait_until(lambda: len(factory.captures[0]) == 2, timeout=4.0)
    assert factory.captures[0][0].released
    assert wait_until(lambda: fake_engine.get(camera).metrics.processed_frames >= 5)
    assert fake_engine.get(camera).state is StreamState.ACTIVE

    fake_engine.stop(camera)
    assert wait_until(lambda: factory.captures[0][1].released)


def test_webcam_local_http_and_rtsp_share_pipeline_without_queue_growth(
    fake_engine,
    synthetic_video: Path,
    monkeypatch,
) -> None:
    import vision_track.readers as readers

    webcam_factory = FakeWebcamFactory()
    monkeypatch.setattr(readers, "open_webcam", webcam_factory)
    original_video_capture = readers.cv2.VideoCapture
    remote_captures: dict[str, FakeWebcamCapture] = {}

    def capture_factory(source):
        uri = str(source)
        if uri.startswith(("http://", "https://", "rtsp://", "rtsps://")):
            capture = FakeWebcamCapture()
            remote_captures[uri] = capture
            return capture
        return original_video_capture(source)

    monkeypatch.setattr(readers.cv2, "VideoCapture", capture_factory)
    stream_ids = {
        "webcam": fake_engine.add_stream(VideoSource.webcam(0)),
        "local": fake_engine.add_stream(str(synthetic_video)),
        "http": fake_engine.add_stream("https://example.test/live.mp4"),
        "rtsp": fake_engine.add_stream("rtsp://example.test/live"),
    }

    fake_engine.start_all()

    for stream_id in stream_ids.values():
        assert wait_until(
            lambda stream_id=stream_id: (
                fake_engine.get(stream_id).metrics.processed_frames >= 3
            )
        )
    for source_name in ("webcam", "http", "rtsp"):
        context = fake_engine.get(stream_ids[source_name])
        assert context.state is StreamState.ACTIVE
        assert context.queue._queue.maxsize == 1
        assert context.queue._queue.qsize() <= 1
        assert context.queue.received >= context.metrics.processed_frames

    assert wait_until(
        lambda: fake_engine.get(stream_ids["local"]).state is StreamState.EOF
    )
    fake_engine.stop_all()

    assert webcam_factory.captures[0][0].released
    assert all(capture.released for capture in remote_captures.values())


def test_webcam_stop_during_blocked_render_is_prompt_and_drops_stale_publication(
    fake_engine,
    monkeypatch,
) -> None:
    import vision_track.readers as readers

    webcam_factory = FakeWebcamFactory()
    monkeypatch.setattr(readers, "open_webcam", webcam_factory)
    render_started = threading.Event()
    release_render = threading.Event()
    render_finished = threading.Event()

    def blocked_render(frame, *_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=3.0)
        render_finished.set()
        return frame

    monkeypatch.setattr("vision_track.scheduler.render_frame", blocked_render)
    camera = fake_engine.add_stream(VideoSource.webcam(0))
    context = fake_engine.get(camera)
    try:
        fake_engine.start(camera)
        assert render_started.wait(timeout=3.0)

        started = time.perf_counter()
        fake_engine.stop(camera)
        stop_seconds = time.perf_counter() - started

        assert stop_seconds < 1.0
        assert webcam_factory.captures[0][0].released

        release_render.set()
        assert render_finished.wait(timeout=1.0)
        fake_engine.scheduler.stop()

        assert context.state is StreamState.STOPPED
        assert context.latest_rendered_frame is None
        assert context.metrics.processed_frames == 0
    finally:
        release_render.set()


def test_working_and_broken_sources_do_not_share_failure(
    fake_engine, synthetic_video: Path, tmp_path: Path
) -> None:
    working = fake_engine.add_stream(str(synthetic_video))
    broken = fake_engine.add_stream(str(tmp_path / "missing.mp4"))
    fake_engine.start_all()
    assert wait_until(lambda: fake_engine.get(broken).state is StreamState.FAILED)
    assert wait_until(lambda: fake_engine.get(working).metrics.processed_frames > 0)


def test_tracker_parameter_change_is_scoped(fake_engine, synthetic_video: Path) -> None:
    first = fake_engine.add_stream(str(synthetic_video))
    second = fake_engine.add_stream(str(synthetic_video))
    first_context = fake_engine.get(first)
    second_context = fake_engine.get(second)
    first_tracker = first_context.tracker
    second_tracker = second_context.tracker
    first_context.counter.in_count = 4
    fake_engine.update_tracker(first, lost_track_buffer=45)
    assert first_context.tracker is not first_tracker
    assert second_context.tracker is second_tracker
    assert first_context.counter.in_count == 4


def test_stop_restart_remove_and_replace(
    fake_engine, synthetic_video: Path, tmp_path: Path
) -> None:
    first = fake_engine.add_stream(str(synthetic_video))
    second = fake_engine.add_stream(str(synthetic_video))
    fake_engine.start_all()
    assert wait_until(lambda: fake_engine.get(first).metrics.processed_frames > 0)
    fake_engine.stop(first)
    assert fake_engine.get(first).state is StreamState.STOPPED
    calls_before = fake_engine.detector.calls
    fake_engine.restart(first)
    assert wait_until(lambda: fake_engine.detector.calls > calls_before)
    fake_engine.replace_source(second, str(tmp_path / "replacement.mp4"))
    assert fake_engine.get(second).state is StreamState.CREATED
    assert fake_engine.get(first).source.uri == str(synthetic_video)
    fake_engine.remove(second)
    assert second not in [context.stream_id for context in fake_engine.contexts()]


def test_start_after_eof_replays_from_clean_state(fake_engine, synthetic_video: Path, monkeypatch) -> None:
    stream_id = fake_engine.add_stream(str(synthetic_video))
    context = fake_engine.get(stream_id)
    original_tracker = context.tracker
    original_counter = context.counter
    context.force_state(StreamState.EOF)
    context.queue.put(
        FramePacket(
            frame=np.zeros((4, 4, 3), dtype=np.uint8),
            frame_index=9,
            captured_at=1.0,
        )
    )
    context.counter.in_count = 5
    context.metrics.processed_frames = 12
    context.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    context.latest_rendered_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    context.latest_detections = object()
    context.actual_backend = "onnxruntime"
    context.actual_device = "cpu"
    context.actual_provider = "CPUExecutionProvider"
    context.error = "previous failure"

    class NoopReader:
        running = False
        started = False

        def start(self):
            self.started = True

        def stop(self, timeout=3.0):
            pass

    reader = NoopReader()
    monkeypatch.setattr(fake_engine, "_build_reader", lambda _context: reader)

    fake_engine.start(stream_id)

    assert wait_until(lambda: reader.started)
    assert context.state is StreamState.PREPARING
    assert context.queue.received == 0
    assert context.metrics.processed_frames == 0
    assert context.latest_frame is None
    assert context.latest_rendered_frame is None
    assert context.latest_detections is None
    assert context.actual_backend is None
    assert context.actual_device is None
    assert context.actual_provider is None
    assert context.error is None
    assert context.tracker is not original_tracker
    assert context.counter is not original_counter
    assert context.counter.in_count == 0
