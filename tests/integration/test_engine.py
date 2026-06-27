from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vision_track.lifecycle import StreamState
from vision_track.queues import FramePacket


pytestmark = pytest.mark.integration


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


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

    assert reader.started
    assert context.state is StreamState.CREATED
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
