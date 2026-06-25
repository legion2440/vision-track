from __future__ import annotations

import time
from pathlib import Path

import pytest

from vision_track.lifecycle import StreamState


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
