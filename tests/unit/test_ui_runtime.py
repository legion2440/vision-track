from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vision_track.engine as engine_module
import vision_track.scheduler as scheduler_module
import vision_track.ui as ui_module
from vision_track.context import StreamContext, StreamOptions
from vision_track.counting import ZoneGeometry
from vision_track.detections import Detections
from vision_track.detector import InferenceResult
from vision_track.engine import ProcessingEngine
from vision_track.lifecycle import StreamState
from vision_track.queues import FramePacket, LatestFrameQueue
from vision_track.scheduler import SharedInferenceScheduler
from vision_track.sources import VideoSource
from vision_track.tracking import ByteTrackSettings
from vision_track.ui import (
    CachedStreamFrame,
    snapshot_stream_frame,
    stream_source_token,
    update_stream_frame_cache,
)


class FakeReader:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_calls += 1


class FakeCounter:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.geometry = ZoneGeometry()
        self.in_count = 4
        self.out_count = 5
        self.occupancy = 6

    def reset(self) -> None:
        self.reset_calls += 1
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0


def _context(stream_id: str = "stream-1") -> StreamContext:
    return StreamContext(stream_id, VideoSource.from_uri(f"{stream_id}.mp4"))


def _minimal_engine(context: StreamContext) -> ProcessingEngine:
    engine = ProcessingEngine.__new__(ProcessingEngine)
    engine._contexts = {context.stream_id: context}
    engine._lock = threading.RLock()
    return engine


def _scheduler(context: StreamContext) -> SharedInferenceScheduler:
    return SharedInferenceScheduler(
        detector=object(),
        contexts_provider=lambda: [context],
        logger=logging.getLogger("test-ui-runtime"),
    )


def _packet(runtime_generation: int, frame_index: int = 17) -> FramePacket:
    return FramePacket(
        frame=np.full((4, 4, 3), 11, dtype=np.uint8),
        frame_index=frame_index,
        captured_at=time.perf_counter(),
        runtime_generation=runtime_generation,
    )


def _result() -> InferenceResult:
    return InferenceResult(
        detections=Detections.empty(),
        latency_ms=2.5,
        backend="pytorch",
        device="cpu",
        provider=None,
    )


def test_scheduler_publishes_rendered_version_atomically(monkeypatch) -> None:
    context = _context()
    context.force_state(StreamState.ACTIVE)
    context.runtime_generation = 8
    context.options.detection_enabled = False
    context.options.tracking_enabled = False
    context.options.counting_enabled = False
    packet = _packet(runtime_generation=8, frame_index=17)
    rendered = np.full((4, 4, 3), 99, dtype=np.uint8)
    scheduler = _scheduler(context)

    monkeypatch.setattr(
        scheduler_module,
        "render_frame",
        lambda *_args, **_kwargs: rendered,
    )

    assert scheduler._finalize(context, packet, _result()) is True
    assert context.latest_frame is packet.frame
    assert context.latest_rendered_frame is rendered
    assert context.latest_rendered_version == (8, 1)
    assert context.render_revision == 1
    assert context.latest_detections is not None
    assert context.metrics.processed_frames == 1


def test_scheduler_second_publication_advances_render_revision(monkeypatch) -> None:
    context = _context()
    context.force_state(StreamState.ACTIVE)
    context.runtime_generation = 8
    context.options.detection_enabled = False
    context.options.tracking_enabled = False
    context.options.counting_enabled = False
    first_packet = _packet(runtime_generation=8, frame_index=17)
    second_packet = _packet(runtime_generation=8, frame_index=99)
    scheduler = _scheduler(context)

    monkeypatch.setattr(
        scheduler_module,
        "render_frame",
        lambda *_args, **_kwargs: np.full((4, 4, 3), 99, dtype=np.uint8),
    )

    assert scheduler._finalize(context, first_packet, _result()) is True
    assert context.latest_rendered_version == (8, 1)
    assert scheduler._finalize(context, second_packet, _result()) is True
    assert context.latest_frame is second_packet.frame
    assert context.latest_rendered_version == (8, 2)
    assert context.render_revision == 2


def test_stale_packet_does_not_publish_version(monkeypatch) -> None:
    context = _context()
    context.runtime_generation = 9
    context.render_revision = 4
    previous_raw = np.full((4, 4, 3), 3, dtype=np.uint8)
    previous_frame = np.full((4, 4, 3), 5, dtype=np.uint8)
    context.latest_frame = previous_raw
    context.latest_rendered_frame = previous_frame
    context.latest_rendered_version = (9, 4)
    context.metrics.processed_frames = 3
    packet = _packet(runtime_generation=8, frame_index=17)
    scheduler = _scheduler(context)

    monkeypatch.setattr(
        scheduler_module,
        "render_frame",
        lambda *_args, **_kwargs: pytest.fail("stale packet rendered a frame"),
    )
    monkeypatch.setattr(
        context,
        "publish_rendered_frame",
        lambda *_args, **_kwargs: pytest.fail("stale packet published a frame"),
    )

    assert scheduler._finalize(context, packet, _result()) is False
    assert context.latest_frame is previous_raw
    assert context.latest_rendered_frame is previous_frame
    assert context.latest_rendered_version == (9, 4)
    assert context.render_revision == 4
    assert context.metrics.processed_frames == 3


def test_engine_stop_preserves_published_frame_identity() -> None:
    context = _context()
    engine = _minimal_engine(context)
    reader = FakeReader()
    raw = np.full((4, 4, 3), 6, dtype=np.uint8)
    rendered = np.full((4, 4, 3), 7, dtype=np.uint8)
    context.reader = reader
    context.latest_frame = raw
    context.latest_rendered_frame = rendered
    context.latest_rendered_version = (3, 15)
    context.render_revision = 15
    context.runtime_generation = 3

    ProcessingEngine.stop(engine, context.stream_id)

    assert context.runtime_generation == 4
    assert context.state is StreamState.STOPPED
    assert reader.stop_calls == 1
    assert context.latest_frame is raw
    assert context.latest_rendered_frame is rendered
    assert context.latest_rendered_version == (3, 15)
    assert context.render_revision == 15


def test_repeat_stop_preserves_published_frame_identity() -> None:
    context = _context()
    engine = _minimal_engine(context)
    reader = FakeReader()
    raw = np.full((4, 4, 3), 6, dtype=np.uint8)
    rendered = np.full((4, 4, 3), 7, dtype=np.uint8)
    context.reader = reader
    context.latest_frame = raw
    context.latest_rendered_frame = rendered
    context.latest_rendered_version = (3, 15)
    context.render_revision = 15
    context.runtime_generation = 3

    ProcessingEngine.stop(engine, context.stream_id)
    ProcessingEngine.stop(engine, context.stream_id)

    assert context.runtime_generation == 5
    assert context.state is StreamState.STOPPED
    assert reader.stop_calls == 2
    assert context.latest_frame is raw
    assert context.latest_rendered_frame is rendered
    assert context.latest_rendered_version == (3, 15)
    assert context.render_revision == 15


def test_runtime_reset_clears_context_frame_and_version(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.latest_frame = np.full((4, 4, 3), 1, dtype=np.uint8)
    context.latest_rendered_frame = np.full((4, 4, 3), 2, dtype=np.uint8)
    context.latest_rendered_version = (4, 22)
    context.latest_detections = object()
    context.render_revision = 22

    monkeypatch.setattr(engine, "_tracker_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(engine, "_zone_geometry", lambda: SimpleNamespace())
    monkeypatch.setattr(
        engine_module,
        "StreamTracker",
        lambda settings: SimpleNamespace(settings=settings, trajectories={}),
    )
    monkeypatch.setattr(
        engine_module,
        "ZoneCounter",
        lambda geometry: SimpleNamespace(
            geometry=geometry,
            in_count=0,
            out_count=0,
            occupancy=0,
        ),
    )

    ProcessingEngine._reset_context_runtime(engine, context)

    assert context.latest_frame is None
    assert context.latest_rendered_frame is None
    assert context.latest_rendered_version is None
    assert context.latest_detections is None
    assert context.render_revision == 22


def test_cache_survives_reset_until_first_new_frame(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.runtime_generation = 3
    context.render_revision = 15
    old_cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(3, 15),
        jpeg=b"old-jpeg",
    )
    cache = {context.stream_id: old_cached}

    monkeypatch.setattr(engine, "_tracker_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(engine, "_zone_geometry", lambda: SimpleNamespace())
    monkeypatch.setattr(
        engine_module,
        "StreamTracker",
        lambda settings: SimpleNamespace(settings=settings, trajectories={}),
    )
    monkeypatch.setattr(
        engine_module,
        "ZoneCounter",
        lambda geometry: SimpleNamespace(
            geometry=geometry,
            in_count=0,
            out_count=0,
            occupancy=0,
        ),
    )

    ProcessingEngine._reset_context_runtime(engine, context)
    reset_snapshot = snapshot_stream_frame(context, cached_frame=old_cached)
    reset_update = update_stream_frame_cache(cache, reset_snapshot)

    assert cache[context.stream_id] == old_cached
    assert reset_update.render_jpeg is None
    assert reset_update.clear_image is False
    assert reset_update.show_waiting is False

    monkeypatch.setattr(ui_module, "encode_frame_jpeg", lambda frame: b"new-jpeg")
    context.publish_rendered_frame(
        np.full((4, 4, 3), 22, dtype=np.uint8),
        np.full((4, 4, 3), 33, dtype=np.uint8),
        Detections.empty(),
    )

    new_snapshot = snapshot_stream_frame(context, cached_frame=cache[context.stream_id])
    new_update = update_stream_frame_cache(cache, new_snapshot)

    assert new_update.render_jpeg == b"new-jpeg"
    assert cache[context.stream_id].jpeg == b"new-jpeg"
    assert cache[context.stream_id].frame_version == (4, 16)


def test_queue_snapshot_stats_blocks_on_queue_lock() -> None:
    queue = LatestFrameQueue()
    queue.received = 12
    queue.dropped = 4
    attempted = threading.Event()
    finished = threading.Event()
    result: list[tuple[int, int]] = []

    def worker() -> None:
        attempted.set()
        result.append(queue.snapshot_stats())
        finished.set()

    with queue._lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert attempted.wait(1.0)
        assert not finished.wait(0.05)

    assert finished.wait(1.0)
    thread.join(timeout=1.0)
    assert result == [(12, 4)]


def test_dropped_rate_uses_atomic_snapshot(monkeypatch) -> None:
    queue = LatestFrameQueue()

    monkeypatch.setattr(queue, "snapshot_stats", lambda: (10, 3))

    assert queue.dropped_rate == pytest.approx(0.3)


def test_reset_counters_uses_context_lock() -> None:
    context = _context()
    engine = _minimal_engine(context)
    counter = FakeCounter()
    context.counter = counter
    attempted = threading.Event()
    finished = threading.Event()

    def worker() -> None:
        attempted.set()
        engine.reset_counters(context.stream_id)
        finished.set()

    with context.lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert attempted.wait(1.0)
        assert counter.reset_calls == 0
        assert not finished.wait(0.05)

    assert finished.wait(1.0)
    thread.join(timeout=1.0)
    assert counter.reset_calls == 1


def test_reset_counters_rerenders_current_frame_with_zero_counts(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    counter = FakeCounter()
    raw = np.full((4, 4, 3), 13, dtype=np.uint8)
    detections = Detections.empty()
    rendered = np.full((4, 4, 3), 77, dtype=np.uint8)
    calls = []
    context.counter = counter
    context.tracker = SimpleNamespace(trajectories={1: [(1, 2), (2, 3)]})
    context.latest_frame = raw
    context.latest_detections = detections
    context.latest_rendered_version = (12, 3)
    context.render_revision = 3
    context.runtime_generation = 12
    context.metrics.processed_frames = 9
    context.options = StreamOptions(
        detection_enabled=False,
        tracking_enabled=True,
        counting_enabled=True,
    )

    def fake_render(frame, rendered_detections, **kwargs):
        calls.append((frame, rendered_detections, kwargs))
        return rendered

    monkeypatch.setattr(engine_module, "render_frame", fake_render)

    engine.reset_counters(context.stream_id)

    assert counter.reset_calls == 1
    assert len(calls) == 1
    frame, rendered_detections, kwargs = calls[0]
    assert frame is raw
    assert rendered_detections is detections
    assert kwargs["trajectories"] is context.tracker.trajectories
    assert kwargs["geometry"] is counter.geometry
    assert kwargs["in_count"] == 0
    assert kwargs["out_count"] == 0
    assert kwargs["occupancy"] == 0
    assert kwargs["show_detections"] is False
    assert kwargs["show_tracking"] is True
    assert kwargs["show_counting"] is True
    assert context.metrics.processed_frames == 9
    assert context.latest_rendered_frame is rendered
    assert context.latest_rendered_version == (12, 4)
    assert context.render_revision == 4


def test_reset_counters_without_current_frame_does_not_rerender(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    counter = FakeCounter()
    previous_rendered = np.full((4, 4, 3), 19, dtype=np.uint8)
    context.counter = counter
    context.latest_rendered_frame = previous_rendered
    context.latest_rendered_version = (2, 7)
    context.render_revision = 7

    monkeypatch.setattr(
        engine_module,
        "render_frame",
        lambda *_args, **_kwargs: pytest.fail("missing current frame rendered"),
    )

    engine.reset_counters(context.stream_id)

    assert counter.reset_calls == 1
    assert context.latest_rendered_frame is previous_rendered
    assert context.latest_rendered_version == (2, 7)
    assert context.render_revision == 7


def test_reset_counters_handles_none() -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.counter = None

    engine.reset_counters(context.stream_id)


def test_snapshot_for_rebuild_captures_stream_restore_state() -> None:
    context = _context()
    engine = _minimal_engine(context)
    options = StreamOptions(
        confidence=0.55,
        iou=0.75,
        detection_enabled=False,
        tracking_enabled=True,
        counting_enabled=False,
    )
    tracker_settings = ByteTrackSettings(
        track_activation_threshold=0.45,
        lost_track_buffer=120,
        minimum_matching_threshold=0.66,
    )
    context.options = options
    context.tracker = SimpleNamespace(settings=tracker_settings)
    context.force_state(StreamState.ACTIVE)

    snapshot = engine.snapshot_for_rebuild()[0]

    assert snapshot.stream_id == context.stream_id
    assert snapshot.source is context.source
    assert snapshot.options == options
    assert snapshot.options is not options
    context.options.confidence = 0.15
    assert snapshot.options.confidence == 0.55
    assert snapshot.tracker_settings == tracker_settings
    assert snapshot.tracker_settings is not tracker_settings
    assert snapshot.was_running is True
