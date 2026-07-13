from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import vision_track.engine as engine_module
import vision_track.scheduler as scheduler_module
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
from vision_track.ui import snapshot_stream_metrics


class FakeReader:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_calls += 1


class FakeCounter:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.reset_tracking_state_calls = 0
        self.geometry = ZoneGeometry()
        self.in_count = 4
        self.out_count = 5
        self.occupancy = 6

    def reset(self) -> None:
        self.reset_calls += 1
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0

    def reset_tracking_state(self) -> None:
        self.reset_tracking_state_calls += 1


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


def test_blocked_render_does_not_block_metrics_or_stop(monkeypatch) -> None:
    context = _context()
    context.force_state(StreamState.ACTIVE)
    context.runtime_generation = 4
    context.options = StreamOptions(
        detection_enabled=False,
        tracking_enabled=False,
        counting_enabled=False,
    )
    context.reader = FakeReader()
    engine = _minimal_engine(context)
    scheduler = _scheduler(context)
    packet = _packet(runtime_generation=4)
    render_started = threading.Event()
    release_render = threading.Event()
    metrics_finished = threading.Event()
    stop_finished = threading.Event()
    finalize_result: list[bool] = []

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    monkeypatch.setattr(scheduler_module, "render_frame", blocked_render)
    finalize_thread = threading.Thread(
        target=lambda: finalize_result.append(
            scheduler._finalize(context, packet, _result())
        ),
        daemon=True,
    )
    metrics_thread = threading.Thread(
        target=lambda: (snapshot_stream_metrics(context), metrics_finished.set()),
        daemon=True,
    )
    stop_thread = threading.Thread(
        target=lambda: (engine.stop(context.stream_id), stop_finished.set()),
        daemon=True,
    )

    try:
        finalize_thread.start()
        assert render_started.wait(timeout=1.0)
        metrics_thread.start()
        stop_thread.start()
        assert metrics_finished.wait(timeout=0.5)
        assert stop_finished.wait(timeout=0.5)
    finally:
        release_render.set()
        finalize_thread.join(timeout=1.0)
        metrics_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

    assert finalize_result == [False]
    assert context.latest_rendered_frame is None
    assert context.metrics.processed_frames == 0


def test_render_result_is_discarded_after_generation_change(monkeypatch) -> None:
    context = _context()
    context.runtime_generation = 5
    context.options = StreamOptions(
        detection_enabled=False,
        tracking_enabled=False,
        counting_enabled=False,
    )
    scheduler = _scheduler(context)
    packet = _packet(runtime_generation=5)
    render_started = threading.Event()
    release_render = threading.Event()
    generation_changed = threading.Event()
    finalize_result: list[bool] = []

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    def change_generation() -> None:
        with context.lock:
            context.runtime_generation += 1
        generation_changed.set()

    monkeypatch.setattr(scheduler_module, "render_frame", blocked_render)
    finalize_thread = threading.Thread(
        target=lambda: finalize_result.append(
            scheduler._finalize(context, packet, _result())
        ),
        daemon=True,
    )
    generation_thread = threading.Thread(target=change_generation, daemon=True)

    try:
        finalize_thread.start()
        assert render_started.wait(timeout=1.0)
        generation_thread.start()
        assert generation_changed.wait(timeout=0.5)
    finally:
        release_render.set()
        finalize_thread.join(timeout=1.0)
        generation_thread.join(timeout=1.0)

    assert finalize_result == [False]
    assert context.latest_rendered_frame is None
    assert context.latest_rendered_version is None
    assert context.metrics.processed_frames == 0


def test_counter_reset_invalidates_in_flight_render(monkeypatch) -> None:
    context = _context()
    context.runtime_generation = 6
    context.options = StreamOptions(
        detection_enabled=False,
        tracking_enabled=False,
        counting_enabled=False,
    )
    context.counter = FakeCounter()
    engine = _minimal_engine(context)
    scheduler = _scheduler(context)
    packet = _packet(runtime_generation=6)
    render_started = threading.Event()
    release_render = threading.Event()
    reset_finished = threading.Event()
    finalize_result: list[bool] = []

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    monkeypatch.setattr(scheduler_module, "render_frame", blocked_render)
    finalize_thread = threading.Thread(
        target=lambda: finalize_result.append(
            scheduler._finalize(context, packet, _result())
        ),
        daemon=True,
    )
    reset_thread = threading.Thread(
        target=lambda: (engine.reset_counters(context.stream_id), reset_finished.set()),
        daemon=True,
    )

    try:
        finalize_thread.start()
        assert render_started.wait(timeout=1.0)
        reset_thread.start()
        assert reset_finished.wait(timeout=0.5)
    finally:
        release_render.set()
        finalize_thread.join(timeout=1.0)
        reset_thread.join(timeout=1.0)

    assert context.render_state_revision == 1
    assert finalize_result == [False]
    assert context.latest_rendered_frame is None
    assert context.metrics.processed_frames == 0


@pytest.mark.parametrize("state_change", ["options", "tracker"])
def test_render_state_change_invalidates_in_flight_render(
    monkeypatch,
    state_change: str,
) -> None:
    context = _context()
    context.runtime_generation = 7
    context.options = StreamOptions(
        detection_enabled=False,
        tracking_enabled=False,
        counting_enabled=False,
    )
    context.tracker = SimpleNamespace(
        settings=ByteTrackSettings(),
        trajectories={1: [(1, 2), (2, 3)]},
    )
    context.counter = FakeCounter()
    engine = _minimal_engine(context)
    scheduler = _scheduler(context)
    packet = _packet(runtime_generation=7)
    render_started = threading.Event()
    release_render = threading.Event()
    state_changed = threading.Event()
    finalize_result: list[bool] = []

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    def change_state() -> None:
        if state_change == "options":
            engine.update_options(context.stream_id, detection_enabled=True)
        else:
            engine.update_tracker(context.stream_id, lost_track_buffer=45)
        state_changed.set()

    monkeypatch.setattr(scheduler_module, "render_frame", blocked_render)
    monkeypatch.setattr(
        engine_module,
        "StreamTracker",
        lambda settings: SimpleNamespace(settings=settings, trajectories={}),
    )
    finalize_thread = threading.Thread(
        target=lambda: finalize_result.append(
            scheduler._finalize(context, packet, _result())
        ),
        daemon=True,
    )
    state_thread = threading.Thread(target=change_state, daemon=True)

    try:
        finalize_thread.start()
        assert render_started.wait(timeout=1.0)
        state_thread.start()
        assert state_changed.wait(timeout=0.5)
    finally:
        release_render.set()
        finalize_thread.join(timeout=1.0)
        state_thread.join(timeout=1.0)

    assert context.render_state_revision == 1
    assert finalize_result == [False]
    assert context.latest_rendered_frame is None
    assert context.latest_rendered_version is None
    assert context.metrics.processed_frames == 0


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
    assert kwargs["trajectories"] == {1: ((1, 2), (2, 3))}
    assert kwargs["trajectories"] is not context.tracker.trajectories
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


def test_reset_rerender_does_not_block_metrics_or_stop(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.counter = FakeCounter()
    context.tracker = SimpleNamespace(trajectories={})
    context.reader = FakeReader()
    context.runtime_generation = 10
    context.render_revision = 2
    context.latest_frame = np.full((4, 4, 3), 13, dtype=np.uint8)
    context.latest_detections = Detections.empty()
    previous_rendered = np.full((4, 4, 3), 14, dtype=np.uint8)
    context.latest_rendered_frame = previous_rendered
    context.latest_rendered_version = (10, 2)
    render_started = threading.Event()
    release_render = threading.Event()
    metrics_finished = threading.Event()
    stop_finished = threading.Event()
    reset_finished = threading.Event()
    snapshots = []

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    monkeypatch.setattr(engine_module, "render_frame", blocked_render)
    reset_thread = threading.Thread(
        target=lambda: (engine.reset_counters(context.stream_id), reset_finished.set()),
        daemon=True,
    )
    metrics_thread = threading.Thread(
        target=lambda: (
            snapshots.append(snapshot_stream_metrics(context)),
            metrics_finished.set(),
        ),
        daemon=True,
    )
    stop_thread = threading.Thread(
        target=lambda: (engine.stop(context.stream_id), stop_finished.set()),
        daemon=True,
    )

    try:
        reset_thread.start()
        assert render_started.wait(timeout=1.0)
        metrics_thread.start()
        stop_thread.start()
        assert metrics_finished.wait(timeout=0.5)
        assert stop_finished.wait(timeout=0.5)
        assert not reset_finished.is_set()
    finally:
        release_render.set()
        reset_thread.join(timeout=1.0)
        metrics_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

    assert snapshots[0].in_count == 0
    assert snapshots[0].out_count == 0
    assert snapshots[0].occupancy == 0
    assert reset_finished.is_set()
    assert context.latest_rendered_frame is previous_rendered
    assert context.latest_rendered_version == (10, 2)
    assert context.render_revision == 2


def test_reset_rerender_is_discarded_after_render_state_change(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.counter = FakeCounter()
    context.tracker = SimpleNamespace(trajectories={})
    context.runtime_generation = 11
    context.render_revision = 3
    context.latest_frame = np.full((4, 4, 3), 13, dtype=np.uint8)
    context.latest_detections = Detections.empty()
    previous_rendered = np.full((4, 4, 3), 14, dtype=np.uint8)
    context.latest_rendered_frame = previous_rendered
    context.latest_rendered_version = (11, 3)
    render_started = threading.Event()
    release_render = threading.Event()
    options_updated = threading.Event()

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    monkeypatch.setattr(engine_module, "render_frame", blocked_render)
    reset_thread = threading.Thread(
        target=lambda: engine.reset_counters(context.stream_id),
        daemon=True,
    )
    options_thread = threading.Thread(
        target=lambda: (
            engine.update_options(context.stream_id, detection_enabled=False),
            options_updated.set(),
        ),
        daemon=True,
    )

    try:
        reset_thread.start()
        assert render_started.wait(timeout=1.0)
        options_thread.start()
        assert options_updated.wait(timeout=0.5)
    finally:
        release_render.set()
        reset_thread.join(timeout=1.0)
        options_thread.join(timeout=1.0)

    assert context.render_state_revision == 2
    assert context.latest_rendered_frame is previous_rendered
    assert context.latest_rendered_version == (11, 3)
    assert context.render_revision == 3


def test_reset_rerender_does_not_overwrite_newer_publication(monkeypatch) -> None:
    context = _context()
    engine = _minimal_engine(context)
    context.counter = FakeCounter()
    context.tracker = SimpleNamespace(trajectories={})
    context.runtime_generation = 12
    context.render_revision = 4
    context.latest_frame = np.full((4, 4, 3), 13, dtype=np.uint8)
    context.latest_detections = Detections.empty()
    context.latest_rendered_frame = np.full((4, 4, 3), 14, dtype=np.uint8)
    context.latest_rendered_version = (12, 4)
    render_started = threading.Event()
    release_render = threading.Event()
    newer_published = threading.Event()
    newer_raw = np.full((4, 4, 3), 21, dtype=np.uint8)
    newer_rendered = np.full((4, 4, 3), 22, dtype=np.uint8)
    newer_detections = Detections.empty()

    def blocked_render(*_args, **_kwargs):
        render_started.set()
        release_render.wait(timeout=2.0)
        return np.full((4, 4, 3), 99, dtype=np.uint8)

    def publish_newer() -> None:
        context.publish_rendered_frame(newer_raw, newer_rendered, newer_detections)
        newer_published.set()

    monkeypatch.setattr(engine_module, "render_frame", blocked_render)
    reset_thread = threading.Thread(
        target=lambda: engine.reset_counters(context.stream_id),
        daemon=True,
    )
    publish_thread = threading.Thread(target=publish_newer, daemon=True)

    try:
        reset_thread.start()
        assert render_started.wait(timeout=1.0)
        publish_thread.start()
        assert newer_published.wait(timeout=0.5)
    finally:
        release_render.set()
        reset_thread.join(timeout=1.0)
        publish_thread.join(timeout=1.0)

    assert context.latest_frame is newer_raw
    assert context.latest_rendered_frame is newer_rendered
    assert context.latest_detections is newer_detections
    assert context.latest_rendered_version == (12, 5)
    assert context.render_revision == 5


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
