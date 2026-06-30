from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import vision_track.ui as ui_module
from vision_track.context import StreamContext, StreamOptions
from vision_track.lifecycle import StreamState
from vision_track.sources import SourceType, VideoSource
from vision_track.tracking import ByteTrackSettings
from vision_track.ui import (
    CachedStreamFrame,
    StreamControlSnapshot,
    StreamFrameSnapshot,
    StreamMetricsSnapshot,
    UI_CANVAS_HEIGHT,
    UI_CANVAS_WIDTH,
    clear_stream_frame_cache,
    encode_frame_jpeg,
    fit_frame_to_canvas,
    prune_stream_frame_cache,
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    snapshot_stream_controls,
    snapshot_stream_frame,
    snapshot_stream_identity,
    snapshot_stream_metrics,
    stream_grid_columns,
    stream_source_token,
    update_stream_frame_cache,
    waiting_slot_transition,
)


def _nonzero_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(frame != 0, axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _snapshot_for_cache(
    *,
    stream_id: str = "stream-1",
    source_token: str = "a" * 64,
    frame_version: tuple[int, int] | None = (0, 1),
    frame_jpeg: bytes | None = None,
) -> StreamFrameSnapshot:
    return StreamFrameSnapshot(
        stream_id=stream_id,
        source_token=source_token,
        frame_version=frame_version,
        frame_jpeg=frame_jpeg,
    )


def _metrics_snapshot(
    *,
    actual_backend: str | None = None,
    actual_device: str | None = None,
    actual_provider: str | None = None,
) -> StreamMetricsSnapshot:
    source = VideoSource.from_uri("video.mp4")
    return StreamMetricsSnapshot(
        stream_id="stream-1",
        display_name=source.display_name,
        source_type=source.source_type,
        source_token=stream_source_token(source),
        state=StreamState.ACTIVE,
        error=None,
        processed_frames=12,
        fps=30.0,
        inference_latency_ms=4.0,
        end_to_end_latency_ms=8.0,
        dropped_rate=0.0,
        in_count=1,
        out_count=2,
        occupancy=3,
        actual_backend=actual_backend,
        actual_device=actual_device,
        actual_provider=actual_provider,
    )


def _control_snapshot(
    *,
    source_type: SourceType = SourceType.LOCAL,
    state: StreamState = StreamState.ACTIVE,
    processed_frames: int = 0,
    actual_backend: str | None = None,
    actual_device: str | None = None,
    actual_provider: str | None = None,
) -> StreamControlSnapshot:
    return StreamControlSnapshot(
        stream_id="stream-1",
        source_type=source_type,
        state=state,
        processed_frames=processed_frames,
        confidence=0.35,
        iou=0.5,
        detection_enabled=True,
        tracking_enabled=True,
        counting_enabled=True,
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        actual_backend=actual_backend,
        actual_device=actual_device,
        actual_provider=actual_provider,
    )


def test_stream_grid_uses_one_bounded_column_for_single_stream() -> None:
    assert stream_grid_columns(1) == 1
    assert single_stream_column_weights(1) == [1.0, 1.6, 1.0]


def test_stream_grid_uses_two_columns_for_multiple_streams() -> None:
    assert stream_grid_columns(2) == 2
    assert stream_grid_columns(5) == 2


def test_snapshot_stream_identity_captures_source_under_context_lock() -> None:
    source = VideoSource.from_uri("video.mp4", display_name="Lobby")
    context = StreamContext("stream-1", source)
    attempted = threading.Event()
    finished = threading.Event()
    snapshots = []

    def worker() -> None:
        attempted.set()
        snapshots.append(snapshot_stream_identity(context))
        finished.set()

    with context.lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert attempted.wait(1.0)
        assert not finished.wait(0.05)

    assert finished.wait(1.0)
    thread.join(timeout=1.0)
    snapshot = snapshots[0]
    assert snapshot.stream_id == "stream-1"
    assert snapshot.source is source
    assert snapshot.source_token == stream_source_token(source)
    assert snapshot.display_name == "Lobby"


def test_snapshot_stream_controls_captures_control_scalars() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("rtsp://example.test/live"))
    context.options = StreamOptions(
        confidence=0.45,
        iou=0.65,
        detection_enabled=False,
        tracking_enabled=True,
        counting_enabled=False,
    )
    context.force_state(StreamState.RECONNECTING)
    context.metrics.processed_frames = 42
    context.tracker = SimpleNamespace(
        settings=ByteTrackSettings(
            track_activation_threshold=0.4,
            lost_track_buffer=90,
            minimum_matching_threshold=0.7,
        )
    )
    context.actual_backend = "onnxruntime"
    context.actual_device = "cpu"
    context.actual_provider = "CPUExecutionProvider"

    snapshot = snapshot_stream_controls(context)

    assert snapshot.stream_id == "stream-1"
    assert snapshot.source_type is SourceType.RTSP
    assert snapshot.state is StreamState.RECONNECTING
    assert snapshot.processed_frames == 42
    assert snapshot.confidence == 0.45
    assert snapshot.iou == 0.65
    assert snapshot.detection_enabled is False
    assert snapshot.tracking_enabled is True
    assert snapshot.counting_enabled is False
    assert snapshot.track_activation_threshold == 0.4
    assert snapshot.lost_track_buffer == 90
    assert snapshot.minimum_matching_threshold == 0.7
    assert snapshot.actual_backend == "onnxruntime"
    assert snapshot.actual_device == "cpu"
    assert snapshot.actual_provider == "CPUExecutionProvider"


def test_runtime_backend_summary_is_pending_before_first_inference() -> None:
    summary = runtime_backend_summary(
        None,
        requested_backend="pytorch",
        requested_device="cuda",
    )
    assert "Requested backend `pytorch`" in summary
    assert "pending first inference" in summary


def test_runtime_backend_summary_reports_actual_provider_from_control_snapshot() -> None:
    control = _control_snapshot(
        actual_backend="onnxruntime",
        actual_device="cpu",
        actual_provider="CPUExecutionProvider",
    )
    summary = runtime_backend_summary(
        control,
        requested_backend="onnxruntime",
        requested_device="cuda",
    )
    assert "Actual backend `onnxruntime`" in summary
    assert "provider `CPUExecutionProvider`" in summary


def test_runtime_backend_summary_reports_metrics_snapshot_actual_provider() -> None:
    snapshot = _metrics_snapshot(
        actual_backend="onnxruntime",
        actual_device="cpu",
        actual_provider="CPUExecutionProvider",
    )
    summary = runtime_backend_summary(
        snapshot,
        requested_backend="onnxruntime",
        requested_device="cuda",
    )
    assert "Actual backend `onnxruntime`" in summary
    assert "provider `CPUExecutionProvider`" in summary


def test_local_completed_stream_uses_replay_label() -> None:
    control = _control_snapshot(state=StreamState.EOF, processed_frames=1)

    assert replay_button_label(control) == "Replay"


def test_fit_landscape_frame_to_canvas_without_black_bars() -> None:
    frame = np.full((1080, 1920, 3), 64, dtype=np.uint8)
    original = frame.copy()

    canvas = fit_frame_to_canvas(frame)

    assert canvas.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)
    assert canvas.dtype == np.uint8
    assert canvas.flags.c_contiguous
    assert np.all(canvas != 0)
    np.testing.assert_array_equal(frame, original)


def test_fit_portrait_frame_to_canvas_with_side_bars() -> None:
    frame = np.full((1920, 1080, 3), 80, dtype=np.uint8)

    canvas = fit_frame_to_canvas(frame)

    x0, y0, x1, y1 = _nonzero_bbox(canvas)
    assert canvas.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)
    assert y0 == 0
    assert y1 == UI_CANVAS_HEIGHT
    assert x0 > 0
    assert x1 < UI_CANVAS_WIDTH
    assert np.all(canvas[:, :x0] == 0)
    assert np.all(canvas[:, x1:] == 0)
    assert np.any(canvas[:, x0:x1] != 0)


def test_fit_square_frame_to_canvas_with_centered_side_bars() -> None:
    frame = np.full((400, 400, 3), 96, dtype=np.uint8)

    canvas = fit_frame_to_canvas(frame)

    x0, y0, x1, y1 = _nonzero_bbox(canvas)
    assert canvas.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)
    assert (x1 - x0, y1 - y0) == (UI_CANVAS_HEIGHT, UI_CANVAS_HEIGHT)
    assert y0 == 0
    assert y1 == UI_CANVAS_HEIGHT
    assert x0 == (UI_CANVAS_WIDTH - UI_CANVAS_HEIGHT) // 2
    assert x1 == x0 + UI_CANVAS_HEIGHT
    assert np.all(canvas[:, :x0] == 0)
    assert np.all(canvas[:, x1:] == 0)


@pytest.mark.parametrize(
    "frame",
    [
        np.array([], dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
        np.zeros((10, 0, 3), dtype=np.uint8),
    ],
)
def test_fit_frame_to_canvas_rejects_invalid_input(frame: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_frame_to_canvas(frame)


@pytest.mark.parametrize(
    ("shape", "expected_width", "expected_height"),
    [
        ((1920, 1080, 3), 304, UI_CANVAS_HEIGHT),
        ((400, 400, 3), UI_CANVAS_HEIGHT, UI_CANVAS_HEIGHT),
    ],
)
def test_fit_frame_to_canvas_preserves_content_aspect(
    shape: tuple[int, int, int],
    expected_width: int,
    expected_height: int,
) -> None:
    canvas = fit_frame_to_canvas(np.full(shape, 100, dtype=np.uint8))

    x0, y0, x1, y1 = _nonzero_bbox(canvas)
    actual_width = x1 - x0
    actual_height = y1 - y0
    expected_x0 = (UI_CANVAS_WIDTH - expected_width) // 2
    expected_y0 = (UI_CANVAS_HEIGHT - expected_height) // 2

    assert abs(actual_width - expected_width) <= 1
    assert abs(actual_height - expected_height) <= 1
    assert abs(x0 - expected_x0) <= 1
    assert abs(y0 - expected_y0) <= 1


def test_encode_frame_jpeg_returns_decodable_bytes() -> None:
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)

    payload = encode_frame_jpeg(frame)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert isinstance(payload, bytes)
    assert payload
    assert decoded is not None
    assert decoded.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)


def test_frame_snapshot_uses_published_version() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = (7, 12)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    context.metrics.processed_frames = 999
    context.runtime_generation = 100

    snapshot = snapshot_stream_frame(context, cached_frame=None)

    assert snapshot.frame_version == (7, 12)


def test_unchanged_published_frame_skips_copy_and_encode(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = (3, 11)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(3, 11),
        jpeg=b"cached",
    )
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)

    snapshot = snapshot_stream_frame(context, cached_frame=cached)

    assert calls == {"copy": 0, "encode": 0}
    assert snapshot.frame_jpeg is None


def test_runtime_generation_change_without_new_published_frame_preserves_cache(
    monkeypatch,
) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.runtime_generation = 4
    context.latest_rendered_version = (4, 20)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(4, 20),
        jpeg=b"cached",
    )
    cache = {"stream-1": cached}
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)
    context.runtime_generation = 5

    snapshot = snapshot_stream_frame(context, cached_frame=cached)
    update = update_stream_frame_cache(cache, snapshot)

    assert snapshot.frame_version == (4, 20)
    assert calls == {"copy": 0, "encode": 0}
    assert cache["stream-1"] == cached
    assert update.render_jpeg is None
    assert update.clear_image is False
    assert update.show_waiting is False


def test_new_published_frame_encodes_once(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = (4, 21)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(4, 20),
        jpeg=b"cached",
    )
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)

    snapshot = snapshot_stream_frame(context, cached_frame=cached)

    assert calls == {"copy": 1, "encode": 1}
    assert snapshot.frame_version == (4, 21)
    assert snapshot.frame_jpeg == b"encoded"


def test_reset_counter_republished_version_encodes_once(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = (4, 22)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(4, 21),
        jpeg=b"cached",
    )
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)

    snapshot = snapshot_stream_frame(context, cached_frame=cached)

    assert calls == {"copy": 1, "encode": 1}
    assert snapshot.frame_version == (4, 22)
    assert snapshot.frame_jpeg == b"encoded"


def test_new_generation_with_same_render_revision_encodes_once(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = (5, 0)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(4, 0),
        jpeg=b"cached",
    )
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)

    snapshot = snapshot_stream_frame(context, cached_frame=cached)

    assert snapshot.frame_version == (5, 0)
    assert snapshot.frame_version != cached.frame_version
    assert calls == {"copy": 1, "encode": 1}
    assert snapshot.frame_jpeg == b"encoded"


def test_no_published_version_skips_copy_and_encode(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.latest_rendered_version = None
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    calls = {"copy": 0, "encode": 0}

    def fake_copy(frame: np.ndarray) -> np.ndarray:
        calls["copy"] += 1
        return frame.copy()

    def fake_encode(frame: np.ndarray) -> bytes:
        calls["encode"] += 1
        return b"encoded"

    monkeypatch.setattr(ui_module, "_copy_frame_for_ui", fake_copy)
    monkeypatch.setattr(ui_module, "encode_frame_jpeg", fake_encode)

    snapshot = snapshot_stream_frame(context, cached_frame=None)

    assert calls == {"copy": 0, "encode": 0}
    assert snapshot.frame_jpeg is None


def test_metrics_snapshot_does_not_access_frame_encoder(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.force_state(StreamState.ACTIVE)
    context.error = "temporary"
    context.metrics.processed_frames = 12
    context.metrics.fps = 24.0
    context.metrics.inference_latency_ms = 7.5
    context.metrics.end_to_end_latency_ms = 12.5
    context.queue.received = 10
    context.queue.dropped = 2
    context.counter = SimpleNamespace(in_count=3, out_count=4, occupancy=5)
    context.actual_backend = "pytorch"
    context.actual_device = "cuda"
    context.actual_provider = None
    context.latest_rendered_version = (1, 2)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)

    monkeypatch.setattr(
        ui_module,
        "_copy_frame_for_ui",
        lambda frame: pytest.fail("metrics snapshot copied a frame"),
    )
    monkeypatch.setattr(
        ui_module,
        "encode_frame_jpeg",
        lambda frame: pytest.fail("metrics snapshot encoded a frame"),
    )

    snapshot = snapshot_stream_metrics(context)

    assert snapshot.stream_id == "stream-1"
    assert snapshot.display_name == "video.mp4"
    assert snapshot.source_type is SourceType.LOCAL
    assert snapshot.source_token == stream_source_token(context.source)
    assert snapshot.state is StreamState.ACTIVE
    assert snapshot.error == "temporary"
    assert snapshot.processed_frames == 12
    assert snapshot.fps == 24.0
    assert snapshot.inference_latency_ms == 7.5
    assert snapshot.end_to_end_latency_ms == 12.5
    assert snapshot.dropped_rate == 0.2
    assert snapshot.in_count == 3
    assert snapshot.out_count == 4
    assert snapshot.occupancy == 5
    assert snapshot.actual_backend == "pytorch"
    assert snapshot.actual_device == "cuda"
    assert snapshot.actual_provider is None


def test_metrics_snapshot_uses_atomic_queue_stats(monkeypatch) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    calls = {"snapshot_stats": 0}

    def fake_snapshot_stats() -> tuple[int, int]:
        calls["snapshot_stats"] += 1
        return 10, 3

    monkeypatch.setattr(context.queue, "snapshot_stats", fake_snapshot_stats)

    snapshot = snapshot_stream_metrics(context)

    assert calls == {"snapshot_stats": 1}
    assert snapshot.dropped_rate == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("previous_visible", "desired_visible", "expected"),
    [
        (False, False, None),
        (True, True, None),
        (False, True, True),
        (True, False, False),
    ],
)
def test_waiting_slot_transition(
    previous_visible: bool,
    desired_visible: bool,
    expected: bool | None,
) -> None:
    assert waiting_slot_transition(previous_visible, desired_visible) is expected


@pytest.mark.parametrize("state", [StreamState.STOPPED, StreamState.CONNECTING])
def test_frame_cache_keeps_image_when_state_changes_without_new_frame(
    state: StreamState,
    monkeypatch,
) -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.force_state(state)
    context.latest_rendered_version = (1, 2)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(1, 2),
        jpeg=b"cached",
    )
    cache = {"stream-1": cached}

    monkeypatch.setattr(
        ui_module,
        "encode_frame_jpeg",
        lambda frame: pytest.fail("state-only change encoded a frame"),
    )

    snapshot = snapshot_stream_frame(context, cached_frame=cached)
    update = update_stream_frame_cache(cache, snapshot)

    assert cache["stream-1"].jpeg == b"cached"
    assert update.render_jpeg is None
    assert update.clear_image is False
    assert update.show_waiting is False


def test_frame_cache_keeps_image_after_reset_without_frame() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    cached = CachedStreamFrame(
        source_token=stream_source_token(context.source),
        frame_version=(1, 2),
        jpeg=b"cached",
    )
    cache = {"stream-1": cached}

    snapshot = snapshot_stream_frame(context, cached_frame=cached)
    update = update_stream_frame_cache(cache, snapshot)

    assert cache["stream-1"] == cached
    assert update.render_jpeg is None
    assert update.clear_image is False
    assert update.show_waiting is False


def test_frame_cache_survives_simulated_full_rerun() -> None:
    cache: dict[str, CachedStreamFrame] = {}
    first = _snapshot_for_cache(frame_version=(1, 2), frame_jpeg=b"first")

    first_update = update_stream_frame_cache(cache, first)
    second_update = update_stream_frame_cache(
        cache,
        _snapshot_for_cache(frame_version=(1, 2), frame_jpeg=None),
    )

    assert first_update.render_jpeg == b"first"
    assert cache["stream-1"].jpeg == b"first"
    assert second_update.render_jpeg is None
    assert second_update.clear_image is False
    assert second_update.show_waiting is False


def test_frame_cache_clears_stale_image_for_source_replacement_without_frame() -> None:
    cache = {
        "stream-1": CachedStreamFrame(
            source_token="a" * 64,
            frame_version=(1, 2),
            jpeg=b"cached",
        )
    }

    update = update_stream_frame_cache(
        cache,
        _snapshot_for_cache(source_token="b" * 64, frame_jpeg=None),
    )

    assert "stream-1" not in cache
    assert update.render_jpeg is None
    assert update.clear_image is True
    assert update.show_waiting is True


def test_frame_cache_replaces_stale_image_when_new_source_frame_is_ready() -> None:
    cache = {
        "stream-1": CachedStreamFrame(
            source_token="a" * 64,
            frame_version=(1, 2),
            jpeg=b"cached",
        )
    }

    update = update_stream_frame_cache(
        cache,
        _snapshot_for_cache(source_token="b" * 64, frame_version=(2, 1), frame_jpeg=b"new"),
    )

    assert cache["stream-1"].source_token == "b" * 64
    assert cache["stream-1"].frame_version == (2, 1)
    assert cache["stream-1"].jpeg == b"new"
    assert update.render_jpeg == b"new"
    assert update.clear_image is False
    assert update.show_waiting is False


def test_frame_cache_rejects_versionless_frame_update() -> None:
    cache: dict[str, CachedStreamFrame] = {}

    with pytest.raises(RuntimeError):
        update_stream_frame_cache(
            cache,
            _snapshot_for_cache(frame_version=None, frame_jpeg=b"invalid"),
        )


def test_clear_stream_frame_cache_removes_existing_and_ignores_missing() -> None:
    cache = {
        "stream-1": CachedStreamFrame(
            source_token="a" * 64,
            frame_version=(1, 2),
            jpeg=b"cached",
        )
    }

    clear_stream_frame_cache(cache, "stream-1")
    clear_stream_frame_cache(cache, "missing")

    assert cache == {}


def test_prune_stream_frame_cache_keeps_only_active_matching_sources() -> None:
    cache = {
        "matching": CachedStreamFrame("a" * 64, (1, 1), b"matching"),
        "mismatch": CachedStreamFrame("b" * 64, (1, 1), b"mismatch"),
        "removed": CachedStreamFrame("c" * 64, (1, 1), b"removed"),
    }

    prune_stream_frame_cache(cache, {"matching": "a" * 64, "mismatch": "x" * 64})

    assert cache == {
        "matching": CachedStreamFrame("a" * 64, (1, 1), b"matching")
    }


def test_stream_source_token_is_stable_hashed_source_identity() -> None:
    local = VideoSource.from_uri("video.mp4")
    same_local = VideoSource.from_uri("video.mp4")
    other_uri = VideoSource.from_uri("other.mp4")
    same_uri_different_type = VideoSource(
        uri="video.mp4",
        source_type=SourceType.RTSP,
        display_name="video.mp4",
    )
    remote = VideoSource.from_uri("rtsp://user:pass@example.test/stream?token=secret")

    token = stream_source_token(local)

    assert token == stream_source_token(same_local)
    assert token != stream_source_token(other_uri)
    assert token != stream_source_token(same_uri_different_type)
    assert remote.uri not in stream_source_token(remote)
    assert len(token) == 64
    assert token == token.lower()
    assert all(item in "0123456789abcdef" for item in token)


def test_cached_jpeg_bytes_are_independent_from_source_frame_mutation() -> None:
    frame = np.full((54, 96, 3), 90, dtype=np.uint8)
    payload = encode_frame_jpeg(frame)
    cache: dict[str, CachedStreamFrame] = {}

    update_stream_frame_cache(cache, _snapshot_for_cache(frame_jpeg=payload))
    frame[:] = 0

    assert isinstance(cache["stream-1"].jpeg, bytes)
    assert cache["stream-1"].jpeg == payload
