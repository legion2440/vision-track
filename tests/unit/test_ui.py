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
    StreamControlSnapshot,
    StreamMetricsSnapshot,
    UI_CANVAS_HEIGHT,
    UI_CANVAS_WIDTH,
    encode_frame_jpeg,
    fit_frame_to_canvas,
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    snapshot_stream_controls,
    snapshot_stream_identity,
    snapshot_stream_metrics,
    stream_grid_columns,
    stream_source_token,
)


def _nonzero_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(frame != 0, axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


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


def test_webcam_stream_always_uses_restart_label() -> None:
    control = _control_snapshot(
        source_type=SourceType.WEBCAM,
        state=StreamState.FAILED,
        processed_frames=3,
    )

    assert replay_button_label(control) == "Restart"


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


def test_encode_frame_jpeg_preserves_vertical_aspect_ratio() -> None:
    frame = np.full((1280, 720, 3), 120, dtype=np.uint8)

    payload = encode_frame_jpeg(frame)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert decoded.shape == (UI_CANVAS_HEIGHT, 304, 3)


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
