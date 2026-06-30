from __future__ import annotations

from types import SimpleNamespace
import threading

import cv2
import numpy as np
import pytest

from vision_track.context import StreamContext
from vision_track.lifecycle import StreamState
from vision_track.sources import SourceType
from vision_track.sources import VideoSource
from vision_track.ui import (
    UI_CANVAS_HEIGHT,
    UI_CANVAS_WIDTH,
    StreamUISnapshot,
    encode_frame_jpeg,
    fit_frame_to_canvas,
    replay_button_label,
    runtime_backend_summary,
    snapshot_stream_context,
    single_stream_column_weights,
    stream_grid_columns,
)


def _nonzero_bbox(frame: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(frame != 0, axis=2)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def test_stream_grid_uses_one_bounded_column_for_single_stream() -> None:
    assert stream_grid_columns(1) == 1
    assert single_stream_column_weights(1) == [1.0, 1.6, 1.0]


def test_stream_grid_uses_two_columns_for_multiple_streams() -> None:
    assert stream_grid_columns(2) == 2
    assert stream_grid_columns(5) == 2


def test_runtime_backend_summary_is_pending_before_first_inference() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    summary = runtime_backend_summary(
        context,
        requested_backend="pytorch",
        requested_device="cuda",
    )
    assert "Requested backend `pytorch`" in summary
    assert "pending first inference" in summary


def test_runtime_backend_summary_reports_actual_provider() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.actual_backend = "onnxruntime"
    context.actual_device = "cpu"
    context.actual_provider = "CPUExecutionProvider"
    summary = runtime_backend_summary(
        context,
        requested_backend="onnxruntime",
        requested_device="cuda",
    )
    assert "Actual backend `onnxruntime`" in summary
    assert "provider `CPUExecutionProvider`" in summary


def test_runtime_backend_summary_reports_snapshot_actual_provider() -> None:
    snapshot = StreamUISnapshot(
        stream_id="stream-1",
        display_name="video.mp4",
        source_type=SourceType.LOCAL,
        state=StreamState.ACTIVE,
        error=None,
        frame_version=3,
        frame_jpeg=None,
        fps=30.0,
        inference_latency_ms=4.0,
        end_to_end_latency_ms=8.0,
        dropped_rate=0.0,
        in_count=1,
        out_count=2,
        occupancy=3,
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
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.force_state(StreamState.EOF)
    context.metrics.processed_frames = 1
    assert replay_button_label(context) == "Replay"


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


def test_encode_frame_jpeg_returns_decodable_bytes() -> None:
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)

    payload = encode_frame_jpeg(frame)
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert isinstance(payload, bytes)
    assert payload
    assert decoded is not None
    assert decoded.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)


def test_snapshot_without_frame_copies_scalar_fields() -> None:
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

    snapshot = snapshot_stream_context(context)

    assert snapshot.frame_jpeg is None
    assert snapshot.frame_version == 12
    assert snapshot.stream_id == "stream-1"
    assert snapshot.display_name == "video.mp4"
    assert snapshot.source_type is SourceType.LOCAL
    assert snapshot.state is StreamState.ACTIVE
    assert snapshot.error == "temporary"
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


def test_snapshot_is_independent_from_context_mutation() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    frame = np.full((54, 96, 3), (30, 80, 130), dtype=np.uint8)
    context.latest_rendered_frame = frame
    context.force_state(StreamState.ACTIVE)
    context.metrics.processed_frames = 5
    context.metrics.fps = 18.0
    context.queue.received = 8
    context.queue.dropped = 1
    context.counter = SimpleNamespace(in_count=2, out_count=3, occupancy=4)
    context.actual_backend = "onnxruntime"
    context.actual_device = "cpu"
    context.actual_provider = "CPUExecutionProvider"

    snapshot = snapshot_stream_context(context)
    frame[:] = 0
    context.metrics.processed_frames = 99
    context.metrics.fps = 1.0
    context.queue.received = 100
    context.queue.dropped = 100
    context.counter.in_count = 20
    context.counter.out_count = 30
    context.counter.occupancy = 40
    context.force_state(StreamState.STOPPED)
    context.actual_backend = "mutated"
    context.actual_device = "mutated"
    context.actual_provider = "mutated"

    decoded = cv2.imdecode(
        np.frombuffer(snapshot.frame_jpeg or b"", dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    assert snapshot.frame_version == 5
    assert snapshot.fps == 18.0
    assert snapshot.dropped_rate == 0.125
    assert snapshot.in_count == 2
    assert snapshot.out_count == 3
    assert snapshot.occupancy == 4
    assert snapshot.state is StreamState.ACTIVE
    assert snapshot.actual_backend == "onnxruntime"
    assert snapshot.actual_device == "cpu"
    assert snapshot.actual_provider == "CPUExecutionProvider"
    assert decoded is not None
    assert decoded.shape == (UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3)
    assert float(decoded[..., 2].mean()) > 100.0


def test_snapshot_uses_context_lock_for_consistent_state() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.force_state(StreamState.CONNECTING)
    context.metrics.processed_frames = 1
    context.counter = SimpleNamespace(in_count=1, out_count=1, occupancy=1)
    context.latest_rendered_frame = np.full((54, 96, 3), 10, dtype=np.uint8)
    started = threading.Event()
    done = threading.Event()
    holder: dict[str, StreamUISnapshot] = {}

    def worker() -> None:
        started.set()
        holder["snapshot"] = snapshot_stream_context(context)
        done.set()

    with context.lock:
        thread = threading.Thread(target=worker)
        thread.start()
        assert started.wait(1.0)
        assert not done.wait(0.05)
        context.force_state(StreamState.ACTIVE)
        context.metrics.processed_frames = 9
        context.metrics.fps = 27.0
        context.metrics.inference_latency_ms = 4.0
        context.metrics.end_to_end_latency_ms = 6.0
        context.queue.received = 20
        context.queue.dropped = 5
        context.counter.in_count = 7
        context.counter.out_count = 8
        context.counter.occupancy = 9
        context.latest_rendered_frame = np.full((54, 96, 3), 140, dtype=np.uint8)
        context.actual_backend = "pytorch"
        context.actual_device = "cuda"
        context.actual_provider = None

    assert done.wait(1.0)
    thread.join(timeout=1.0)
    snapshot = holder["snapshot"]
    assert snapshot.state is StreamState.ACTIVE
    assert snapshot.frame_version == 9
    assert snapshot.fps == 27.0
    assert snapshot.inference_latency_ms == 4.0
    assert snapshot.end_to_end_latency_ms == 6.0
    assert snapshot.dropped_rate == 0.25
    assert snapshot.in_count == 7
    assert snapshot.out_count == 8
    assert snapshot.occupancy == 9
    assert snapshot.actual_backend == "pytorch"
    assert snapshot.actual_device == "cuda"


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
