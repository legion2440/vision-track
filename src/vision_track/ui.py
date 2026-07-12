from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from .context import StreamContext
from .lifecycle import StreamState
from .sources import SourceType, VideoSource
from .tracking import ByteTrackSettings

UI_CANVAS_WIDTH = 960
UI_CANVAS_HEIGHT = 540
UI_JPEG_QUALITY = 85

@dataclass(frozen=True)
class StreamMetricsSnapshot:
    stream_id: str
    display_name: str
    source_type: SourceType
    source_token: str
    state: StreamState
    error: str | None
    processed_frames: int
    fps: float
    inference_latency_ms: float
    end_to_end_latency_ms: float
    dropped_rate: float
    in_count: int
    out_count: int
    occupancy: int
    actual_backend: str | None
    actual_device: str | None
    actual_provider: str | None


@dataclass(frozen=True)
class StreamIdentitySnapshot:
    stream_id: str
    source: VideoSource
    source_token: str
    display_name: str


@dataclass(frozen=True)
class StreamControlSnapshot:
    stream_id: str
    source_type: SourceType
    state: StreamState
    processed_frames: int

    confidence: float
    iou: float
    detection_enabled: bool
    tracking_enabled: bool
    counting_enabled: bool

    track_activation_threshold: float
    lost_track_buffer: int
    minimum_matching_threshold: float

    actual_backend: str | None
    actual_device: str | None
    actual_provider: str | None


def stream_source_token(source: VideoSource) -> str:
    payload = f"{source.source_type.value}\0{source.uri}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stream_grid_columns(stream_count: int) -> int:
    return 1 if stream_count <= 1 else 2


def single_stream_column_weights(stream_count: int) -> list[float]:
    return [1.0, 1.6, 1.0] if stream_count == 1 else [1.0] * stream_grid_columns(stream_count)


def snapshot_stream_identity(
    context: StreamContext,
) -> StreamIdentitySnapshot:
    with context.lock:
        source = context.source
        return StreamIdentitySnapshot(
            stream_id=context.stream_id,
            source=source,
            source_token=stream_source_token(source),
            display_name=source.display_name,
        )


def snapshot_stream_controls(
    context: StreamContext,
) -> StreamControlSnapshot:
    with context.lock:
        source = context.source
        options = context.options
        metrics = context.metrics
        tracker_settings = (
            context.tracker.settings
            if context.tracker is not None
            else ByteTrackSettings()
        )
        return StreamControlSnapshot(
            stream_id=context.stream_id,
            source_type=source.source_type,
            state=context.state,
            processed_frames=metrics.processed_frames,
            confidence=options.confidence,
            iou=options.iou,
            detection_enabled=options.detection_enabled,
            tracking_enabled=options.tracking_enabled,
            counting_enabled=options.counting_enabled,
            track_activation_threshold=tracker_settings.track_activation_threshold,
            lost_track_buffer=tracker_settings.lost_track_buffer,
            minimum_matching_threshold=tracker_settings.minimum_matching_threshold,
            actual_backend=context.actual_backend,
            actual_device=context.actual_device,
            actual_provider=context.actual_provider,
        )


def replay_button_label(control: StreamControlSnapshot) -> str:
    if (
        control.source_type is SourceType.LOCAL
        and control.state in {StreamState.EOF, StreamState.FAILED, StreamState.STOPPED}
        and control.processed_frames > 0
    ):
        return "Replay"
    return "Restart"


def fit_frame_to_canvas(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise ValueError("Frame must be a numpy array")
    if frame.dtype != np.uint8:
        raise ValueError("Frame must use uint8 dtype")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frame must have shape HxWx3")
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Frame dimensions must be positive")

    scale = min(UI_CANVAS_WIDTH / source_width, UI_CANVAS_HEIGHT / source_height)
    resized_width = min(UI_CANVAS_WIDTH, max(1, int(round(source_width * scale))))
    resized_height = min(UI_CANVAS_HEIGHT, max(1, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros((UI_CANVAS_HEIGHT, UI_CANVAS_WIDTH, 3), dtype=np.uint8)
    x_offset = (UI_CANVAS_WIDTH - resized_width) // 2
    y_offset = (UI_CANVAS_HEIGHT - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized
    return np.ascontiguousarray(canvas)


def encode_frame_jpeg(frame: np.ndarray) -> bytes:
    canvas = fit_frame_to_canvas(frame)
    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [int(cv2.IMWRITE_JPEG_QUALITY), UI_JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return encoded.tobytes()


def snapshot_stream_metrics(
    context: StreamContext,
) -> StreamMetricsSnapshot:
    with context.lock:
        source = context.source
        metrics = context.metrics
        counter = context.counter
        received, dropped = context.queue.snapshot_stats()
        dropped_rate = dropped / received if received else 0.0
        in_count = counter.in_count if counter is not None else 0
        out_count = counter.out_count if counter is not None else 0
        occupancy = counter.occupancy if counter is not None else 0
        return StreamMetricsSnapshot(
            stream_id=context.stream_id,
            display_name=source.display_name,
            source_type=source.source_type,
            source_token=stream_source_token(source),
            state=context.state,
            error=context.error,
            processed_frames=metrics.processed_frames,
            fps=metrics.fps,
            inference_latency_ms=metrics.inference_latency_ms,
            end_to_end_latency_ms=metrics.end_to_end_latency_ms,
            dropped_rate=dropped_rate,
            in_count=in_count,
            out_count=out_count,
            occupancy=occupancy,
            actual_backend=context.actual_backend,
            actual_device=context.actual_device,
            actual_provider=context.actual_provider,
        )


def runtime_backend_summary(
    context: StreamControlSnapshot | StreamMetricsSnapshot | None,
    *,
    requested_backend: str,
    requested_device: str,
) -> str:
    if context is None or context.actual_backend is None or context.actual_device is None:
        return (
            f"Requested backend `{requested_backend}` · requested device `{requested_device}` · "
            "actual runtime pending first inference"
        )
    provider = f" · provider `{context.actual_provider}`" if context.actual_provider else ""
    return (
        f"Actual backend `{context.actual_backend}` · actual device `{context.actual_device}`"
        f"{provider}"
    )
