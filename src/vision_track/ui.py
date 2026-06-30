from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .context import StreamContext
from .lifecycle import StreamState
from .sources import SourceType

UI_CANVAS_WIDTH = 960
UI_CANVAS_HEIGHT = 540
UI_JPEG_QUALITY = 85


@dataclass(frozen=True)
class StreamUISnapshot:
    stream_id: str
    display_name: str
    source_type: SourceType
    state: StreamState
    error: str | None
    frame_version: int
    frame_jpeg: bytes | None
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


def stream_grid_columns(stream_count: int) -> int:
    return 1 if stream_count <= 1 else 2


def single_stream_column_weights(stream_count: int) -> list[float]:
    return [1.0, 1.6, 1.0] if stream_count == 1 else [1.0] * stream_grid_columns(stream_count)


def replay_button_label(context: StreamContext) -> str:
    if (
        context.source.source_type is SourceType.LOCAL
        and context.state in {StreamState.EOF, StreamState.FAILED, StreamState.STOPPED}
        and context.metrics.processed_frames > 0
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


def snapshot_stream_context(context: StreamContext) -> StreamUISnapshot:
    with context.lock:
        source = context.source
        metrics = context.metrics
        counter = context.counter
        queue = context.queue
        rendered_frame = context.latest_rendered_frame
        frame_copy = (
            np.ascontiguousarray(rendered_frame).copy()
            if rendered_frame is not None
            else None
        )
        frame_version = metrics.processed_frames
        received = queue.received
        dropped = queue.dropped
        dropped_rate = dropped / received if received else 0.0
        in_count = counter.in_count if counter is not None else 0
        out_count = counter.out_count if counter is not None else 0
        occupancy = counter.occupancy if counter is not None else 0
        stream_id = context.stream_id
        display_name = source.display_name
        source_type = source.source_type
        state = context.state
        error = context.error
        fps = metrics.fps
        inference_latency_ms = metrics.inference_latency_ms
        end_to_end_latency_ms = metrics.end_to_end_latency_ms
        actual_backend = context.actual_backend
        actual_device = context.actual_device
        actual_provider = context.actual_provider

    frame_jpeg = encode_frame_jpeg(frame_copy) if frame_copy is not None else None
    return StreamUISnapshot(
        stream_id=stream_id,
        display_name=display_name,
        source_type=source_type,
        state=state,
        error=error,
        frame_version=frame_version,
        frame_jpeg=frame_jpeg,
        fps=fps,
        inference_latency_ms=inference_latency_ms,
        end_to_end_latency_ms=end_to_end_latency_ms,
        dropped_rate=dropped_rate,
        in_count=in_count,
        out_count=out_count,
        occupancy=occupancy,
        actual_backend=actual_backend,
        actual_device=actual_device,
        actual_provider=actual_provider,
    )


def runtime_backend_summary(
    context: StreamContext | StreamUISnapshot | None,
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
