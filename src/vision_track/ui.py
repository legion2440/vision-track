from __future__ import annotations

import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import cv2
import numpy as np

from .context import StreamContext
from .lifecycle import StreamState
from .sources import SourceType, VideoSource

UI_CANVAS_WIDTH = 960
UI_CANVAS_HEIGHT = 540
UI_JPEG_QUALITY = 85

FrameVersion = tuple[int, int]


@dataclass(frozen=True)
class StreamFrameSnapshot:
    stream_id: str
    source_token: str
    frame_version: FrameVersion | None
    frame_jpeg: bytes | None


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
class CachedStreamFrame:
    source_token: str
    frame_version: FrameVersion
    jpeg: bytes


@dataclass(frozen=True)
class StreamFrameUpdate:
    render_jpeg: bytes | None
    clear_image: bool
    show_waiting: bool


def stream_source_token(source: VideoSource) -> str:
    payload = f"{source.source_type.value}\0{source.uri}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def _copy_frame_for_ui(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame).copy()


def update_stream_frame_cache(
    cache: MutableMapping[str, CachedStreamFrame],
    snapshot: StreamFrameSnapshot,
) -> StreamFrameUpdate:
    cached = cache.get(snapshot.stream_id)
    source_changed = False
    if cached is not None and cached.source_token != snapshot.source_token:
        cache.pop(snapshot.stream_id, None)
        cached = None
        source_changed = True

    if snapshot.frame_jpeg is not None:
        if snapshot.frame_version is None:
            raise RuntimeError("Cannot cache a rendered frame without a published version")
        cache[snapshot.stream_id] = CachedStreamFrame(
            source_token=snapshot.source_token,
            frame_version=snapshot.frame_version,
            jpeg=snapshot.frame_jpeg,
        )
        return StreamFrameUpdate(
            render_jpeg=snapshot.frame_jpeg,
            clear_image=False,
            show_waiting=False,
        )

    if cached is not None:
        return StreamFrameUpdate(
            render_jpeg=None,
            clear_image=False,
            show_waiting=False,
        )

    if source_changed:
        return StreamFrameUpdate(
            render_jpeg=None,
            clear_image=True,
            show_waiting=True,
        )

    return StreamFrameUpdate(
        render_jpeg=None,
        clear_image=False,
        show_waiting=True,
    )


def waiting_slot_transition(
    previous_visible: bool,
    desired_visible: bool,
) -> bool | None:
    if previous_visible == desired_visible:
        return None
    return desired_visible


def clear_stream_frame_cache(
    cache: MutableMapping[str, CachedStreamFrame],
    stream_id: str,
) -> None:
    cache.pop(stream_id, None)


def prune_stream_frame_cache(
    cache: MutableMapping[str, CachedStreamFrame],
    active_sources: Mapping[str, str],
) -> None:
    for stream_id, cached in list(cache.items()):
        if stream_id not in active_sources or cached.source_token != active_sources[stream_id]:
            cache.pop(stream_id, None)


def snapshot_stream_frame(
    context: StreamContext,
    *,
    cached_frame: CachedStreamFrame | None = None,
) -> StreamFrameSnapshot:
    with context.lock:
        source = context.source
        source_token = stream_source_token(source)
        frame_version = context.latest_rendered_version
        cached_matches = (
            cached_frame is not None
            and cached_frame.source_token == source_token
            and frame_version is not None
            and cached_frame.frame_version == frame_version
        )
        should_copy_frame = (
            frame_version is not None
            and context.latest_rendered_frame is not None
            and not cached_matches
        )
        frame_copy = (
            _copy_frame_for_ui(context.latest_rendered_frame)
            if should_copy_frame
            else None
        )
        stream_id = context.stream_id

    frame_jpeg = encode_frame_jpeg(frame_copy) if frame_copy is not None else None
    return StreamFrameSnapshot(
        stream_id=stream_id,
        source_token=source_token,
        frame_version=frame_version,
        frame_jpeg=frame_jpeg,
    )


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
    context: StreamContext | StreamMetricsSnapshot | None,
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
