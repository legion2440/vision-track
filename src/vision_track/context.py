from __future__ import annotations

import threading
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from .lifecycle import StreamState, validate_transition
from .queues import LatestFrameQueue
from .sources import VideoSource


@dataclass
class StreamMetrics:
    processed_frames: int = 0
    inference_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    inference_latency_total_ms: float = 0.0
    end_to_end_latency_total_ms: float = 0.0
    fps: float = 0.0
    last_processed_at: float | None = None

    def update(self, inference_ms: float, captured_at: float) -> None:
        now = perf_counter()
        if self.last_processed_at is not None:
            delta = now - self.last_processed_at
            instantaneous = 1.0 / delta if delta > 0 else 0.0
            self.fps = instantaneous if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneous
        self.last_processed_at = now
        self.processed_frames += 1
        self.inference_latency_total_ms += inference_ms
        self.inference_latency_ms = (
            inference_ms
            if self.processed_frames == 1
            else 0.9 * self.inference_latency_ms + 0.1 * inference_ms
        )
        end_to_end = (now - captured_at) * 1000.0
        self.end_to_end_latency_total_ms += end_to_end
        self.end_to_end_latency_ms = (
            end_to_end
            if self.processed_frames == 1
            else 0.9 * self.end_to_end_latency_ms + 0.1 * end_to_end
        )


@dataclass
class StreamOptions:
    confidence: float = 0.35
    iou: float = 0.50
    detection_enabled: bool = True
    tracking_enabled: bool = True
    counting_enabled: bool = True


@dataclass
class StreamContext:
    stream_id: str
    source: VideoSource
    options: StreamOptions = field(default_factory=StreamOptions)
    state: StreamState = StreamState.CREATED
    error: str | None = None
    queue: LatestFrameQueue = field(default_factory=LatestFrameQueue)
    reader: Any = None
    tracker: Any = None
    counter: Any = None
    latest_frame: np.ndarray | None = None
    latest_rendered_frame: np.ndarray | None = None
    latest_rendered_version: tuple[int, int] | None = None
    latest_detections: Any = None
    actual_backend: str | None = None
    actual_device: str | None = None
    actual_provider: str | None = None
    runtime_generation: int = 0
    render_revision: int = 0
    render_state_revision: int = 0
    trajectories: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def set_state(self, state: StreamState) -> None:
        with self.lock:
            validate_transition(self.state, state)
            self.state = state

    def force_state(self, state: StreamState) -> None:
        with self.lock:
            self.state = state

    def set_error(self, error: str | None) -> None:
        with self.lock:
            self.error = error

    def publish_rendered_frame(
        self,
        source_frame: np.ndarray,
        rendered_frame: np.ndarray,
        detections: Any,
    ) -> tuple[int, int]:
        with self.lock:
            self.render_revision += 1
            version = (self.runtime_generation, self.render_revision)
            self.latest_frame = source_frame
            self.latest_rendered_frame = rendered_frame
            self.latest_rendered_version = version
            self.latest_detections = detections
            return version

    @property
    def composite_ids(self) -> list[tuple[str, int]]:
        detections = self.latest_detections
        if detections is None or detections.tracker_id is None:
            return []
        return [(self.stream_id, int(item)) for item in detections.tracker_id]
