from __future__ import annotations

import atexit
import threading
from dataclasses import replace
from pathlib import Path

from .configuration import AppConfig, load_config, resolve_project_path
from .context import StreamContext, StreamOptions
from .counting import ZoneCounter, ZoneGeometry
from .detector import DetectorBackend, create_backend
from .device import DeviceInfo, select_device
from .lifecycle import StreamState
from .logging_utils import configure_logging
from .readers import VideoReader
from .scheduler import SharedInferenceScheduler
from .sources import VideoSource, new_stream_id
from .tracking import ByteTrackSettings, StreamTracker


class ProcessingEngine:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        backend_name: str = "pytorch",
        device: DeviceInfo | None = None,
        detector: DetectorBackend | None = None,
    ) -> None:
        self.config = config or load_config()
        self.device = device or select_device()
        self.logger = configure_logging(resolve_project_path(self.config.log_file))
        self._contexts: dict[str, StreamContext] = {}
        self._lock = threading.RLock()
        model_path = self._select_model_path(backend_name)
        self.detector = detector or create_backend(
            backend_name,
            model_path,
            self.device,
            image_size=self.config.model.image_size,
            confidence=self.config.model.confidence,
            iou=self.config.model.iou,
            person_class_id=self.config.model.person_class_id,
        )
        self.scheduler = SharedInferenceScheduler(
            self.detector,
            self.contexts,
            self.logger,
            idle_seconds=self.config.runtime.scheduler_idle_seconds,
            max_batch_size=self.config.runtime.max_batch_size,
            max_batch_wait_ms=self.config.runtime.max_batch_wait_ms,
        )
        self._shutdown = False
        atexit.register(self.shutdown)

    def _select_model_path(self, backend_name: str) -> str:
        if backend_name == "onnxruntime":
            return str(resolve_project_path(self.config.model.quantized_checkpoint))
        checkpoint = resolve_project_path(self.config.model.checkpoint)
        return str(checkpoint) if checkpoint.exists() else self.config.model.pretrained

    def contexts(self) -> list[StreamContext]:
        with self._lock:
            return list(self._contexts.values())

    def get(self, stream_id: str) -> StreamContext:
        with self._lock:
            if stream_id not in self._contexts:
                raise KeyError(f"Unknown stream: {stream_id}")
            return self._contexts[stream_id]

    def _tracker_settings(self, frame_rate: float | None = None) -> ByteTrackSettings:
        cfg = self.config.tracking
        return ByteTrackSettings(
            track_activation_threshold=cfg.track_activation_threshold,
            lost_track_buffer=cfg.lost_track_buffer,
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            minimum_consecutive_frames=cfg.minimum_consecutive_frames,
            frame_rate=frame_rate or cfg.frame_rate,
            trajectory_length=cfg.trajectory_length,
        )

    def _zone_geometry(self) -> ZoneGeometry:
        cfg = self.config.counting
        return ZoneGeometry(cfg.line_start, cfg.line_end, cfg.polygon)

    def add_stream(
        self,
        source: VideoSource | str,
        *,
        stream_id: str | None = None,
        options: StreamOptions | None = None,
    ) -> str:
        source_obj = source if isinstance(source, VideoSource) else VideoSource.from_uri(source)
        identifier = stream_id or new_stream_id()
        with self._lock:
            if identifier in self._contexts:
                raise ValueError(f"Stream ID already exists: {identifier}")
            context = StreamContext(
                stream_id=identifier,
                source=source_obj,
                options=options
                or StreamOptions(
                    confidence=self.config.model.confidence,
                    iou=self.config.model.iou,
                ),
            )
            context.tracker = StreamTracker(self._tracker_settings())
            context.counter = ZoneCounter(self._zone_geometry())
            self._contexts[identifier] = context
        return identifier

    def _build_reader(self, context: StreamContext) -> VideoReader:
        return VideoReader(
            stream_id=context.stream_id,
            source=context.source,
            frame_queue=context.queue,
            state_callback=context.force_state,
            error_callback=context.set_error,
            logger=self.logger,
            reconnect_attempts=self.config.runtime.reconnect_attempts,
            reconnect_backoff_seconds=self.config.runtime.reconnect_backoff_seconds,
        )

    def start(self, stream_id: str) -> None:
        context = self.get(stream_id)
        with context.lock:
            if context.reader and context.reader.running:
                return
            context.reader = self._build_reader(context)
            context.error = None
        self.scheduler.start()
        context.reader.start()

    def stop(self, stream_id: str) -> None:
        context = self.get(stream_id)
        reader = context.reader
        if reader:
            reader.stop()
        context.force_state(StreamState.STOPPED)

    def restart(self, stream_id: str) -> None:
        self.stop(stream_id)
        context = self.get(stream_id)
        context.queue.clear()
        context.tracker.reset()
        context.counter.reset()
        context.metrics = type(context.metrics)()
        context.latest_frame = None
        context.latest_rendered_frame = None
        context.latest_detections = None
        self.start(stream_id)

    def remove(self, stream_id: str) -> None:
        context = self.get(stream_id)
        self.stop(stream_id)
        with self._lock:
            self._contexts.pop(context.stream_id, None)

    def replace_source(self, stream_id: str, source: VideoSource | str) -> None:
        self.stop(stream_id)
        context = self.get(stream_id)
        source_obj = source if isinstance(source, VideoSource) else VideoSource.from_uri(source)
        with context.lock:
            context.source = source_obj
            context.reader = None
            context.queue.clear()
            context.tracker.reset()
            context.counter.reset()
            context.metrics = type(context.metrics)()
            context.latest_frame = None
            context.latest_rendered_frame = None
            context.latest_detections = None
            context.error = None
            context.force_state(StreamState.CREATED)

    def reset_counters(self, stream_id: str) -> None:
        self.get(stream_id).counter.reset()

    def update_options(self, stream_id: str, **changes) -> None:
        context = self.get(stream_id)
        with context.lock:
            context.options = replace(context.options, **changes)

    def update_tracker(self, stream_id: str, **changes) -> None:
        context = self.get(stream_id)
        settings = replace(context.tracker.settings, **changes)
        with context.lock:
            context.tracker = StreamTracker(settings)
            context.counter.reset_tracking_state()

    def start_all(self) -> None:
        for context in self.contexts():
            self.start(context.stream_id)

    def stop_all(self) -> None:
        for context in self.contexts():
            self.stop(context.stream_id)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for context in self.contexts():
            reader = context.reader
            if reader:
                reader.stop()
            context.force_state(StreamState.STOPPED)
        self.scheduler.stop()
