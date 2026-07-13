from __future__ import annotations

import atexit
import threading
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .configuration import AppConfig, load_config, resolve_project_path
from .context import StreamContext, StreamOptions
from .counting import ZoneCounter, ZoneGeometry
from .detector import DetectorBackend, create_backend
from .device import DeviceInfo, select_device
from .lifecycle import StreamState
from .logging_utils import configure_logging, log_stream_error
from .queues import FramePacket
from .readers import VideoReader
from .rendering import render_frame
from .scheduler import SharedInferenceScheduler
from .sources import SourceType, VideoSource, new_stream_id
from .tracking import ByteTrackSettings, StreamTracker


@dataclass(frozen=True)
class StreamRestoreSnapshot:
    stream_id: str
    source: VideoSource
    options: StreamOptions
    tracker_settings: ByteTrackSettings
    was_running: bool


@dataclass(frozen=True)
class _LatestRenderSnapshot:
    runtime_generation: int
    render_state_revision: int
    render_revision: int
    source_frame: Any
    detections: Any
    trajectories: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    geometry: ZoneGeometry | None
    in_count: int
    out_count: int
    occupancy: int
    show_detections: bool
    show_tracking: bool
    show_counting: bool


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

    def _ensure_webcam_available_locked(
        self,
        source: VideoSource,
        *,
        exclude_stream_id: str | None = None,
    ) -> None:
        if source.source_type is not SourceType.WEBCAM:
            return
        for context in self._contexts.values():
            if context.stream_id == exclude_stream_id:
                continue
            with context.lock:
                existing_source = context.source
            if existing_source.uri == source.uri:
                raise ValueError(f"Camera {source.webcam_index} is already added")

    def _is_clean_context(self, context: StreamContext) -> bool:
        return (
            context.metrics.processed_frames == 0
            and context.latest_frame is None
            and context.latest_rendered_frame is None
            and context.latest_rendered_version is None
            and context.latest_detections is None
            and context.error is None
            and context.queue.received == 0
            and context.state in {StreamState.CREATED, StreamState.STOPPED}
        )

    def snapshot_for_rebuild(self) -> list[StreamRestoreSnapshot]:
        snapshots: list[StreamRestoreSnapshot] = []
        for context in self.contexts():
            with context.lock:
                snapshots.append(
                    StreamRestoreSnapshot(
                        stream_id=context.stream_id,
                        source=context.source,
                        options=replace(context.options),
                        tracker_settings=replace(context.tracker.settings),
                        was_running=context.state
                        in {
                            StreamState.PREPARING,
                            StreamState.CONNECTING,
                            StreamState.ACTIVE,
                            StreamState.RECONNECTING,
                        },
                    )
                )
        return snapshots

    def _invalidate_context_runtime(self, context: StreamContext) -> None:
        with context.lock:
            context.runtime_generation += 1

    def _reset_context_runtime(
        self,
        context: StreamContext,
        *,
        source: VideoSource | None = None,
        state: StreamState = StreamState.CREATED,
    ) -> None:
        with context.lock:
            context.runtime_generation += 1
            if source is not None:
                context.source = source
            context.reader = None
            context.queue.clear(reset_stats=True)
            context.tracker = StreamTracker(self._tracker_settings())
            context.counter = ZoneCounter(self._zone_geometry())
            context.metrics = type(context.metrics)()
            context.latest_frame = None
            context.latest_rendered_frame = None
            context.latest_rendered_version = None
            context.latest_detections = None
            context.trajectories.clear()
            context.actual_backend = None
            context.actual_device = None
            context.actual_provider = None
            context.error = None
            context.force_state(state)

    def add_stream(
        self,
        source: VideoSource | str,
        *,
        stream_id: str | None = None,
        options: StreamOptions | None = None,
        tracker_settings: ByteTrackSettings | None = None,
    ) -> str:
        source_obj = source if isinstance(source, VideoSource) else VideoSource.from_uri(source)
        identifier = stream_id or new_stream_id()
        with self._lock:
            if identifier in self._contexts:
                raise ValueError(f"Stream ID already exists: {identifier}")
            self._ensure_webcam_available_locked(source_obj)
            context = StreamContext(
                stream_id=identifier,
                source=source_obj,
                options=options
                or StreamOptions(
                    confidence=self.config.model.confidence,
                    iou=self.config.model.iou,
                ),
            )
            context.tracker = StreamTracker(tracker_settings or self._tracker_settings())
            context.counter = ZoneCounter(self._zone_geometry())
            self._contexts[identifier] = context
        return identifier

    def _build_reader(self, context: StreamContext) -> VideoReader:
        with context.lock:
            captured_generation = context.runtime_generation
            captured_source = context.source
            stream_id = context.stream_id
            wait_for_initial_processing = (
                context.options.detection_enabled and self.scheduler.detector_ready
            )

        def is_current() -> bool:
            with context.lock:
                return context.runtime_generation == captured_generation

        def state_callback(state: StreamState) -> bool:
            with context.lock:
                if context.runtime_generation != captured_generation:
                    return False
                context.force_state(state)
                return True

        def error_callback(error: str | None) -> bool:
            with context.lock:
                if context.runtime_generation != captured_generation:
                    return False
                context.error = error
                return True

        def packet_callback(packet: FramePacket) -> bool:
            with context.lock:
                if context.runtime_generation != captured_generation:
                    return False
                context.queue.put(packet)
                return True

        def failure_log_callback(exc: Exception, state: StreamState) -> bool:
            with context.lock:
                if context.runtime_generation != captured_generation:
                    return False
                log_stream_error(
                    self.logger,
                    stream_id=stream_id,
                    source_type=captured_source.source_type.value,
                    state=state.value,
                    exc=exc,
                    unexpected=False,
                )
                return True

        return VideoReader(
            stream_id=stream_id,
            source=captured_source,
            frame_queue=context.queue,
            state_callback=state_callback,
            error_callback=error_callback,
            logger=self.logger,
            reconnect_attempts=self.config.runtime.reconnect_attempts,
            reconnect_backoff_seconds=self.config.runtime.reconnect_backoff_seconds,
            runtime_generation=captured_generation,
            is_current_callback=is_current,
            packet_callback=packet_callback,
            failure_log_callback=failure_log_callback,
            wait_for_initial_processing=wait_for_initial_processing,
        )

    def _start_reader_if_current(
        self,
        context: StreamContext,
        runtime_generation: int,
    ) -> None:
        with context.lock:
            if self._shutdown or context.runtime_generation != runtime_generation:
                return
            if context.reader and context.reader.running:
                return
            context.reader = self._build_reader(context)
            context.error = None
            self.scheduler.start()
            context.reader.start()

    def _finish_detector_preparation(
        self,
        context: StreamContext,
        runtime_generation: int,
        future: Future[None],
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            with context.lock:
                if (
                    self._shutdown
                    or context.runtime_generation != runtime_generation
                    or context.state is not StreamState.PREPARING
                ):
                    return
                context.error = f"Detector preparation failed: {exc}"
                context.force_state(StreamState.FAILED)
                log_stream_error(
                    self.logger,
                    stream_id=context.stream_id,
                    source_type=context.source.source_type.value,
                    state=StreamState.FAILED.value,
                    exc=exc,
                )
            return
        with context.lock:
            if context.state is not StreamState.PREPARING:
                return
        self._start_reader_if_current(context, runtime_generation)

    def start(self, stream_id: str) -> None:
        context = self.get(stream_id)
        with context.lock:
            if context.reader and context.reader.running:
                return
            if context.state is StreamState.PREPARING:
                return
            if not self._is_clean_context(context):
                self._reset_context_runtime(context)
            runtime_generation = context.runtime_generation
            detection_enabled = context.options.detection_enabled
            context.error = None
            if detection_enabled:
                context.force_state(StreamState.PREPARING)

        if not detection_enabled:
            self._start_reader_if_current(context, runtime_generation)
            return

        preparation = self.scheduler.prepare_detector()
        preparation.add_done_callback(
            lambda future: self._finish_detector_preparation(
                context,
                runtime_generation,
                future,
            )
        )

    def stop(self, stream_id: str) -> None:
        context = self.get(stream_id)
        with context.lock:
            context.runtime_generation += 1
            reader = context.reader
        if reader:
            reader.stop()
        context.force_state(StreamState.STOPPED)

    def restart(self, stream_id: str) -> None:
        self.stop(stream_id)
        context = self.get(stream_id)
        self._reset_context_runtime(context)
        self.start(stream_id)

    def remove(self, stream_id: str) -> None:
        context = self.get(stream_id)
        self.stop(stream_id)
        with self._lock:
            self._contexts.pop(context.stream_id, None)

    def replace_source(self, stream_id: str, source: VideoSource | str) -> None:
        source_obj = source if isinstance(source, VideoSource) else VideoSource.from_uri(source)
        with self._lock:
            if stream_id not in self._contexts:
                raise KeyError(f"Unknown stream: {stream_id}")
            self._ensure_webcam_available_locked(
                source_obj,
                exclude_stream_id=stream_id,
            )
        self.stop(stream_id)
        context = self.get(stream_id)
        with self._lock:
            self._ensure_webcam_available_locked(
                source_obj,
                exclude_stream_id=stream_id,
            )
            self._reset_context_runtime(context, source=source_obj)

    def reset_counters(self, stream_id: str) -> None:
        context = self.get(stream_id)
        with context.lock:
            if context.counter is None:
                return
            context.render_state_revision += 1
            context.counter.reset()
            snapshot = self._snapshot_latest_render_locked(context)
        if snapshot is not None:
            self._rerender_latest(context, snapshot)

    @staticmethod
    def _snapshot_latest_render_locked(
        context: StreamContext,
    ) -> _LatestRenderSnapshot | None:
        if context.latest_frame is None or context.latest_detections is None:
            return None

        counter = context.counter
        tracker = context.tracker
        options = context.options
        trajectories = (
            tuple(
                (tracker_id, tuple(points))
                for tracker_id, points in tracker.trajectories.items()
            )
            if tracker is not None
            else ()
        )
        return _LatestRenderSnapshot(
            runtime_generation=context.runtime_generation,
            render_state_revision=context.render_state_revision,
            render_revision=context.render_revision,
            source_frame=context.latest_frame,
            detections=context.latest_detections,
            trajectories=trajectories,
            geometry=counter.geometry if counter is not None else None,
            in_count=counter.in_count if counter is not None else 0,
            out_count=counter.out_count if counter is not None else 0,
            occupancy=counter.occupancy if counter is not None else 0,
            show_detections=options.detection_enabled,
            show_tracking=options.tracking_enabled,
            show_counting=options.counting_enabled,
        )

    @staticmethod
    def _rerender_latest(
        context: StreamContext,
        snapshot: _LatestRenderSnapshot,
    ) -> bool:
        rendered = render_frame(
            snapshot.source_frame,
            snapshot.detections,
            trajectories=dict(snapshot.trajectories),
            geometry=snapshot.geometry,
            in_count=snapshot.in_count,
            out_count=snapshot.out_count,
            occupancy=snapshot.occupancy,
            show_detections=snapshot.show_detections,
            show_tracking=snapshot.show_tracking,
            show_counting=snapshot.show_counting,
        )
        with context.lock:
            if (
                snapshot.runtime_generation != context.runtime_generation
                or snapshot.render_state_revision != context.render_state_revision
                or snapshot.render_revision != context.render_revision
            ):
                return False
            context.publish_rendered_frame(
                snapshot.source_frame,
                rendered,
                snapshot.detections,
            )
            return True

    def update_options(self, stream_id: str, **changes) -> None:
        context = self.get(stream_id)
        with context.lock:
            updated_options = replace(context.options, **changes)
            if updated_options == context.options:
                return
            context.options = updated_options
            context.render_state_revision += 1

    def update_tracker(self, stream_id: str, **changes) -> None:
        context = self.get(stream_id)
        with context.lock:
            settings = replace(context.tracker.settings, **changes)
            context.tracker = StreamTracker(settings)
            context.counter.reset_tracking_state()
            context.render_state_revision += 1

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
            with context.lock:
                context.runtime_generation += 1
                reader = context.reader
            if reader:
                reader.stop()
            context.force_state(StreamState.STOPPED)
        self.scheduler.stop()
