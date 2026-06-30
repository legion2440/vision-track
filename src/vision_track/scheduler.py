from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable

from .context import StreamContext
from .detections import Detections
from .detector import DetectorBackend, InferenceResult, nms_detections
from .lifecycle import StreamState
from .logging_utils import log_stream_error
from .queues import FramePacket
from .rendering import render_frame


class SharedInferenceScheduler:
    def __init__(
        self,
        detector: DetectorBackend,
        contexts_provider: Callable[[], list[StreamContext]],
        logger: logging.Logger,
        *,
        idle_seconds: float = 0.005,
        max_batch_size: int = 4,
        max_batch_wait_ms: int = 10,
    ) -> None:
        self.detector = detector
        self.contexts_provider = contexts_provider
        self.logger = logger
        self.idle_seconds = idle_seconds
        self.max_batch_size = max(1, max_batch_size)
        self.max_batch_wait_ms = max(0, max_batch_wait_ms)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._loaded = False
        self._lock = threading.Lock()
        self._cursor = 0

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="vision-shared-inference",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _load_detector(self) -> None:
        if self._loaded:
            return
        self.detector.load()
        self.detector.warmup()
        self._loaded = True

    def _take_batch(self) -> list[tuple[StreamContext, FramePacket]]:
        contexts = self.contexts_provider()
        if not contexts:
            self._cursor = 0
            return []

        self._cursor %= len(contexts)
        batch: list[tuple[StreamContext, FramePacket]] = []
        seen_streams: set[str] = set()
        last_considered: int | None = None
        deadline: float | None = None
        start = self._cursor

        def scan_once() -> bool:
            nonlocal deadline, last_considered
            added = False
            for offset in range(len(contexts)):
                if len(batch) >= self.max_batch_size:
                    break
                index = (start + offset) % len(contexts)
                last_considered = index
                context = contexts[index]
                # A fast local reader can reach EOF before the scheduler consumes
                # its final queued frame. EOF therefore remains drainable.
                with context.lock:
                    stream_id = context.stream_id
                    state = context.state
                if state not in {StreamState.ACTIVE, StreamState.EOF}:
                    continue
                if stream_id in seen_streams:
                    continue
                try:
                    packet = context.queue.get_nowait()
                    with context.lock:
                        if (
                            context.state not in {StreamState.ACTIVE, StreamState.EOF}
                            or packet.runtime_generation != context.runtime_generation
                        ):
                            continue
                    batch.append((context, packet))
                    seen_streams.add(stream_id)
                    added = True
                    if deadline is None:
                        deadline = time.perf_counter() + self.max_batch_wait_ms / 1000.0
                except queue.Empty:
                    continue
            return added

        scan_once()
        if not batch or len(batch) >= self.max_batch_size or self._stop_event.is_set():
            if last_considered is not None:
                self._cursor = (last_considered + 1) % len(contexts)
            return batch

        while len(batch) < self.max_batch_size and not self._stop_event.is_set():
            if deadline is None:
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if self._stop_event.wait(min(self.idle_seconds, remaining)):
                break
            scan_once()
            if len(batch) >= self.max_batch_size:
                break
        if last_considered is not None:
            self._cursor = (last_considered + 1) % len(contexts)
        return batch

    @staticmethod
    def _packet_is_current(context: StreamContext, packet: FramePacket) -> bool:
        with context.lock:
            return packet.runtime_generation == context.runtime_generation

    def _finalize(
        self,
        context: StreamContext,
        packet: FramePacket,
        result: InferenceResult,
    ) -> bool:
        with context.lock:
            if packet.runtime_generation != context.runtime_generation:
                return False
            detections = result.detections
            if not context.options.detection_enabled:
                detections = Detections.empty()
            else:
                detections = detections.filter(confidence=context.options.confidence)
                detections = nms_detections(detections, context.options.iou)

            if context.options.tracking_enabled and context.tracker is not None:
                detections = context.tracker.update(detections)
            if (
                context.options.counting_enabled
                and context.options.tracking_enabled
                and context.counter is not None
            ):
                context.counter.update(detections, packet.frame.shape[:2])

            in_count = context.counter.in_count if context.counter else 0
            out_count = context.counter.out_count if context.counter else 0
            occupancy = context.counter.occupancy if context.counter else 0
            trajectories = context.tracker.trajectories if context.tracker else {}
            rendered = render_frame(
                packet.frame,
                detections,
                trajectories=trajectories,
                geometry=context.counter.geometry if context.counter else None,
                in_count=in_count,
                out_count=out_count,
                occupancy=occupancy,
                show_detections=context.options.detection_enabled,
                show_tracking=context.options.tracking_enabled,
                show_counting=context.options.counting_enabled,
            )
            context.publish_rendered_frame(
                packet.frame,
                rendered,
                detections,
            )
            context.actual_backend = result.backend
            context.actual_device = result.device
            context.actual_provider = result.provider
            context.metrics.update(result.latency_ms, packet.captured_at)
            context.error = None
            return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._take_batch()
            if not batch:
                self._stop_event.wait(self.idle_seconds)
                continue
            try:
                batch = [
                    (context, packet)
                    for context, packet in batch
                    if self._packet_is_current(context, packet)
                ]
                if not batch:
                    continue
                detection_contexts = [
                    (context, packet)
                    for context, packet in batch
                    if context.options.detection_enabled
                    and self._packet_is_current(context, packet)
                ]
                result_by_stream: dict[str, InferenceResult] = {}
                if detection_contexts:
                    self._load_detector()
                    self.detector.confidence = min(
                        context.options.confidence for context, _ in detection_contexts
                    )
                    self.detector.iou = max(
                        context.options.iou for context, _ in detection_contexts
                    )
                    results = self.detector.infer_batch(
                        [packet.frame for _, packet in detection_contexts]
                    )
                    result_by_stream = {
                        context.stream_id: result
                        for (context, _), result in zip(detection_contexts, results)
                    }
                for context, packet in batch:
                    if not self._packet_is_current(context, packet):
                        continue
                    result = result_by_stream.get(
                        context.stream_id,
                        InferenceResult(
                            Detections.empty(),
                            0.0,
                            self.detector.name,
                            self.detector.device.torch_device,
                            getattr(self.detector, "actual_provider", None) or None,
                        ),
                    )
                    try:
                        self._finalize(context, packet, result)
                    except Exception as exc:
                        with context.lock:
                            if packet.runtime_generation != context.runtime_generation:
                                continue
                            context.error = str(exc)
                            state = context.state.value
                            source_type = context.source.source_type.value
                        log_stream_error(
                            self.logger,
                            stream_id=context.stream_id,
                            source_type=source_type,
                            state=state,
                            exc=exc,
                        )
            except Exception as exc:
                for context, packet in batch:
                    with context.lock:
                        if packet.runtime_generation != context.runtime_generation:
                            continue
                        context.error = str(exc)
                        state = context.state.value
                        source_type = context.source.source_type.value
                    log_stream_error(
                        self.logger,
                        stream_id=context.stream_id,
                        source_type=source_type,
                        state=state,
                        exc=exc,
                    )
                self._stop_event.wait(0.25)
