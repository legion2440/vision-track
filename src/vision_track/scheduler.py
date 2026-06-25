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
        batch: list[tuple[StreamContext, FramePacket]] = []
        deadline = time.perf_counter() + self.max_batch_wait_ms / 1000.0
        while len(batch) < self.max_batch_size:
            added = False
            for context in self.contexts_provider():
                if len(batch) >= self.max_batch_size:
                    break
                # A fast local reader can reach EOF before the scheduler consumes
                # its final queued frame. EOF therefore remains drainable.
                if context.state not in {StreamState.ACTIVE, StreamState.EOF}:
                    continue
                if any(existing.stream_id == context.stream_id for existing, _ in batch):
                    continue
                try:
                    batch.append((context, context.queue.get_nowait()))
                    added = True
                except queue.Empty:
                    continue
            if batch and (time.perf_counter() >= deadline or not added):
                break
            if not batch and not added:
                break
        return batch

    def _finalize(
        self,
        context: StreamContext,
        packet: FramePacket,
        result: InferenceResult,
    ) -> None:
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
        with context.lock:
            context.latest_frame = packet.frame
            context.latest_rendered_frame = rendered
            context.latest_detections = detections
            context.metrics.update(result.latency_ms, packet.captured_at)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            batch = self._take_batch()
            if not batch:
                self._stop_event.wait(self.idle_seconds)
                continue
            try:
                detection_contexts = [
                    (context, packet)
                    for context, packet in batch
                    if context.options.detection_enabled
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
                    result = result_by_stream.get(
                        context.stream_id,
                        InferenceResult(
                            Detections.empty(),
                            0.0,
                            self.detector.name,
                            self.detector.device.kind,
                        ),
                    )
                    try:
                        self._finalize(context, packet, result)
                        context.set_error(None)
                    except Exception as exc:
                        context.set_error(str(exc))
                        log_stream_error(
                            self.logger,
                            stream_id=context.stream_id,
                            source_type=context.source.source_type.value,
                            state=context.state.value,
                            exc=exc,
                        )
            except Exception as exc:
                for context, _ in batch:
                    context.set_error(str(exc))
                    log_stream_error(
                        self.logger,
                        stream_id=context.stream_id,
                        source_type=context.source.source_type.value,
                        state=context.state.value,
                        exc=exc,
                    )
                self._stop_event.wait(0.25)
