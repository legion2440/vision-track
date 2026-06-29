from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from vision_track.configuration import load_config
from vision_track.context import StreamContext
from vision_track.detections import Detections
from vision_track.detector import DetectorBackend, InferenceResult
from vision_track.device import DeviceInfo
from vision_track.engine import ProcessingEngine
from vision_track.lifecycle import StreamState
from vision_track.queues import FramePacket
from vision_track.scheduler import SharedInferenceScheduler
from vision_track.sources import VideoSource


class RecordingTracker:
    def __init__(self) -> None:
        self.calls = 0
        self.trajectories = {}

    def update(self, detections):
        self.calls += 1
        return detections


class RecordingCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0
        self.geometry = None

    def update(self, _detections, _shape) -> None:
        self.calls += 1
        self.in_count += 1


class NoopReader:
    running = False

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True


class BlockingDetector(DetectorBackend):
    name = "blocking"

    def __init__(self) -> None:
        super().__init__(
            "blocking",
            DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
            image_size=64,
        )
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = 0

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[InferenceResult]:
        self.started.set()
        self.release.wait(timeout=2.0)
        self.calls += 1
        results = [_result("blocking", "cpu") for _frame in frames]
        self.finished.set()
        return results


class ImmediateDetector(DetectorBackend):
    name = "immediate"

    def __init__(self) -> None:
        super().__init__(
            "immediate",
            DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
            image_size=64,
        )

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[InferenceResult]:
        return [_result("immediate", "cpu") for _frame in frames]


def _result(backend: str, device: str) -> InferenceResult:
    return InferenceResult(
        detections=Detections([[0, 0, 2, 2]], [0.9], [0]),
        latency_ms=1.0,
        backend=backend,
        device=device,
    )


def _packet(runtime_generation: int, value: int = 1) -> FramePacket:
    return FramePacket(
        frame=np.full((4, 4, 3), value, dtype=np.uint8),
        frame_index=value,
        captured_at=time.perf_counter(),
        runtime_generation=runtime_generation,
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _engine(tmp_path: Path, detector: DetectorBackend) -> ProcessingEngine:
    config = replace(load_config(), log_file=str(tmp_path / "app_errors.log"))
    return ProcessingEngine(
        config,
        device=DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
        detector=detector,
    )


def _prepare_context(engine: ProcessingEngine, stream_id: str) -> StreamContext:
    context = engine.get(stream_id)
    context.force_state(StreamState.ACTIVE)
    context.tracker = RecordingTracker()
    context.counter = RecordingCounter()
    context.queue.put(_packet(context.runtime_generation))
    return context


def _start_blocked_scheduler(
    engine: ProcessingEngine,
    detector: BlockingDetector,
) -> None:
    engine.scheduler.start()
    assert _wait_until(detector.started.is_set)


def _finish_blocked_scheduler(
    engine: ProcessingEngine,
    detector: BlockingDetector,
) -> None:
    detector.release.set()
    assert _wait_until(detector.finished.is_set)
    time.sleep(0.05)
    engine.scheduler.stop()


def test_old_inference_after_stop_cannot_update_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    detector = BlockingDetector()
    engine = _engine(tmp_path, detector)
    stream_id = engine.add_stream("memory-stop.mp4")
    context = _prepare_context(engine, stream_id)
    tracker = context.tracker
    counter = context.counter
    try:
        _start_blocked_scheduler(engine, detector)
        engine.stop(stream_id)
        _finish_blocked_scheduler(engine, detector)

        assert context.latest_frame is None
        assert context.metrics.processed_frames == 0
        assert tracker.calls == 0
        assert counter.calls == 0
        assert counter.in_count == 0
    finally:
        engine.shutdown()


def test_old_inference_after_restart_cannot_update_new_replay_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    detector = BlockingDetector()
    engine = _engine(tmp_path, detector)
    monkeypatch.setattr(engine, "_build_reader", lambda _context: NoopReader())
    stream_id = engine.add_stream("memory-restart.mp4")
    context = _prepare_context(engine, stream_id)
    old_tracker = context.tracker
    old_counter = context.counter
    try:
        _start_blocked_scheduler(engine, detector)
        engine.restart(stream_id)
        _finish_blocked_scheduler(engine, detector)

        assert context.latest_frame is None
        assert context.metrics.processed_frames == 0
        assert context.tracker is not old_tracker
        assert context.counter is not old_counter
        assert old_tracker.calls == 0
        assert old_counter.calls == 0
    finally:
        engine.shutdown()


def test_old_inference_after_replace_source_cannot_update_replacement_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    detector = BlockingDetector()
    engine = _engine(tmp_path, detector)
    stream_id = engine.add_stream("memory-original.mp4")
    context = _prepare_context(engine, stream_id)
    try:
        _start_blocked_scheduler(engine, detector)
        engine.replace_source(stream_id, "memory-replacement.mp4")
        _finish_blocked_scheduler(engine, detector)

        assert context.source.uri == "memory-replacement.mp4"
        assert context.latest_frame is None
        assert context.metrics.processed_frames == 0
    finally:
        engine.shutdown()


def test_old_inference_after_remove_causes_no_mutation_or_exception(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    detector = BlockingDetector()
    engine = _engine(tmp_path, detector)
    stream_id = engine.add_stream("memory-remove.mp4")
    context = _prepare_context(engine, stream_id)
    try:
        _start_blocked_scheduler(engine, detector)
        engine.remove(stream_id)
        _finish_blocked_scheduler(engine, detector)

        assert stream_id not in [item.stream_id for item in engine.contexts()]
        assert context.latest_frame is None
        assert context.metrics.processed_frames == 0
    finally:
        engine.shutdown()


def test_current_generation_packet_finalizes_normally(monkeypatch) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    context = StreamContext(
        stream_id="current",
        source=VideoSource.from_uri("current.mp4"),
        state=StreamState.ACTIVE,
    )
    context.tracker = RecordingTracker()
    context.counter = RecordingCounter()
    context.queue.put(_packet(context.runtime_generation))
    scheduler = SharedInferenceScheduler(
        ImmediateDetector(),
        lambda: [context],
        logging.getLogger("test-current-generation"),
        idle_seconds=0.001,
        max_batch_size=1,
        max_batch_wait_ms=0,
    )
    try:
        scheduler.start()
        assert _wait_until(lambda: context.metrics.processed_frames == 1)

        assert context.latest_frame is not None
        assert context.actual_backend == "immediate"
        assert context.tracker.calls == 1
        assert context.counter.calls == 1
    finally:
        scheduler.stop()


def test_replay_resets_runtime_and_accepts_only_new_generation_packets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_a, **_k: frame)
    engine = _engine(tmp_path, ImmediateDetector())
    monkeypatch.setattr(engine, "_build_reader", lambda _context: NoopReader())
    stream_id = engine.add_stream("memory-replay.mp4")
    context = engine.get(stream_id)
    old_generation = context.runtime_generation
    context.force_state(StreamState.EOF)
    context.queue.put(_packet(old_generation, value=1))
    context.metrics.processed_frames = 12
    context.latest_frame = np.ones((4, 4, 3), dtype=np.uint8)
    context.latest_rendered_frame = np.ones((4, 4, 3), dtype=np.uint8)
    context.latest_detections = object()
    context.actual_backend = "old"
    context.actual_device = "old-device"
    context.actual_provider = "old-provider"
    context.counter.in_count = 3
    try:
        engine.start(stream_id)
        context.force_state(StreamState.ACTIVE)
        context.tracker = RecordingTracker()
        context.counter = RecordingCounter()

        assert context.runtime_generation != old_generation
        assert context.latest_frame is None
        assert context.metrics.processed_frames == 0
        assert context.actual_backend is None
        assert context.counter.in_count == 0

        context.queue.put(_packet(old_generation, value=2))
        assert _wait_until(context.queue.empty)
        assert context.metrics.processed_frames == 0

        context.queue.put(_packet(context.runtime_generation, value=3))
        assert _wait_until(lambda: context.metrics.processed_frames == 1)
        assert context.latest_frame is not None
        assert int(context.latest_frame[0, 0, 0]) == 3
    finally:
        engine.shutdown()
