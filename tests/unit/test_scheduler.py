from __future__ import annotations

import logging
import queue

import numpy as np

from vision_track.context import StreamContext
from vision_track.detections import Detections
from vision_track.detector import InferenceResult
from vision_track.lifecycle import StreamState
from vision_track.queues import FramePacket
from vision_track.scheduler import SharedInferenceScheduler
from vision_track.sources import VideoSource


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStopEvent:
    def __init__(self, clock: FakeClock, on_wait=None) -> None:
        self.clock = clock
        self.on_wait = on_wait
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.clock.advance(timeout)
        if self.on_wait is not None:
            self.on_wait()
        return False


class ScriptedQueue:
    def __init__(self, packets=None) -> None:
        self.packets = list(packets or [])

    def put(self, packet) -> None:
        self.packets.append(packet)

    def get_nowait(self):
        if not self.packets:
            raise queue.Empty
        return self.packets.pop(0)


def _packet(index: int) -> FramePacket:
    return FramePacket(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_index=index,
        captured_at=0.0,
    )


def _context(stream_id: str, scripted_queue: ScriptedQueue) -> StreamContext:
    context = StreamContext(
        stream_id=stream_id,
        source=VideoSource.from_uri(f"{stream_id}.mp4"),
        state=StreamState.ACTIVE,
    )
    context.queue = scripted_queue
    return context


def _scheduler(contexts, clock: FakeClock, stop_event: FakeStopEvent, monkeypatch):
    import vision_track.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module.time, "perf_counter", clock)
    scheduler = SharedInferenceScheduler(
        detector=object(),
        contexts_provider=lambda: contexts,
        logger=logging.getLogger("test"),
        idle_seconds=0.001,
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    scheduler._stop_event = stop_event
    return scheduler


def test_take_batch_waits_for_second_stream_within_batch_window(monkeypatch) -> None:
    clock = FakeClock()
    first_queue = ScriptedQueue([_packet(1)])
    second_queue = ScriptedQueue()
    contexts = [_context("first", first_queue), _context("second", second_queue)]

    def add_second_once() -> None:
        if not second_queue.packets:
            second_queue.put(_packet(2))

    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock, add_second_once), monkeypatch)

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first", "second"]
    assert scheduler._stop_event.waits


def test_take_batch_uses_at_most_one_frame_per_stream(monkeypatch) -> None:
    clock = FakeClock()
    first_queue = ScriptedQueue([_packet(1), _packet(2)])
    contexts = [_context("first", first_queue)]
    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock), monkeypatch)

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first"]
    assert len(first_queue.packets) == 1


def test_take_batch_returns_when_deadline_expires(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [_context("first", ScriptedQueue([_packet(1)]))]
    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock), monkeypatch)

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first"]
    assert clock.now >= 0.01
    assert scheduler._stop_event.waits


def test_finalize_persists_actual_backend_device_and_provider(monkeypatch) -> None:
    clock = FakeClock()
    context = _context("first", ScriptedQueue())
    scheduler = _scheduler([context], clock, FakeStopEvent(clock), monkeypatch)
    monkeypatch.setattr("vision_track.scheduler.render_frame", lambda frame, *_args, **_kwargs: frame)
    packet = _packet(1)
    result = InferenceResult(
        detections=Detections.empty(),
        latency_ms=2.5,
        backend="onnxruntime",
        device="cpu",
        provider="CPUExecutionProvider",
    )

    scheduler._finalize(context, packet, result)

    assert context.actual_backend == "onnxruntime"
    assert context.actual_device == "cpu"
    assert context.actual_provider == "CPUExecutionProvider"
