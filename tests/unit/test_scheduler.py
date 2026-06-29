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
    def __init__(self, clock: FakeClock, on_wait=None, *, interrupt_on_wait: bool = False) -> None:
        self.clock = clock
        self.on_wait = on_wait
        self.interrupt_on_wait = interrupt_on_wait
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float) -> bool:
        self.waits.append(timeout)
        self.clock.advance(timeout)
        if self.on_wait is not None:
            self.on_wait()
        return self.interrupt_on_wait


class ScriptedQueue:
    def __init__(self, packets=None) -> None:
        self.packets = list(packets or [])

    def put(self, packet) -> None:
        self.packets.append(packet)

    def get_nowait(self):
        if not self.packets:
            raise queue.Empty
        return self.packets.pop(0)


class AdvancingQueue(ScriptedQueue):
    def __init__(self, clock: FakeClock, advance_seconds: float, packets=None) -> None:
        super().__init__(packets)
        self.clock = clock
        self.advance_seconds = advance_seconds
        self.advanced = False

    def get_nowait(self):
        if not self.advanced:
            self.clock.advance(self.advance_seconds)
            self.advanced = True
        return super().get_nowait()


def _packet(index: int, runtime_generation: int = 0) -> FramePacket:
    return FramePacket(
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        frame_index=index,
        captured_at=0.0,
        runtime_generation=runtime_generation,
    )


def _context(stream_id: str, scripted_queue: ScriptedQueue) -> StreamContext:
    context = StreamContext(
        stream_id=stream_id,
        source=VideoSource.from_uri(f"{stream_id}.mp4"),
        state=StreamState.ACTIVE,
    )
    context.queue = scripted_queue
    return context


def _scheduler(
    contexts,
    clock: FakeClock,
    stop_event: FakeStopEvent,
    monkeypatch,
    *,
    max_batch_wait_ms: int = 10,
):
    import vision_track.scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module.time, "perf_counter", clock)
    scheduler = SharedInferenceScheduler(
        detector=object(),
        contexts_provider=lambda: contexts,
        logger=logging.getLogger("test"),
        idle_seconds=0.001,
        max_batch_size=2,
        max_batch_wait_ms=max_batch_wait_ms,
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


def test_take_batch_no_valid_first_packet_returns_without_batch_wait(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [
        _context("first", ScriptedQueue()),
        _context("second", ScriptedQueue()),
    ]
    stop_event = FakeStopEvent(clock)
    scheduler = _scheduler(contexts, clock, stop_event, monkeypatch)

    batch = scheduler._take_batch()

    assert batch == []
    assert clock.now == 0.0
    assert stop_event.waits == []


def test_take_batch_first_packet_available_after_clock_advance_gets_full_window(
    monkeypatch,
) -> None:
    clock = FakeClock()
    first_queue = AdvancingQueue(clock, 0.009, [_packet(1)])
    second_queue = ScriptedQueue()
    contexts = [_context("first", first_queue), _context("second", second_queue)]

    def add_second_at_18ms() -> None:
        if clock.now >= 0.018 and not second_queue.packets:
            second_queue.put(_packet(2))

    scheduler = _scheduler(
        contexts,
        clock,
        FakeStopEvent(clock, add_second_at_18ms),
        monkeypatch,
    )

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first", "second"]
    assert clock.now <= 0.019


def test_take_batch_stale_packet_does_not_start_batch_deadline(monkeypatch) -> None:
    clock = FakeClock()
    context = _context("stale", ScriptedQueue([_packet(1, runtime_generation=0)]))
    context.runtime_generation = 1
    stop_event = FakeStopEvent(clock)
    scheduler = _scheduler([context], clock, stop_event, monkeypatch)

    batch = scheduler._take_batch()

    assert batch == []
    assert clock.now == 0.0
    assert stop_event.waits == []


def test_take_batch_inactive_context_does_not_start_batch_deadline(monkeypatch) -> None:
    clock = FakeClock()
    context = _context("inactive", ScriptedQueue([_packet(1)]))
    context.force_state(StreamState.STOPPED)
    stop_event = FakeStopEvent(clock)
    scheduler = _scheduler([context], clock, stop_event, monkeypatch)

    batch = scheduler._take_batch()

    assert batch == []
    assert clock.now == 0.0
    assert stop_event.waits == []


def test_take_batch_zero_wait_returns_without_additional_wait(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [
        _context("first", ScriptedQueue([_packet(1)])),
        _context("second", ScriptedQueue([_packet(2)])),
    ]
    stop_event = FakeStopEvent(clock)
    scheduler = _scheduler(
        contexts,
        clock,
        stop_event,
        monkeypatch,
        max_batch_wait_ms=0,
    )

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first", "second"]
    assert stop_event.waits == []


def test_take_batch_stop_during_batch_window_returns_promptly(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [_context("first", ScriptedQueue([_packet(1)]))]
    stop_event = FakeStopEvent(clock, interrupt_on_wait=True)
    scheduler = _scheduler(contexts, clock, stop_event, monkeypatch)

    batch = scheduler._take_batch()

    assert [context.stream_id for context, _ in batch] == ["first"]
    assert len(stop_event.waits) == 1


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


def test_take_batch_round_robins_six_ready_streams_across_three_batches(
    monkeypatch,
) -> None:
    clock = FakeClock()
    contexts = [
        _context(
            f"stream-{index}",
            ScriptedQueue([_packet(index * 10 + offset) for offset in range(3)]),
        )
        for index in range(6)
    ]
    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock), monkeypatch)

    batches = [scheduler._take_batch() for _ in range(3)]
    selected = [
        context.stream_id
        for batch in batches
        for context, _packet_item in batch
    ]

    assert selected == [f"stream-{index}" for index in range(6)]
    for batch in batches:
        stream_ids = [context.stream_id for context, _packet_item in batch]
        assert len(stream_ids) == len(set(stream_ids))


def test_take_batch_cursor_remains_valid_after_removing_stream(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [
        _context(f"stream-{index}", ScriptedQueue([_packet(index)]))
        for index in range(4)
    ]
    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock), monkeypatch)
    scheduler._cursor = 3
    contexts.pop(1)

    scheduler._take_batch()

    assert 0 <= scheduler._cursor < len(contexts)


def test_take_batch_cursor_remains_valid_after_adding_stream(monkeypatch) -> None:
    clock = FakeClock()
    contexts = [
        _context(f"stream-{index}", ScriptedQueue([_packet(index)]))
        for index in range(2)
    ]
    scheduler = _scheduler(contexts, clock, FakeStopEvent(clock), monkeypatch)
    scheduler._take_batch()
    contexts.append(_context("stream-added", ScriptedQueue([_packet(99)])))

    scheduler._take_batch()

    assert 0 <= scheduler._cursor < len(contexts)


def test_take_batch_drains_final_frame_from_eof_stream(monkeypatch) -> None:
    clock = FakeClock()
    context = _context("final", ScriptedQueue([_packet(1)]))
    context.force_state(StreamState.EOF)
    scheduler = _scheduler([context], clock, FakeStopEvent(clock), monkeypatch)

    batch = scheduler._take_batch()

    assert [item[0].stream_id for item in batch] == ["final"]
