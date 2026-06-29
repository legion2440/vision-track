from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class FramePacket:
    frame: np.ndarray
    frame_index: int
    captured_at: float
    runtime_generation: int = 0
    source_timestamp_ms: float | None = None

    @classmethod
    def create(
        cls,
        frame: np.ndarray,
        frame_index: int,
        source_timestamp_ms: float | None = None,
        runtime_generation: int = 0,
    ) -> "FramePacket":
        return cls(
            frame=frame,
            frame_index=frame_index,
            captured_at=perf_counter(),
            runtime_generation=runtime_generation,
            source_timestamp_ms=source_timestamp_ms,
        )


class LatestFrameQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self.received = 0
        self.dropped = 0

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self.received += 1
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass
            self._queue.put_nowait(packet)

    def get_nowait(self) -> FramePacket:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def clear(self, *, reset_stats: bool = False) -> None:
        with self._lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            if reset_stats:
                self.received = 0
                self.dropped = 0

    @property
    def dropped_rate(self) -> float:
        return self.dropped / self.received if self.received else 0.0
