from __future__ import annotations

import logging
import threading
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from .lifecycle import StreamState
from .logging_utils import log_stream_error
from .queues import FramePacket, LatestFrameQueue
from .sources import SourceType, VideoSource


class VideoReader:
    def __init__(
        self,
        *,
        stream_id: str,
        source: VideoSource,
        frame_queue: LatestFrameQueue,
        state_callback: Callable[[StreamState], None],
        error_callback: Callable[[str | None], None],
        logger: logging.Logger,
        reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 1.0,
    ) -> None:
        self.stream_id = stream_id
        self.source = source
        self.frame_queue = frame_queue
        self.state_callback = state_callback
        self.error_callback = error_callback
        self.logger = logger
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.reconnect_backoff_seconds = max(0.0, reconnect_backoff_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
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
                name=f"vision-reader-{self.stream_id}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._capture = None

    def _open(self) -> cv2.VideoCapture:
        if self.source.source_type is SourceType.LOCAL and not Path(self.source.uri).is_file():
            raise FileNotFoundError(f"Video file not found: {self.source.safe_uri}")
        capture = cv2.VideoCapture(self.source.uri)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Unable to open video source: {self.source.safe_uri}")
        return capture

    def _run(self) -> None:
        frame_index = 0
        attempt = 0
        self.state_callback(StreamState.CONNECTING)
        while not self._stop_event.is_set():
            try:
                self._capture = self._open()
                self.error_callback(None)
                self.state_callback(StreamState.ACTIVE)
                attempt = 0
                while not self._stop_event.is_set():
                    ok, frame = self._capture.read()
                    if not ok or frame is None:
                        if self.source.source_type is SourceType.LOCAL:
                            self.state_callback(StreamState.EOF)
                            return
                        raise OSError(
                            f"Decoder/read failure for source: {self.source.safe_uri}"
                        )
                    timestamp = self._capture.get(cv2.CAP_PROP_POS_MSEC)
                    self.frame_queue.put(
                        FramePacket(
                            frame=frame,
                            frame_index=frame_index,
                            captured_at=perf_counter(),
                            source_timestamp_ms=timestamp if timestamp >= 0 else None,
                        )
                    )
                    frame_index += 1
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                attempt += 1
                self.error_callback(str(exc))
                current = (
                    StreamState.RECONNECTING
                    if self.source.is_remote and attempt <= self.reconnect_attempts
                    else StreamState.FAILED
                )
                self.state_callback(current)
                log_stream_error(
                    self.logger,
                    stream_id=self.stream_id,
                    source_type=self.source.source_type.value,
                    state=current.value,
                    exc=exc,
                    unexpected=False,
                )
                if not self.source.is_remote or attempt > self.reconnect_attempts:
                    return
                delay = self.reconnect_backoff_seconds * (2 ** min(attempt - 1, 4))
                if self._stop_event.wait(delay):
                    break
            finally:
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
        self.state_callback(StreamState.STOPPED)

