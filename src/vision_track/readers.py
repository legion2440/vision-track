from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .lifecycle import StreamState
from .logging_utils import log_stream_error
from .queues import FramePacket, LatestFrameQueue
from .sources import SourceType, VideoSource
from .webcams import open_webcam, webcam_backend_preferences


DEFAULT_LOCAL_FPS = 30.0
MAX_REASONABLE_FPS = 240.0
LOCAL_READ_RETRIES = 3
LOCAL_RETRY_WAIT_SECONDS = 0.01
WEBCAM_READ_FAILURE_LIMIT = 3
WEBCAM_STABLE_FRAME_COUNT = 30
WEBCAM_STABLE_SECONDS = 3.0


def _safe_capture_value(capture: cv2.VideoCapture, prop: int) -> float | None:
    try:
        value = float(capture.get(prop))
    except Exception:
        return None
    return value if math.isfinite(value) else None


def _normalized_fps(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return DEFAULT_LOCAL_FPS
    if value <= 0 or value > MAX_REASONABLE_FPS:
        return DEFAULT_LOCAL_FPS
    return float(value)


def _valid_timestamp_ms(value: float | None, previous: float | None) -> float | None:
    if value is None or not math.isfinite(value) or value < 0:
        return None
    if previous is not None and value <= previous:
        return None
    return float(value)


def _local_read_is_eof(capture: cv2.VideoCapture, frames_read: int) -> bool:
    frame_count = _safe_capture_value(capture, cv2.CAP_PROP_FRAME_COUNT)
    if frame_count is not None and frame_count > 0:
        current_position = _safe_capture_value(capture, cv2.CAP_PROP_POS_FRAMES)
        if current_position is not None and current_position >= frame_count:
            return True
        if frames_read >= int(frame_count):
            return True
        return False
    ratio = _safe_capture_value(capture, cv2.CAP_PROP_POS_AVI_RATIO)
    return ratio is not None and ratio >= 0.99


def _webcam_connection_is_stable(
    successful_frames: int,
    stable_since: float | None,
    now: float,
) -> bool:
    return (
        successful_frames >= WEBCAM_STABLE_FRAME_COUNT
        and stable_since is not None
        and now - stable_since >= WEBCAM_STABLE_SECONDS
    )


class VideoReader:
    def __init__(
        self,
        *,
        stream_id: str,
        source: VideoSource,
        frame_queue: LatestFrameQueue,
        state_callback: Callable[[StreamState], bool | None],
        error_callback: Callable[[str | None], bool | None],
        logger: logging.Logger,
        reconnect_attempts: int = 5,
        reconnect_backoff_seconds: float = 1.0,
        runtime_generation: int = 0,
        is_current_callback: Callable[[], bool] | None = None,
        packet_callback: Callable[[FramePacket], bool | None] | None = None,
        failure_log_callback: Callable[[Exception, StreamState], bool | None] | None = None,
        wait_for_initial_processing: bool = False,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.stream_id = stream_id
        self.source = source
        self.frame_queue = frame_queue
        self.state_callback = state_callback
        self.error_callback = error_callback
        self.logger = logger
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.reconnect_backoff_seconds = max(0.0, reconnect_backoff_seconds)
        self.runtime_generation = runtime_generation
        self.is_current_callback = is_current_callback
        self.packet_callback = packet_callback
        self.failure_log_callback = failure_log_callback
        self.wait_for_initial_processing = wait_for_initial_processing
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._prefetched_webcam_frame: tuple[np.ndarray, float] | None = None
        self._webcam_backend_start_index = 0
        self._active_webcam_backend: int | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop_event.clear()
            self._webcam_backend_start_index = 0
            self._active_webcam_backend = None
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
        self._prefetched_webcam_frame = None
        if self.source.source_type is SourceType.WEBCAM:
            backend_preferences = webcam_backend_preferences()
            opened = open_webcam(
                self.source.webcam_index,
                backends=backend_preferences[self._webcam_backend_start_index :],
                clock=self._clock,
                capture_callback=lambda capture: setattr(self, "_capture", capture),
                cancelled=lambda: self._stop_event.is_set() or not self._is_current(),
            )
            self._prefetched_webcam_frame = (
                opened.first_frame,
                opened.captured_at,
            )
            self._active_webcam_backend = opened.backend
            return opened.capture
        if self.source.source_type is SourceType.LOCAL and not Path(self.source.uri).is_file():
            raise FileNotFoundError(f"Video file not found: {self.source.safe_uri}")
        capture = cv2.VideoCapture(self.source.uri)
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Unable to open video source: {self.source.safe_uri}")
        return capture

    def _is_current(self) -> bool:
        if self.is_current_callback is None:
            return True
        return bool(self.is_current_callback())

    @staticmethod
    def _callback_applied(result: bool | None) -> bool:
        return result is not False

    def _emit_state(self, state: StreamState) -> bool:
        return self._callback_applied(self.state_callback(state))

    def _emit_error(self, error: str | None) -> bool:
        return self._callback_applied(self.error_callback(error))

    def _emit_packet(self, packet: FramePacket) -> bool:
        if self.packet_callback is None:
            self.frame_queue.put(packet)
            return True
        return self._callback_applied(self.packet_callback(packet))

    def _emit_failure_log(self, exc: Exception, state: StreamState) -> bool:
        if self.failure_log_callback is None:
            log_stream_error(
                self.logger,
                stream_id=self.stream_id,
                source_type=self.source.source_type.value,
                state=state.value,
                exc=exc,
                unexpected=False,
            )
            return True
        return self._callback_applied(self.failure_log_callback(exc, state))

    def _wait_for_local_presentation(
        self,
        *,
        playback_origin: float,
        target_seconds: float,
    ) -> bool:
        target = playback_origin + target_seconds
        while not self._stop_event.is_set():
            remaining = target - self._clock()
            if remaining <= 0:
                return True
            if self._stop_event.wait(min(remaining, 0.05)):
                return False
        return False

    def _wait_for_processing(self, complete: threading.Event) -> bool:
        while not complete.is_set():
            if not self._is_current():
                return False
            if self._stop_event.wait(0.05):
                return False
        return self._is_current() and not self._stop_event.is_set()

    def _run(self) -> None:
        frame_index = 0
        attempt = 0
        if not self._is_current():
            return
        if not self._emit_state(StreamState.CONNECTING):
            return
        while not self._stop_event.is_set():
            if not self._is_current():
                return
            try:
                self._capture = self._open()
                if not self._is_current():
                    return
                source_fps = _normalized_fps(
                    _safe_capture_value(self._capture, cv2.CAP_PROP_FPS)
                )
                previous_timestamp_ms: float | None = None
                playback_origin: float | None = None
                local_failures = 0
                webcam_failures = 0
                webcam_stable_frames = 0
                webcam_stable_since: float | None = None
                webcam_connection_stable = False
                if not self._is_current():
                    return
                if not self._emit_error(None):
                    return
                if not self._is_current():
                    return
                if not self._emit_state(StreamState.ACTIVE):
                    return
                if self.source.source_type is not SourceType.WEBCAM:
                    attempt = 0
                while not self._stop_event.is_set():
                    if not self._is_current():
                        return
                    captured_at: float | None = None
                    if self._prefetched_webcam_frame is not None:
                        frame, captured_at = self._prefetched_webcam_frame
                        self._prefetched_webcam_frame = None
                        ok = True
                    else:
                        ok, frame = self._capture.read()
                        if self.source.source_type is SourceType.WEBCAM:
                            captured_at = self._clock()
                    if not ok or frame is None:
                        if self._stop_event.is_set():
                            break
                        if self.source.source_type is SourceType.LOCAL:
                            if _local_read_is_eof(self._capture, frame_index):
                                if not self._is_current():
                                    return
                                self._emit_state(StreamState.EOF)
                                return
                            local_failures += 1
                            if local_failures <= LOCAL_READ_RETRIES:
                                if self._stop_event.wait(LOCAL_RETRY_WAIT_SECONDS):
                                    break
                                continue
                            if _local_read_is_eof(self._capture, frame_index):
                                if not self._is_current():
                                    return
                                self._emit_state(StreamState.EOF)
                                return
                            raise OSError(
                                "Decode failure before end of local video: "
                                f"{self.source.safe_uri}"
                            )
                        if self.source.source_type is SourceType.WEBCAM:
                            webcam_failures += 1
                            webcam_stable_frames = 0
                            webcam_stable_since = None
                            if webcam_failures < WEBCAM_READ_FAILURE_LIMIT:
                                if self._stop_event.wait(LOCAL_RETRY_WAIT_SECONDS):
                                    break
                                continue
                            if not webcam_connection_stable:
                                self._advance_webcam_backend()
                        raise OSError(
                            f"Decoder/read failure for source: {self.source.safe_uri}"
                        )
                    local_failures = 0
                    webcam_failures = 0
                    if self.source.source_type is SourceType.WEBCAM:
                        timestamp = None
                        target_seconds = 0.0
                        captured_at = captured_at if captured_at is not None else self._clock()
                        if webcam_stable_since is None:
                            webcam_stable_since = captured_at
                        webcam_stable_frames += 1
                        if _webcam_connection_is_stable(
                            webcam_stable_frames,
                            webcam_stable_since,
                            captured_at,
                        ):
                            attempt = 0
                            self._webcam_backend_start_index = 0
                            webcam_connection_stable = True
                    else:
                        timestamp = _valid_timestamp_ms(
                            _safe_capture_value(self._capture, cv2.CAP_PROP_POS_MSEC),
                            previous_timestamp_ms,
                        )
                        if timestamp is not None:
                            previous_timestamp_ms = timestamp
                        target_seconds = (
                            timestamp / 1000.0
                            if timestamp is not None
                            else frame_index / source_fps
                        )
                    if self.source.source_type is SourceType.LOCAL:
                        if playback_origin is None:
                            if not self.wait_for_initial_processing:
                                playback_origin = self._clock() - target_seconds
                        elif not self._wait_for_local_presentation(
                            playback_origin=playback_origin,
                            target_seconds=target_seconds,
                        ):
                            break
                    if not self._is_current():
                        return
                    packet = FramePacket(
                        frame=frame,
                        frame_index=frame_index,
                        captured_at=(
                            captured_at
                            if captured_at is not None
                            else self._clock()
                        ),
                        runtime_generation=self.runtime_generation,
                        source_timestamp_ms=timestamp,
                        # The first real prediction can still initialize source-shaped
                        # predictor state after the generic detector warmup.
                        processing_complete=(
                            threading.Event()
                            if self.source.source_type is SourceType.LOCAL
                            and self.wait_for_initial_processing
                            and playback_origin is None
                            else None
                        ),
                    )
                    if not self._emit_packet(packet):
                        return
                    if packet.processing_complete is not None:
                        if not self._wait_for_processing(packet.processing_complete):
                            break
                        # Start media time after frame zero has actually been published,
                        # so the latest-only queue cannot enter catch-up mode at startup.
                        playback_origin = self._clock() - target_seconds
                    frame_index += 1
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                if not self._is_current():
                    return
                attempt += 1
                if not self._emit_error(str(exc)):
                    return
                current = (
                    StreamState.RECONNECTING
                    if self.source.is_reconnectable
                    and attempt <= self.reconnect_attempts
                    else StreamState.FAILED
                )
                if not self._is_current():
                    return
                if not self._emit_state(current):
                    return
                if not self._is_current():
                    return
                if not self._emit_failure_log(exc, current):
                    return
                if not self.source.is_reconnectable or attempt > self.reconnect_attempts:
                    return
                if not self._is_current():
                    return
                delay = self.reconnect_backoff_seconds * (2 ** min(attempt - 1, 4))
                if self._stop_event.wait(delay):
                    break
            finally:
                if self._capture is not None:
                    self._capture.release()
                    self._capture = None
        if self._is_current():
            self._emit_state(StreamState.STOPPED)

    def _advance_webcam_backend(self) -> None:
        backend = self._active_webcam_backend
        if backend is None:
            return
        preferences = webcam_backend_preferences()
        try:
            backend_index = preferences.index(backend)
        except ValueError:
            return
        self._webcam_backend_start_index = max(
            self._webcam_backend_start_index,
            backend_index + 1,
        )
