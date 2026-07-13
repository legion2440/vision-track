from __future__ import annotations

import sys
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import cv2
import numpy as np


WEBCAM_SCAN_INDICES = range(10)


@dataclass(frozen=True)
class OpenedWebcam:
    capture: Any
    first_frame: np.ndarray
    captured_at: float
    backend: int


def webcam_backend_preferences(platform: str | None = None) -> tuple[int, ...]:
    current_platform = platform or sys.platform
    if current_platform == "win32":
        return cv2.CAP_MSMF, cv2.CAP_DSHOW
    return (cv2.CAP_ANY,)


def open_webcam(
    device_index: int,
    *,
    clock: Callable[[], float] = perf_counter,
    capture_factory: Callable[[int, int], Any] | None = None,
    capture_callback: Callable[[Any | None], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> OpenedWebcam:
    if isinstance(device_index, bool) or not isinstance(device_index, int):
        raise TypeError("Webcam device index must be an integer")
    if device_index < 0:
        raise ValueError("Webcam device index must be non-negative")

    factory = capture_factory or cv2.VideoCapture
    is_cancelled = cancelled or (lambda: False)
    notify_capture = capture_callback or (lambda _capture: None)

    for backend in webcam_backend_preferences():
        if is_cancelled():
            break
        capture = None
        keep_capture = False
        try:
            capture = factory(device_index, backend)
            notify_capture(capture)
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            captured_at = clock()
            if is_cancelled():
                continue
            if ok and frame is not None:
                keep_capture = True
                return OpenedWebcam(capture, frame, captured_at, backend)
        except Exception:
            continue
        finally:
            if capture is not None and not keep_capture:
                capture.release()
                notify_capture(None)

    notify_capture(None)
    raise OSError(f"Unable to open webcam device {device_index}")


def discover_webcams(
    *,
    in_use_indices: Collection[int] = (),
    indices: Iterable[int] = WEBCAM_SCAN_INDICES,
) -> list[int]:
    in_use = {int(index) for index in in_use_indices if int(index) >= 0}
    discovered = set(in_use)
    for index in indices:
        index = int(index)
        if index < 0 or index in in_use:
            continue
        try:
            opened = open_webcam(index)
        except OSError:
            continue
        opened.capture.release()
        discovered.add(index)
    return sorted(discovered)
