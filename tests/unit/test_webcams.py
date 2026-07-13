from __future__ import annotations

import cv2
import numpy as np
import pytest

import vision_track.webcams as webcams
from vision_track.webcams import OpenedWebcam, discover_webcams, open_webcam


class FakeCapture:
    def __init__(self, *, opened: bool, read_ok: bool, value: int = 0) -> None:
        self.opened = opened
        self.read_ok = read_ok
        self.value = value
        self.released = False
        self.read_calls = 0

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        self.read_calls += 1
        if not self.read_ok:
            return False, None
        return True, np.full((3, 4, 3), self.value, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_backend_preferences_use_complete_windows_fallback_order() -> None:
    assert webcams.webcam_backend_preferences("win32") == (
        cv2.CAP_MSMF,
        cv2.CAP_DSHOW,
        cv2.CAP_ANY,
    )
    assert webcams.webcam_backend_preferences("linux") == (cv2.CAP_ANY,)


def test_open_webcam_falls_back_after_first_backend_cannot_read() -> None:
    first = FakeCapture(opened=True, read_ok=False)
    second = FakeCapture(opened=True, read_ok=True, value=8)
    captures = {10: first, 20: second}
    factory_calls: list[tuple[int, int]] = []
    callbacks: list[FakeCapture | None] = []

    def factory(index: int, backend: int) -> FakeCapture:
        factory_calls.append((index, backend))
        return captures[backend]

    opened = open_webcam(
        3,
        backends=(10, 20),
        clock=lambda: 42.5,
        capture_factory=factory,
        capture_callback=callbacks.append,
    )

    assert factory_calls == [(3, 10), (3, 20)]
    assert first.read_calls == 1
    assert first.released
    assert not second.released
    assert opened.capture is second
    assert opened.backend == 20
    assert opened.captured_at == 42.5
    assert np.all(opened.first_frame == 8)
    assert callbacks == [first, None, second]


def test_open_webcam_releases_every_failed_backend() -> None:
    captures = [
        FakeCapture(opened=False, read_ok=False),
        FakeCapture(opened=True, read_ok=False),
        FakeCapture(opened=True, read_ok=False),
    ]

    with pytest.raises(OSError, match="device 2"):
        open_webcam(
            2,
            backends=(10, 20, 30),
            capture_factory=lambda _index, backend: captures[backend // 10 - 1],
        )

    assert all(capture.released for capture in captures)
    assert captures[0].read_calls == 0
    assert captures[1].read_calls == 1
    assert captures[2].read_calls == 1


def test_open_webcam_cancellation_does_not_try_a_backend(monkeypatch) -> None:
    monkeypatch.setattr(webcams, "webcam_backend_preferences", lambda: (10, 20))
    calls: list[tuple[int, int]] = []

    with pytest.raises(OSError):
        open_webcam(
            0,
            capture_factory=lambda index, backend: calls.append((index, backend)),
            cancelled=lambda: True,
        )

    assert calls == []


def test_discovery_does_not_reopen_in_use_camera(monkeypatch) -> None:
    opened_indices: list[int] = []
    successful_captures: list[FakeCapture] = []

    def fake_open(index: int) -> OpenedWebcam:
        opened_indices.append(index)
        if index != 2:
            raise OSError("not available")
        capture = FakeCapture(opened=True, read_ok=True)
        successful_captures.append(capture)
        return OpenedWebcam(
            capture=capture,
            first_frame=np.zeros((2, 2, 3), dtype=np.uint8),
            captured_at=1.0,
            backend=0,
        )

    monkeypatch.setattr(webcams, "open_webcam", fake_open)

    assert discover_webcams(in_use_indices={1}, indices=range(4)) == [1, 2]
    assert opened_indices == [0, 2, 3]
    assert successful_captures[0].released
