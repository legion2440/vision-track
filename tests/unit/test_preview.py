from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np

import vision_track.preview as preview_module
from vision_track.context import StreamContext
from vision_track.lifecycle import StreamState
from vision_track.preview import (
    PreviewBinding,
    PreviewJpegCache,
    PreviewRegistry,
    PreviewSnapshot,
    build_preview_component_html,
    snapshot_preview,
)
from vision_track.sources import VideoSource
from vision_track.ui import stream_source_token


def _context(stream_id: str, source: str | None = None) -> StreamContext:
    return StreamContext(stream_id, VideoSource.from_uri(source or f"{stream_id}.mp4"))


def _snapshot(
    *,
    stream_id: str = "stream-1",
    binding_revision: int = 1,
    source_token: str = "source-a",
    frame_version: tuple[int, int] = (1, 1),
    value: int = 10,
) -> PreviewSnapshot:
    return PreviewSnapshot(
        binding_revision=binding_revision,
        stream_id=stream_id,
        source_token=source_token,
        state=StreamState.ACTIVE,
        error=None,
        frame_version=frame_version,
        has_published_frame=True,
        frame=np.full((8, 8, 3), value, dtype=np.uint8),
    )


def test_registry_first_binding_revision_is_one_and_same_context_preserves_it() -> None:
    registry = PreviewRegistry()
    context = _context("stream-1")

    registry.replace_session("session", [context])
    first = registry.resolve("session", "stream-1")
    registry.replace_session("session", [context])
    second = registry.resolve("session", "stream-1")

    assert first == PreviewBinding(1, context)
    assert second == first


def test_registry_replacement_and_remove_readd_advance_revision() -> None:
    registry = PreviewRegistry()
    first_context = _context("stream-1", "first.mp4")
    second_context = _context("stream-1", "second.mp4")

    registry.replace_session("session", [first_context])
    registry.replace_session("session", [second_context])
    assert registry.resolve("session", "stream-1") == PreviewBinding(2, second_context)

    registry.remove_stream("session", "stream-1")
    registry.replace_session("session", [first_context])
    assert registry.resolve("session", "stream-1") == PreviewBinding(3, first_context)


def test_registry_streams_sessions_and_removal_are_isolated() -> None:
    registry = PreviewRegistry()
    first = _context("first")
    second = _context("second")
    other_session = _context("first", "other.mp4")
    registry.replace_session("session-a", [first, second])
    registry.replace_session("session-b", [other_session])

    registry.remove_stream("session-a", "first")

    assert registry.resolve("session-a", "first") is None
    assert registry.resolve("session-a", "second") is not None
    assert registry.resolve("session-b", "first") is not None
    assert registry.resolve("missing", "first") is None
    assert registry.resolve("session-a", "missing") is None


def test_registry_invalidates_cache_path_for_replace_remove_and_session_remove() -> None:
    invalidated: list[tuple[str, str]] = []
    registry = PreviewRegistry(invalidate_stream=lambda *key: invalidated.append(key))
    first = _context("first")
    second = _context("second")
    registry.replace_session("session", [first, second])

    registry.replace_session("session", [_context("first", "replacement.mp4")])
    registry.remove_stream("session", "first")
    registry.replace_session("session", [first, second])
    registry.remove_session("session")

    assert invalidated.count(("session", "first")) >= 3
    assert invalidated.count(("session", "second")) >= 2


def test_remove_session_preserves_revision_tombstone_and_invalidates_cache() -> None:
    cache = PreviewJpegCache()
    registry = PreviewRegistry(invalidate_stream=cache.invalidate_stream)
    first_context = _context("stream-1", "first.mp4")
    registry.replace_session("session", [first_context])
    first_binding = registry.resolve("session", "stream-1")
    assert first_binding is not None

    cached = cache.get_or_encode(
        "session",
        _snapshot(binding_revision=first_binding.binding_revision),
        is_current=lambda: True,
        encoder=lambda _frame: b"old-jpeg",
    )
    assert cached is not None
    assert cache.record_count() == 1
    assert cache.encode_lock_count() == 1

    registry.remove_session("session")

    assert registry.resolve("session", "stream-1") is None
    assert cache.record_count() == 0
    assert cache.encode_lock_count() == 0

    second_context = _context("stream-1", "second.mp4")
    registry.replace_session("session", [second_context])
    second_binding = registry.resolve("session", "stream-1")
    assert second_binding is not None
    assert second_binding.binding_revision > first_binding.binding_revision
    assert (
        cache.get_compatible(
            "session",
            "stream-1",
            binding_revision=second_binding.binding_revision,
            source_token="source-a",
            current_frame_version=None,
        )
        is None
    )


def test_snapshot_captures_values_under_context_lock(monkeypatch) -> None:
    context = _context("stream-1")
    entered = False

    class GuardLock:
        def __enter__(self):
            nonlocal entered
            entered = True

        def __exit__(self, *_args):
            nonlocal entered
            entered = False

    context.lock = GuardLock()

    def guarded_source_token(source: VideoSource) -> str:
        assert entered
        return stream_source_token(source)

    monkeypatch.setattr(preview_module, "stream_source_token", guarded_source_token)

    snapshot = snapshot_preview(PreviewBinding(1, context), copy_frame=False)

    assert snapshot.stream_id == "stream-1"
    assert not entered


def test_snapshot_unchanged_version_does_not_copy_frame(monkeypatch) -> None:
    context = _context("stream-1")
    context.latest_rendered_version = (2, 4)
    context.latest_rendered_frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    monkeypatch.setattr(
        preview_module.np,
        "ascontiguousarray",
        lambda _frame: (_ for _ in ()).throw(AssertionError("frame copied")),
    )

    snapshot = snapshot_preview(
        PreviewBinding(1, context),
        known_frame_version=(2, 4),
    )

    assert snapshot.frame is None
    assert snapshot.has_published_frame is True


def test_snapshot_new_version_returns_independent_copy_and_masks_error() -> None:
    context = _context("stream-1", "rtsp://user:pass@example.test/live")
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    context.latest_rendered_version = (2, 5)
    context.latest_rendered_frame = frame
    context.error = "rtsp://user:pass@example.test/live?token=secret"

    snapshot = snapshot_preview(
        PreviewBinding(3, context),
        known_frame_version=(2, 4),
    )
    frame[:] = 0

    assert snapshot.frame is not None
    assert np.all(snapshot.frame == 7)
    assert snapshot.binding_revision == 3
    assert snapshot.frame_version == (2, 5)
    assert "user:pass" not in (snapshot.error or "")
    assert "secret" not in (snapshot.error or "")


def test_snapshot_source_token_change_is_visible() -> None:
    context = _context("stream-1", "first.mp4")
    binding = PreviewBinding(1, context)
    first = snapshot_preview(binding, copy_frame=False)
    with context.lock:
        context.source = VideoSource.from_uri("second.mp4")
    second = snapshot_preview(binding, copy_frame=False)

    assert first.source_token != second.source_token


def test_cache_isolated_by_session_and_stream_and_restart_reuses_record() -> None:
    cache = PreviewJpegCache()
    first = _snapshot(stream_id="first", value=1)
    second = _snapshot(stream_id="second", value=2)
    other_session = _snapshot(stream_id="first", value=3)

    first_record = cache.get_or_encode(
        "session-a", first, is_current=lambda: True, encoder=lambda _frame: b"first"
    )
    second_record = cache.get_or_encode(
        "session-a", second, is_current=lambda: True, encoder=lambda _frame: b"second"
    )
    other_record = cache.get_or_encode(
        "session-b", other_session, is_current=lambda: True, encoder=lambda _frame: b"other"
    )

    assert first_record is not None and first_record.jpeg == b"first"
    assert second_record is not None and second_record.jpeg == b"second"
    assert other_record is not None and other_record.jpeg == b"other"
    assert cache.get_compatible(
        "session-a",
        "first",
        binding_revision=1,
        source_token="source-a",
        current_frame_version=None,
    ) is first_record


def test_source_or_binding_change_invalidates_cached_record() -> None:
    cache = PreviewJpegCache()
    snapshot = _snapshot()
    cache.get_or_encode(
        "session", snapshot, is_current=lambda: True, encoder=lambda _frame: b"jpeg"
    )

    assert cache.get_compatible(
        "session",
        "stream-1",
        binding_revision=1,
        source_token="source-b",
        current_frame_version=None,
    ) is None

    cache.get_or_encode(
        "session", snapshot, is_current=lambda: True, encoder=lambda _frame: b"jpeg"
    )
    assert cache.get_compatible(
        "session",
        "stream-1",
        binding_revision=2,
        source_token="source-a",
        current_frame_version=None,
    ) is None


def test_same_version_is_encoded_once_for_concurrent_callers() -> None:
    cache = PreviewJpegCache()
    snapshot = _snapshot()
    calls = 0
    calls_lock = threading.Lock()
    results = []

    def encoder(_frame: np.ndarray) -> bytes:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return b"jpeg"

    def worker() -> None:
        results.append(
            cache.get_or_encode(
                "session", snapshot, is_current=lambda: True, encoder=encoder
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert calls == 1
    assert len(results) == 2
    assert results[0] is results[1]


def test_later_version_cannot_be_overwritten_by_older_concurrent_encode() -> None:
    cache = PreviewJpegCache()
    old = _snapshot(frame_version=(1, 1), value=1)
    new = replace(old, frame_version=(1, 2), frame=np.full((8, 8, 3), 2, dtype=np.uint8))
    current_version = [(1, 1)]
    encoding_started = threading.Event()
    release_old = threading.Event()

    def old_encoder(_frame: np.ndarray) -> bytes:
        encoding_started.set()
        assert release_old.wait(1)
        return b"old"

    old_result = []
    new_result = []
    old_thread = threading.Thread(
        target=lambda: old_result.append(
            cache.get_or_encode(
                "session",
                old,
                is_current=lambda: current_version[0] == old.frame_version,
                encoder=old_encoder,
            )
        )
    )
    old_thread.start()
    assert encoding_started.wait(1)
    current_version[0] = (1, 2)
    new_thread = threading.Thread(
        target=lambda: new_result.append(
            cache.get_or_encode(
                "session",
                new,
                is_current=lambda: current_version[0] == new.frame_version,
                encoder=lambda _frame: b"new",
            )
        )
    )
    new_thread.start()
    release_old.set()
    old_thread.join(timeout=1)
    new_thread.join(timeout=1)

    record = cache.get_compatible(
        "session",
        "stream-1",
        binding_revision=1,
        source_token="source-a",
        current_frame_version=(1, 2),
    )
    assert old_result == [None]
    assert new_result and new_result[0] is record
    assert record is not None and record.jpeg == b"new"


def test_unrelated_streams_do_not_share_global_encode_lock() -> None:
    cache = PreviewJpegCache()
    barrier = threading.Barrier(2)
    failures: list[BaseException] = []

    def worker(stream_id: str) -> None:
        try:
            cache.get_or_encode(
                "session",
                _snapshot(stream_id=stream_id),
                is_current=lambda: True,
                encoder=lambda _frame: (barrier.wait(timeout=1), b"jpeg")[1],
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(stream_id,)) for stream_id in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert failures == []


def test_invalidation_removes_record_and_per_stream_encode_lock() -> None:
    cache = PreviewJpegCache()
    cache.get_or_encode(
        "session", _snapshot(), is_current=lambda: True, encoder=lambda _frame: b"jpeg"
    )
    assert cache.record_count() == 1
    assert cache.encode_lock_count() == 1

    cache.invalidate_stream("session", "stream-1")

    assert cache.record_count() == 0
    assert cache.encode_lock_count() == 0


def test_preview_html_has_one_canvas_and_no_live_image_element() -> None:
    html = build_preview_component_html(
        port=4321,
        session_token="token",
        stream_id="stream-1",
    )

    assert html.count("<canvas") == 1
    assert "<img" not in html.lower()
    assert 'width="960" height="540"' in html


def test_preview_html_fills_iframe_and_draws_jpeg_contained() -> None:
    html = build_preview_component_html(
        port=4321,
        session_token="token",
        stream_id="stream-1",
    )

    assert "aspect-ratio: 16 / 9" not in html
    assert "#preview { position: relative; width: 100%; height: 100%" in html
    assert "const bounds = canvas.getBoundingClientRect();" in html
    assert "const scale = Math.min(" in html
    assert "canvas.width / bitmap.width" in html
    assert "canvas.height / bitmap.height" in html
    assert "const drawX = (canvas.width - drawWidth) / 2;" in html
    assert "const drawY = (canvas.height - drawHeight) / 2;" in html
    assert "context.fillRect(0, 0, canvas.width, canvas.height);" in html
    assert "context.drawImage(bitmap, drawX, drawY, drawWidth, drawHeight);" in html
    assert "context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);" not in html


def test_preview_html_encodes_path_and_guards_reconnect_and_decode_order() -> None:
    html = build_preview_component_html(
        port=4321,
        session_token="token /?",
        stream_id="stream /?",
    )

    assert "ws://127.0.0.1:4321/ws/token%20%2F%3F/stream%20%2F%3F" in html
    assert 'binaryType = "arraybuffer"' in html
    assert "reconnectDelay = 250" in html
    assert "maxReconnectDelay = 3000" in html
    assert "reconnectEnabled = false" in html
    assert 'message.type === "removed"' in html
    assert 'addEventListener("pagehide"' in html
    assert 'addEventListener("beforeunload"' in html
    assert "generation !== decodeGeneration" in html
    assert "pendingJpeg = payload" in html
    assert "pendingJpeg = null" in html
    assert "if (decodeActive) return" in html
    assert "bitmap.close()" in html
    assert "canvas.dataset.frameCount" in html
    assert "canvas.dataset.hasFrame" in html


def test_preview_html_policy_violation_disables_reconnect_and_hides_details() -> None:
    html = build_preview_component_html(
        port=4321,
        session_token="secret-session",
        stream_id="secret-stream",
    )
    handler_start = html.index("socket.onclose = (event) => {")
    handler_end = html.index("socket.onerror", handler_start)
    handler = html[handler_start:handler_end]
    policy_branch = """if (event.code === 1008) {
        reconnectEnabled = false;
        if (reconnectTimer !== null) {
          window.clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }
        setStatus("Preview unavailable");
        return;
      }"""

    assert policy_branch in handler
    assert handler.index("return;") < handler.index("scheduleReconnect();")
    assert handler.count("scheduleReconnect();") == 1
    assert "event.reason" not in handler
    assert "secret-session" not in handler
    assert "secret-stream" not in handler
