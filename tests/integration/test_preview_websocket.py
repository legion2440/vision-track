from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

import cv2
import numpy as np
import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection, connect

from vision_track.context import StreamContext
from vision_track.detections import Detections
from vision_track.lifecycle import StreamState
from vision_track.preview import PreviewJpegCache, PreviewRegistry, PreviewServer
from vision_track.sources import VideoSource


pytestmark = pytest.mark.integration


def _context(stream_id: str, source: str | None = None) -> StreamContext:
    return StreamContext(stream_id, VideoSource.from_uri(source or f"{stream_id}.mp4"))


def _publish(context: StreamContext, value: int) -> tuple[int, int]:
    frame = np.full((54, 96, 3), value, dtype=np.uint8)
    return context.publish_rendered_frame(frame, frame, Detections.empty())


@pytest.fixture
def preview_server() -> Iterator[tuple[PreviewServer, PreviewRegistry, PreviewJpegCache]]:
    cache = PreviewJpegCache()
    registry = PreviewRegistry(invalidate_stream=cache.invalidate_stream)
    server = PreviewServer(registry, cache)
    server.start()
    try:
        yield server, registry, cache
    finally:
        server.stop()


def _connect(server: PreviewServer, session: str, stream_id: str) -> ClientConnection:
    return connect(
        f"ws://127.0.0.1:{server.port}/ws/{session}/{stream_id}",
        open_timeout=2,
        close_timeout=1,
        proxy=None,
    )


def _receive_json(
    connection: ClientConnection,
    *,
    message_type: str | None = None,
    timeout: float = 2.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while True:
        message = connection.recv(timeout=max(0.01, deadline - time.monotonic()))
        if isinstance(message, str):
            payload = json.loads(message)
            if message_type is None or payload.get("type") == message_type:
                return payload
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Did not receive JSON message type {message_type!r}")


def _receive_binary(connection: ClientConnection, timeout: float = 2.0) -> bytes:
    deadline = time.monotonic() + timeout
    while True:
        message = connection.recv(timeout=max(0.01, deadline - time.monotonic()))
        if isinstance(message, bytes):
            return message
        if time.monotonic() >= deadline:
            raise TimeoutError("Did not receive a binary preview frame")


def _jpeg_mean(payload: bytes) -> float:
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    return float(decoded.mean())


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize(
    ("session", "stream_id"),
    [("invalid", "stream-1"), ("session", "unknown")],
)
def test_unknown_authorization_closes_with_policy_violation(
    preview_server,
    session: str,
    stream_id: str,
) -> None:
    server, registry, _ = preview_server
    registry.replace_session("session", [_context("stream-1")])

    connection = _connect(server, session, stream_id)
    try:
        with pytest.raises(ConnectionClosed) as caught:
            connection.recv(timeout=1)
        assert caught.value.rcvd is not None
        assert caught.value.rcvd.code == 1008
    finally:
        connection.close()


def test_initial_state_is_delivered(preview_server) -> None:
    server, registry, _ = preview_server
    registry.replace_session("session", [_context("stream-1")])

    with _connect(server, "session", "stream-1") as connection:
        state = _receive_json(connection, message_type="state")

    assert state == {
        "type": "state",
        "state": "created",
        "has_frame": False,
        "error": None,
    }


def test_publication_sends_one_binary_jpeg_and_unchanged_version_is_not_resent(
    preview_server,
) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(context, 80)
        payload = _receive_binary(connection)
        assert payload.startswith(b"\xff\xd8")
        with pytest.raises(TimeoutError):
            connection.recv(timeout=0.25)


def test_frame_published_before_connection_is_delivered(preview_server) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    _publish(context, 70)
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        state = _receive_json(connection, message_type="state")
        payload = _receive_binary(connection)

    assert state["has_frame"] is True
    assert _jpeg_mean(payload) == pytest.approx(70, abs=3)


def test_rapid_publications_deliver_latest_without_application_queue(preview_server) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(context, 30)
        _publish(context, 90)
        _publish(context, 180)
        payload = _receive_binary(connection)

    assert _jpeg_mean(payload) == pytest.approx(180, abs=4)


@pytest.mark.parametrize("terminal_state", [StreamState.STOPPED, StreamState.EOF])
def test_stop_and_eof_preserve_frame_and_cache_after_reconnect(
    preview_server,
    terminal_state: StreamState,
) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(context, 100)
        first = _receive_binary(connection)
        context.force_state(terminal_state)
        state = _receive_json(connection, message_type="state")
        assert state["state"] == terminal_state.value.lower()
        assert state["has_frame"] is True

    with _connect(server, "session", "stream-1") as connection:
        reconnect_state = _receive_json(connection, message_type="state")
        replayed = _receive_binary(connection)

    assert reconnect_state["has_frame"] is True
    assert replayed == first


def test_restart_like_reset_reuses_cached_jpeg_until_new_publication(preview_server) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(context, 110)
        first = _receive_binary(connection)

    with context.lock:
        context.runtime_generation += 1
        context.latest_rendered_frame = None
        context.latest_rendered_version = None
        context.state = StreamState.CONNECTING

    with _connect(server, "session", "stream-1") as connection:
        state = _receive_json(connection, message_type="state")
        cached = _receive_binary(connection)
        assert state["has_frame"] is True
        _publish(context, 160)
        new = _receive_binary(connection)

    assert cached == first
    assert _jpeg_mean(new) == pytest.approx(160, abs=4)


def test_reset_counter_style_new_render_revision_sends_new_jpeg(preview_server) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        first_version = _publish(context, 60)
        first = _receive_binary(connection)
        second_version = _publish(context, 140)
        second = _receive_binary(connection)

    assert second_version > first_version
    assert first != second
    assert _jpeg_mean(second) == pytest.approx(140, abs=4)


def test_source_replacement_sends_clear_and_does_not_reuse_old_cache(preview_server) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1", "first.mp4")
    registry.replace_session("session", [context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(context, 75)
        _receive_binary(connection)
        with context.lock:
            context.source = VideoSource.from_uri("second.mp4")
            context.latest_rendered_frame = None
            context.latest_rendered_version = None
        clear = _receive_json(connection, message_type="clear")
        assert clear["type"] == "clear"

    with _connect(server, "session", "stream-1") as connection:
        state = _receive_json(connection, message_type="state")
        assert state["has_frame"] is False
        with pytest.raises(TimeoutError):
            connection.recv(timeout=0.25)


def test_backend_context_replacement_changes_revision_and_sends_clear(preview_server) -> None:
    server, registry, _ = preview_server
    old_context = _context("stream-1", "video.mp4")
    registry.replace_session("session", [old_context])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        _publish(old_context, 85)
        _receive_binary(connection)
        old_revision = registry.resolve("session", "stream-1").binding_revision
        new_context = _context("stream-1", "video.mp4")
        registry.replace_session("session", [new_context])
        new_revision = registry.resolve("session", "stream-1").binding_revision
        clear = _receive_json(connection, message_type="clear")

    assert new_revision == old_revision + 1
    assert clear["type"] == "clear"


def test_two_streams_have_independent_connections_frames_and_state(preview_server) -> None:
    server, registry, _ = preview_server
    first_context = _context("first")
    second_context = _context("second")
    registry.replace_session("session", [first_context, second_context])

    with (
        _connect(server, "session", "first") as first_connection,
        _connect(server, "session", "second") as second_connection,
    ):
        _receive_json(first_connection, message_type="state")
        _receive_json(second_connection, message_type="state")
        _publish(first_context, 40)
        _publish(second_context, 190)
        first = _receive_binary(first_connection)
        second = _receive_binary(second_connection)
        first_context.force_state(StreamState.STOPPED)
        first_state = _receive_json(first_connection, message_type="state")
        with pytest.raises(TimeoutError):
            second_connection.recv(timeout=0.2)

    assert _jpeg_mean(first) == pytest.approx(40, abs=4)
    assert _jpeg_mean(second) == pytest.approx(190, abs=4)
    assert first_state["state"] == "stopped"


def test_remove_sends_removed_and_ends_connection(preview_server) -> None:
    server, registry, _ = preview_server
    registry.replace_session("session", [_context("stream-1")])

    with _connect(server, "session", "stream-1") as connection:
        _receive_json(connection, message_type="state")
        registry.remove_stream("session", "stream-1")
        removed = _receive_json(connection, message_type="removed")

    assert removed == {"type": "removed"}


def test_client_disconnect_exits_handler_without_traceback_or_second_send(
    preview_server,
    caplog,
) -> None:
    server, registry, _ = preview_server
    context = _context("stream-1")
    registry.replace_session("session", [context])
    caplog.set_level(logging.ERROR, logger="websockets.server")

    connection = _connect(server, "session", "stream-1")
    _receive_json(connection, message_type="state")
    assert _wait_until(lambda: server.active_connections == 1)
    connection.close()
    assert _wait_until(lambda: server.active_connections == 0)

    _publish(context, 200)
    time.sleep(0.15)

    assert server.active_connections == 0
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
