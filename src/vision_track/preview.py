from __future__ import annotations

import asyncio
import atexit
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TypeAlias
from urllib.parse import quote, unquote, urlsplit

import numpy as np
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from .context import StreamContext
from .lifecycle import StreamState
from .logging_utils import mask_sensitive
from .ui import encode_frame_jpeg, stream_source_token


FrameVersion: TypeAlias = tuple[int, int]
PreviewKey: TypeAlias = tuple[str, str]
RegistryInvalidator: TypeAlias = Callable[[str, str], None]

PREVIEW_HOST = "127.0.0.1"
PREVIEW_MAX_FPS = 15.0
PREVIEW_STARTUP_TIMEOUT_SECONDS = 5.0
PREVIEW_CLOSE_REASON = "Preview unavailable"


@dataclass(frozen=True)
class PreviewBinding:
    binding_revision: int
    context: StreamContext


class PreviewRegistry:
    def __init__(
        self,
        *,
        invalidate_stream: RegistryInvalidator | None = None,
    ) -> None:
        self._bindings: dict[str, dict[str, PreviewBinding]] = {}
        self._revisions: dict[PreviewKey, int] = {}
        self._invalidate_stream = invalidate_stream
        self._lock = threading.RLock()

    def replace_session(
        self,
        session_token: str,
        contexts: list[StreamContext],
    ) -> None:
        resolved_contexts: dict[str, StreamContext] = {}
        for context in contexts:
            with context.lock:
                stream_id = context.stream_id
            if stream_id in resolved_contexts:
                raise ValueError(f"Duplicate stream ID: {stream_id}")
            resolved_contexts[stream_id] = context

        invalidated: set[str] = set()
        with self._lock:
            previous = self._bindings.get(session_token, {})
            replacements: dict[str, PreviewBinding] = {}
            for stream_id, context in resolved_contexts.items():
                key = (session_token, stream_id)
                existing = previous.get(stream_id)
                if existing is not None and existing.context is context:
                    revision = existing.binding_revision
                else:
                    revision = self._revisions.get(key, 0) + 1
                    self._revisions[key] = revision
                    if existing is not None:
                        invalidated.add(stream_id)
                replacements[stream_id] = PreviewBinding(revision, context)

            invalidated.update(set(previous) - set(replacements))
            if replacements:
                self._bindings[session_token] = replacements
            else:
                self._bindings.pop(session_token, None)

        self._invalidate_many(session_token, invalidated)

    def resolve(
        self,
        session_token: str,
        stream_id: str,
    ) -> PreviewBinding | None:
        with self._lock:
            return self._bindings.get(session_token, {}).get(stream_id)

    def remove_stream(
        self,
        session_token: str,
        stream_id: str,
    ) -> None:
        removed = False
        with self._lock:
            session = self._bindings.get(session_token)
            if session is not None and stream_id in session:
                session.pop(stream_id)
                removed = True
                if not session:
                    self._bindings.pop(session_token, None)
        if removed:
            self._invalidate_many(session_token, {stream_id})

    def remove_session(self, session_token: str) -> None:
        with self._lock:
            removed = set(self._bindings.pop(session_token, {}))
        self._invalidate_many(session_token, removed)

    def _invalidate_many(self, session_token: str, stream_ids: set[str]) -> None:
        if self._invalidate_stream is None:
            return
        for stream_id in stream_ids:
            self._invalidate_stream(session_token, stream_id)


@dataclass(frozen=True)
class PreviewSnapshot:
    binding_revision: int
    stream_id: str
    source_token: str
    state: StreamState
    error: str | None
    frame_version: FrameVersion | None
    has_published_frame: bool
    frame: np.ndarray | None


def snapshot_preview(
    binding: PreviewBinding,
    *,
    known_frame_version: FrameVersion | None = None,
    copy_frame: bool = True,
) -> PreviewSnapshot:
    context = binding.context
    with context.lock:
        source_token = stream_source_token(context.source)
        frame_version = context.latest_rendered_version
        has_published_frame = (
            frame_version is not None and context.latest_rendered_frame is not None
        )
        should_copy = (
            copy_frame
            and has_published_frame
            and frame_version != known_frame_version
        )
        frame = (
            np.ascontiguousarray(context.latest_rendered_frame).copy()
            if should_copy
            else None
        )
        return PreviewSnapshot(
            binding_revision=binding.binding_revision,
            stream_id=context.stream_id,
            source_token=source_token,
            state=context.state,
            error=mask_sensitive(context.error) if context.error else None,
            frame_version=frame_version,
            has_published_frame=has_published_frame,
            frame=frame,
        )


@dataclass(frozen=True)
class PreviewJpegRecord:
    binding_revision: int
    source_token: str
    frame_version: FrameVersion
    jpeg: bytes


class PreviewJpegCache:
    def __init__(self) -> None:
        self._records: dict[PreviewKey, PreviewJpegRecord] = {}
        self._encode_locks: dict[PreviewKey, threading.Lock] = {}
        self._lock = threading.RLock()

    def get_compatible(
        self,
        session_token: str,
        stream_id: str,
        *,
        binding_revision: int,
        source_token: str,
        current_frame_version: FrameVersion | None,
    ) -> PreviewJpegRecord | None:
        key = (session_token, stream_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if (
                record.binding_revision != binding_revision
                or record.source_token != source_token
            ):
                self._records.pop(key, None)
                return None
            if (
                current_frame_version is not None
                and record.frame_version != current_frame_version
            ):
                self._records.pop(key, None)
                return None
            return record

    def get_or_encode(
        self,
        session_token: str,
        snapshot: PreviewSnapshot,
        *,
        is_current: Callable[[], bool],
        encoder: Callable[[np.ndarray], bytes] = encode_frame_jpeg,
    ) -> PreviewJpegRecord | None:
        if snapshot.frame is None or snapshot.frame_version is None:
            return None

        key = (session_token, snapshot.stream_id)
        with self._lock:
            encode_lock = self._encode_locks.setdefault(key, threading.Lock())

        with encode_lock:
            with self._lock:
                if self._encode_locks.get(key) is not encode_lock:
                    return None
                existing = self._records.get(key)
                if self._record_matches_snapshot(existing, snapshot):
                    return existing

            if not is_current():
                return None
            jpeg = encoder(snapshot.frame)
            if not is_current():
                return None

            record = PreviewJpegRecord(
                binding_revision=snapshot.binding_revision,
                source_token=snapshot.source_token,
                frame_version=snapshot.frame_version,
                jpeg=jpeg,
            )
            with self._lock:
                if self._encode_locks.get(key) is not encode_lock:
                    return None
                existing = self._records.get(key)
                if self._record_matches_snapshot(existing, snapshot):
                    return existing
                if (
                    existing is not None
                    and existing.binding_revision == snapshot.binding_revision
                    and existing.source_token == snapshot.source_token
                    and existing.frame_version > snapshot.frame_version
                ):
                    return None
                self._records[key] = record
                return record

    def discard_record(
        self,
        session_token: str,
        stream_id: str,
        record: PreviewJpegRecord,
    ) -> None:
        key = (session_token, stream_id)
        with self._lock:
            if self._records.get(key) is record:
                self._records.pop(key, None)

    def invalidate_stream(self, session_token: str, stream_id: str) -> None:
        key = (session_token, stream_id)
        with self._lock:
            self._records.pop(key, None)
            self._encode_locks.pop(key, None)

    def invalidate_session(self, session_token: str) -> None:
        with self._lock:
            keys = {
                key
                for key in set(self._records) | set(self._encode_locks)
                if key[0] == session_token
            }
            for key in keys:
                self._records.pop(key, None)
                self._encode_locks.pop(key, None)

    def record_count(self) -> int:
        with self._lock:
            return len(self._records)

    def encode_lock_count(self) -> int:
        with self._lock:
            return len(self._encode_locks)

    @staticmethod
    def _record_matches_snapshot(
        record: PreviewJpegRecord | None,
        snapshot: PreviewSnapshot,
    ) -> bool:
        return bool(
            record is not None
            and record.binding_revision == snapshot.binding_revision
            and record.source_token == snapshot.source_token
            and record.frame_version == snapshot.frame_version
        )


def _parse_preview_path(path: str) -> tuple[str, str] | None:
    parsed = urlsplit(path)
    if parsed.query or parsed.fragment:
        return None
    segments = parsed.path.split("/")
    if len(segments) != 4 or segments[0] != "" or segments[1] != "ws":
        return None
    if not segments[2] or not segments[3]:
        return None
    try:
        session_token = unquote(segments[2], errors="strict")
        stream_id = unquote(segments[3], errors="strict")
    except UnicodeError:
        return None
    if not session_token or not stream_id:
        return None
    return session_token, stream_id


def _state_payload(snapshot: PreviewSnapshot, *, has_frame: bool) -> dict[str, object]:
    return {
        "type": "state",
        "state": snapshot.state.value.lower(),
        "has_frame": has_frame,
        "error": snapshot.error,
    }


def _clear_payload(snapshot: PreviewSnapshot) -> dict[str, object]:
    return {
        "type": "clear",
        "state": snapshot.state.value.lower(),
        "error": snapshot.error,
    }


class PreviewServer:
    def __init__(
        self,
        registry: PreviewRegistry,
        cache: PreviewJpegCache,
        *,
        port: int = 0,
        max_fps: float = PREVIEW_MAX_FPS,
        startup_timeout: float = PREVIEW_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        if max_fps <= 0:
            raise ValueError("Preview max FPS must be positive")
        self.registry = registry
        self.cache = cache
        self.host = PREVIEW_HOST
        self.requested_port = port
        self.max_fps = max_fps
        self.startup_timeout = startup_timeout
        self._poll_interval = 1.0 / max_fps
        self._port: int | None = None
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_stop: asyncio.Event | None = None
        self._startup_event = threading.Event()
        self._startup_error: BaseException | None = None
        self._start_lock = threading.Lock()
        self._started = False
        self._active_connections = 0
        self._connection_lock = threading.Lock()

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("Preview server has not started")
        return self._port

    @property
    def active_connections(self) -> int:
        with self._connection_lock:
            return self._active_connections

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                if self._thread is not None and self._thread.is_alive():
                    return
                raise RuntimeError("Preview server cannot be restarted after shutdown")
            self._started = True
            self._thread = threading.Thread(
                target=self._thread_main,
                name="vision-preview-websocket",
                daemon=True,
            )
            self._thread.start()

        if not self._startup_event.wait(self.startup_timeout):
            self.stop()
            raise TimeoutError("Preview server did not start within the timeout")
        if self._startup_error is not None:
            raise RuntimeError("Preview server failed to start") from self._startup_error

    def stop(self, timeout: float = 5.0) -> None:
        loop = self._loop
        stop_event = self._async_stop
        if loop is not None and stop_event is not None and loop.is_running():
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_server())
        except BaseException as exc:
            self._startup_error = exc
            self._startup_event.set()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _run_server(self) -> None:
        self._async_stop = asyncio.Event()
        async with serve(
            self._handle_connection,
            self.host,
            self.requested_port,
            compression=None,
        ) as websocket_server:
            sockets = websocket_server.sockets
            if not sockets:
                raise RuntimeError("Preview server did not publish a listening socket")
            self._port = int(sockets[0].getsockname()[1])
            self._startup_event.set()
            await self._async_stop.wait()

    async def _handle_connection(self, connection: ServerConnection) -> None:
        request = connection.request
        parsed_path = _parse_preview_path(request.path if request is not None else "")
        if parsed_path is None:
            await connection.close(code=1008, reason=PREVIEW_CLOSE_REASON)
            return
        session_token, stream_id = parsed_path
        binding = self.registry.resolve(session_token, stream_id)
        if binding is None:
            await connection.close(code=1008, reason=PREVIEW_CLOSE_REASON)
            return

        with self._connection_lock:
            self._active_connections += 1
        wait_closed_task = asyncio.create_task(connection.wait_closed())
        try:
            await self._send_preview(
                connection,
                wait_closed_task,
                session_token=session_token,
                stream_id=stream_id,
                initial_binding=binding,
            )
        except ConnectionClosed:
            pass
        finally:
            if not wait_closed_task.done():
                wait_closed_task.cancel()
            await asyncio.gather(wait_closed_task, return_exceptions=True)
            with self._connection_lock:
                self._active_connections -= 1

    async def _send_preview(
        self,
        connection: ServerConnection,
        wait_closed_task: asyncio.Task[None],
        *,
        session_token: str,
        stream_id: str,
        initial_binding: PreviewBinding,
    ) -> None:
        binding_revision = initial_binding.binding_revision
        initial_snapshot = snapshot_preview(initial_binding, copy_frame=False)
        source_token = initial_snapshot.source_token
        last_sent_version: FrameVersion | None = None
        last_state_payload: dict[str, object] | None = None
        last_binary_sent_at: float | None = None

        while not wait_closed_task.done():
            binding = self.registry.resolve(session_token, stream_id)
            if binding is None:
                await connection.send(json.dumps({"type": "removed"}))
                return

            snapshot = snapshot_preview(binding, copy_frame=False)
            identity_changed = (
                binding.binding_revision != binding_revision
                or snapshot.source_token != source_token
            )
            if identity_changed:
                self.cache.invalidate_stream(session_token, stream_id)
                binding_revision = binding.binding_revision
                source_token = snapshot.source_token
                last_sent_version = None
                last_state_payload = None
                await connection.send(json.dumps(_clear_payload(snapshot)))

            cached = self.cache.get_compatible(
                session_token,
                stream_id,
                binding_revision=binding.binding_revision,
                source_token=snapshot.source_token,
                current_frame_version=snapshot.frame_version,
            )
            has_frame = cached is not None or snapshot.has_published_frame
            state_payload = _state_payload(snapshot, has_frame=has_frame)
            if state_payload != last_state_payload:
                await connection.send(json.dumps(state_payload))
                last_state_payload = state_payload

            record_to_send = (
                cached
                if cached is not None and cached.frame_version != last_sent_version
                else None
            )
            if (
                record_to_send is None
                and snapshot.has_published_frame
                and snapshot.frame_version != last_sent_version
            ):
                frame_snapshot = snapshot_preview(
                    binding,
                    known_frame_version=last_sent_version,
                    copy_frame=True,
                )
                if frame_snapshot.frame is not None:
                    record_to_send = await asyncio.to_thread(
                        self.cache.get_or_encode,
                        session_token,
                        frame_snapshot,
                        is_current=lambda: self._snapshot_is_current(
                            session_token,
                            stream_id,
                            frame_snapshot,
                        ),
                    )

            if record_to_send is not None:
                remaining = self._remaining_rate_limit(last_binary_sent_at)
                if remaining > 0 and await self._wait_or_closed(wait_closed_task, remaining):
                    return
                if not self._record_is_current(
                    session_token,
                    stream_id,
                    record_to_send,
                ):
                    self.cache.discard_record(session_token, stream_id, record_to_send)
                    continue
                await connection.send(record_to_send.jpeg)
                last_sent_version = record_to_send.frame_version
                last_binary_sent_at = monotonic()

            if await self._wait_or_closed(wait_closed_task, self._poll_interval):
                return

    def _snapshot_is_current(
        self,
        session_token: str,
        stream_id: str,
        snapshot: PreviewSnapshot,
    ) -> bool:
        binding = self.registry.resolve(session_token, stream_id)
        if binding is None or binding.binding_revision != snapshot.binding_revision:
            return False
        current = snapshot_preview(binding, copy_frame=False)
        return (
            current.source_token == snapshot.source_token
            and current.frame_version == snapshot.frame_version
            and current.has_published_frame
        )

    def _record_is_current(
        self,
        session_token: str,
        stream_id: str,
        record: PreviewJpegRecord,
    ) -> bool:
        binding = self.registry.resolve(session_token, stream_id)
        if binding is None or binding.binding_revision != record.binding_revision:
            return False
        current = snapshot_preview(binding, copy_frame=False)
        if current.source_token != record.source_token:
            return False
        return current.frame_version is None or current.frame_version == record.frame_version

    def _remaining_rate_limit(self, last_sent_at: float | None) -> float:
        if last_sent_at is None:
            return 0.0
        return max(0.0, self._poll_interval - (monotonic() - last_sent_at))

    @staticmethod
    async def _wait_or_closed(
        wait_closed_task: asyncio.Task[None],
        delay: float,
    ) -> bool:
        sleep_task = asyncio.create_task(asyncio.sleep(delay))
        done, _ = await asyncio.wait(
            {wait_closed_task, sleep_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if sleep_task not in done:
            sleep_task.cancel()
            await asyncio.gather(sleep_task, return_exceptions=True)
        return wait_closed_task in done


class PreviewRuntime:
    def __init__(self) -> None:
        self.cache = PreviewJpegCache()
        self.registry = PreviewRegistry(invalidate_stream=self.cache.invalidate_stream)
        self.server = PreviewServer(self.registry, self.cache)
        self.server.start()

    def stop(self) -> None:
        self.server.stop()


_runtime: PreviewRuntime | None = None
_runtime_lock = threading.Lock()


def get_preview_runtime() -> PreviewRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            runtime = PreviewRuntime()
            atexit.register(runtime.stop)
            _runtime = runtime
        return _runtime


def build_preview_component_html(
    *,
    port: int,
    session_token: str,
    stream_id: str,
) -> str:
    encoded_session = quote(session_token, safe="")
    encoded_stream = quote(stream_id, safe="")
    endpoint = json.dumps(
        f"ws://{PREVIEW_HOST}:{port}/ws/{encoded_session}/{encoded_stream}"
    )
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{ margin: 0; padding: 0; background: #000; overflow: hidden; }}
#preview {{ position: relative; width: 100%; aspect-ratio: 16 / 9; background: #000; }}
canvas {{ display: block; width: 100%; height: 100%; background: #000; }}
#status {{
  position: absolute; inset: auto 0 0 0; padding: 8px 12px;
  color: #fff; background: rgba(0, 0, 0, 0.62);
  font: 14px/1.35 system-ui, sans-serif; white-space: pre-wrap;
}}
#status:empty {{ display: none; }}
</style>
</head>
<body>
<div id="preview">
  <canvas id="canvas" width="960" height="540"></canvas>
  <div id="status">Waiting for frames</div>
</div>
<script>
(() => {{
  const endpoint = {endpoint};
  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("2d", {{ alpha: false }});
  const status = document.getElementById("status");
  let socket = null;
  let reconnectTimer = null;
  let reconnectDelay = 250;
  const maxReconnectDelay = 3000;
  let reconnectEnabled = true;
  let canvasHasFrame = false;
  let decodeGeneration = 0;
  let drawnFrameCount = 0;
  let decodeActive = false;
  let pendingJpeg = null;
  let latestState = "created";
  let latestError = null;

  function setStatus(message) {{ status.textContent = message || ""; }}

  function renderState(state, error) {{
    latestState = state;
    latestError = error;
    if (state === "failed") {{ setStatus(error || "Preview failed"); return; }}
    if (state === "reconnecting") {{ setStatus("Reconnecting"); return; }}
    if (state === "stopped") {{ setStatus(canvasHasFrame ? "Stopped" : "Waiting for frames"); return; }}
    if (state === "eof") {{ setStatus(canvasHasFrame ? "EOF" : "Waiting for frames"); return; }}
    if (!canvasHasFrame) {{ setStatus("Waiting for frames"); return; }}
    setStatus("");
  }}

  function clearCanvas() {{
    decodeGeneration += 1;
    pendingJpeg = null;
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    canvasHasFrame = false;
    canvas.dataset.hasFrame = "false";
  }}

  async function decodeLatestJpeg() {{
    if (decodeActive) return;
    decodeActive = true;
    try {{
      while (pendingJpeg !== null) {{
        const payload = pendingJpeg;
        pendingJpeg = null;
        const generation = ++decodeGeneration;
        const bitmap = await createImageBitmap(new Blob([payload], {{ type: "image/jpeg" }}));
        try {{
          if (generation !== decodeGeneration) continue;
          context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          canvasHasFrame = true;
          canvas.dataset.hasFrame = "true";
          drawnFrameCount += 1;
          canvas.dataset.frameCount = String(drawnFrameCount);
          renderState(latestState, latestError);
        }} finally {{
          bitmap.close();
        }}
      }}
    }} finally {{
      decodeActive = false;
      if (pendingJpeg !== null) void decodeLatestJpeg();
    }}
  }}

  function queueJpeg(payload) {{
    pendingJpeg = payload;
    void decodeLatestJpeg();
  }}

  function scheduleReconnect() {{
    if (!reconnectEnabled || reconnectTimer !== null) return;
    reconnectTimer = window.setTimeout(() => {{
      reconnectTimer = null;
      connect();
    }}, reconnectDelay);
    reconnectDelay = Math.min(maxReconnectDelay, reconnectDelay * 2);
  }}

  function connect() {{
    if (!reconnectEnabled) return;
    socket = new WebSocket(endpoint);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {{ reconnectDelay = 250; }};
    socket.onmessage = (event) => {{
      if (typeof event.data !== "string") {{
        queueJpeg(event.data);
        return;
      }}
      const message = JSON.parse(event.data);
      if (message.type === "state") {{
        renderState(message.state, message.error);
      }} else if (message.type === "clear") {{
        clearCanvas();
        renderState(message.state, message.error);
      }} else if (message.type === "removed") {{
        reconnectEnabled = false;
        clearCanvas();
        setStatus("Stream removed");
        if (socket) socket.close(1000, "removed");
      }}
    }};
    socket.onclose = (event) => {{
      socket = null;
      if (event.code === 1008) {{
        reconnectEnabled = false;
        if (reconnectTimer !== null) {{
          window.clearTimeout(reconnectTimer);
          reconnectTimer = null;
        }}
        setStatus("Preview unavailable");
        return;
      }}
      scheduleReconnect();
    }};
    socket.onerror = () => {{
      if (socket) socket.close();
    }};
  }}

  function shutdown() {{
    reconnectEnabled = false;
    if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
    if (socket) socket.close(1000, "iframe unload");
  }}

  window.addEventListener("pagehide", shutdown, {{ once: true }});
  window.addEventListener("beforeunload", shutdown, {{ once: true }});
  canvas.dataset.frameCount = "0";
  canvas.dataset.hasFrame = "false";
  clearCanvas();
  connect();
}})();
</script>
</body>
</html>"""
