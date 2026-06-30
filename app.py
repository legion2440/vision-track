from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.detector import available_backends
from vision_track.engine import ProcessingEngine
from vision_track.lifecycle import StreamState
from vision_track.streamlit_state import ENGINE_KEY
from vision_track.ui import (
    CachedStreamFrame,
    StreamFrameUpdate,
    StreamUISnapshot,
    clear_stream_frame_cache,
    prune_stream_frame_cache,
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    snapshot_stream_context,
    stream_grid_columns,
    stream_source_token,
    update_stream_frame_cache,
)


st.set_page_config(page_title="VisionTrack", page_icon="🎯", layout="wide")
config = load_config()
FRAME_CACHE_SESSION_KEY = "vision_frame_cache_v1"
raw_frame_cache = st.session_state.get(FRAME_CACHE_SESSION_KEY)
if not isinstance(raw_frame_cache, dict):
    raw_frame_cache = {}
    st.session_state[FRAME_CACHE_SESSION_KEY] = raw_frame_cache
frame_cache: dict[str, CachedStreamFrame] = raw_frame_cache


def _engine_for_backend(backend_name: str) -> ProcessingEngine:
    existing = st.session_state.get(ENGINE_KEY)
    existing_backend = st.session_state.get("vision_backend")
    if existing is None or existing_backend != backend_name or getattr(existing, "_shutdown", False):
        snapshots = []
        if existing is not None:
            snapshots = [
                (
                    context.stream_id,
                    context.source,
                    context.options,
                    context.state
                    in {
                        StreamState.CONNECTING,
                        StreamState.ACTIVE,
                        StreamState.RECONNECTING,
                    },
                )
                for context in existing.contexts()
            ]
            existing.shutdown()
        existing = ProcessingEngine(config, backend_name=backend_name)
        for stream_id, source, options, was_running in snapshots:
            existing.add_stream(source, stream_id=stream_id, options=options)
            if was_running:
                existing.start(stream_id)
        st.session_state[ENGINE_KEY] = existing
        st.session_state["vision_backend"] = backend_name
    return existing


def _save_upload(uploaded_file) -> Path:
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    destination = resolve_project_path(f"data/demo/upload-{digest}{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(payload)
    return destination


pytorch_path = resolve_project_path(config.model.checkpoint)
if not pytorch_path.exists():
    pytorch_path = Path(config.model.pretrained)
onnx_path = resolve_project_path(config.model.quantized_checkpoint)
backend_choices = available_backends(pytorch_path, onnx_path)
if not backend_choices:
    backend_choices = ["pytorch"]

with st.sidebar:
    st.header("VisionTrack")
    backend_name = st.selectbox("Detector backend", backend_choices)
    engine = _engine_for_backend(backend_name)
    st.caption(
        f"Requested device: `{engine.device.kind}` · {engine.device.name}\n\n"
        f"Requested backend: `{engine.detector.name}`"
    )

    uploads = st.file_uploader(
        "Add local videos",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        accept_multiple_files=True,
    )
    if st.button("Add uploaded videos", use_container_width=True):
        for upload in uploads or []:
            path = _save_upload(upload)
            if not any(context.source.uri == str(path) for context in engine.contexts()):
                engine.add_stream(str(path))
        st.rerun()

    remote_url = st.text_input("HTTP or RTSP URL", type="password")
    if st.button("Add URL", use_container_width=True, disabled=not remote_url.strip()):
        engine.add_stream(remote_url.strip())
        st.rerun()

    contexts = engine.contexts()
    active_sources = {
        context.stream_id: stream_source_token(context.source) for context in contexts
    }
    prune_stream_frame_cache(frame_cache, active_sources)
    stream_ids = [context.stream_id for context in contexts]
    selected_id = st.selectbox(
        "Selected stream",
        stream_ids,
        format_func=lambda item: f"{item} · {engine.get(item).source.display_name}",
        index=0 if stream_ids else None,
        placeholder="No streams",
    )

    if selected_id:
        selected = engine.get(selected_id)
        st.caption(
            runtime_backend_summary(
                selected,
                requested_backend=engine.detector.name,
                requested_device=engine.device.kind,
            )
        )
        confidence = st.slider(
            "Confidence", 0.05, 0.95, float(selected.options.confidence), 0.05
        )
        iou = st.slider("IoU threshold", 0.10, 0.90, float(selected.options.iou), 0.05)
        detection_enabled = st.toggle(
            "Detection", value=selected.options.detection_enabled
        )
        tracking_enabled = st.toggle(
            "Tracking", value=selected.options.tracking_enabled
        )
        counting_enabled = st.toggle(
            "Counting", value=selected.options.counting_enabled
        )
        activation = st.slider(
            "Track activation",
            0.05,
            0.90,
            float(selected.tracker.settings.track_activation_threshold),
            0.05,
        )
        lost_buffer = st.number_input(
            "Lost track buffer",
            min_value=1,
            max_value=300,
            value=int(selected.tracker.settings.lost_track_buffer),
        )
        matching = st.slider(
            "Matching threshold",
            0.10,
            0.99,
            float(selected.tracker.settings.minimum_matching_threshold),
            0.01,
        )
        if st.button("Apply stream settings", use_container_width=True):
            engine.update_options(
                selected_id,
                confidence=confidence,
                iou=iou,
                detection_enabled=detection_enabled,
                tracking_enabled=tracking_enabled,
                counting_enabled=counting_enabled,
            )
            if (
                activation != selected.tracker.settings.track_activation_threshold
                or lost_buffer != selected.tracker.settings.lost_track_buffer
                or matching != selected.tracker.settings.minimum_matching_threshold
            ):
                engine.update_tracker(
                    selected_id,
                    track_activation_threshold=activation,
                    lost_track_buffer=int(lost_buffer),
                    minimum_matching_threshold=matching,
                )

        first, second = st.columns(2)
        if first.button("Start", use_container_width=True):
            engine.start(selected_id)
        if second.button("Stop", use_container_width=True):
            engine.stop(selected_id)
        if first.button(replay_button_label(selected), use_container_width=True):
            engine.restart(selected_id)
        if second.button("Reset counters", use_container_width=True):
            engine.reset_counters(selected_id)
        if st.button("Remove stream", use_container_width=True):
            clear_stream_frame_cache(frame_cache, selected_id)
            engine.remove(selected_id)
            st.rerun()

    if contexts:
        left, right = st.columns(2)
        if left.button("Start all", use_container_width=True):
            engine.start_all()
        if right.button("Stop all", use_container_width=True):
            engine.stop_all()


st.title("VisionTrack")
st.caption("Multi-stream person detection, ByteTrack tracking, line counting, and ROI occupancy")


def _metric_value(value: float, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value else f"0.0{suffix}"


def _stream_metrics_caption(snapshot: StreamUISnapshot) -> str:
    return (
        f"{snapshot.state.value} · FPS {_metric_value(snapshot.fps)} · "
        f"inference {_metric_value(snapshot.inference_latency_ms, ' ms')} · "
        f"end-to-end {_metric_value(snapshot.end_to_end_latency_ms, ' ms')} · "
        f"dropped {snapshot.dropped_rate:.1%} · "
        f"IN {snapshot.in_count} · OUT {snapshot.out_count} · OCC {snapshot.occupancy}"
    )


def _render_detail_controls(context) -> None:
    control_columns = st.columns(4)
    if control_columns[0].button(
        "Start",
        key=f"detail-start-{context.stream_id}",
        use_container_width=True,
    ):
        engine.start(context.stream_id)
    if control_columns[1].button(
        "Stop",
        key=f"detail-stop-{context.stream_id}",
        use_container_width=True,
    ):
        engine.stop(context.stream_id)
    if control_columns[2].button(
        replay_button_label(context),
        key=f"detail-restart-{context.stream_id}",
        use_container_width=True,
    ):
        engine.restart(context.stream_id)
    if control_columns[3].button(
        "Reset counters",
        key=f"detail-reset-{context.stream_id}",
        use_container_width=True,
    ):
        engine.reset_counters(context.stream_id)


dashboard_contexts = contexts
dashboard_stream_ids = [context.stream_id for context in dashboard_contexts]
dashboard_requested_backend = engine.detector.name
dashboard_requested_device = engine.device.kind
stream_placeholders: dict[str, dict[str, object]] = {}
detail_metric_placeholders = []
detail_runtime_placeholder = None
detail_stream_id = None

if not dashboard_contexts:
    st.info("Add a local video or an HTTP/RTSP source from the sidebar.")
else:
    st.subheader("Streams")
    if len(dashboard_contexts) == 1:
        _, middle, _ = st.columns(single_stream_column_weights(1))
        stream_columns = [middle]
    else:
        stream_columns = st.columns(stream_grid_columns(len(dashboard_contexts)))

    for index, context in enumerate(dashboard_contexts):
        with stream_columns[index % len(stream_columns)]:
            st.markdown(f"**{context.source.display_name}**")
            image_placeholder = st.empty()
            waiting_placeholder = st.empty()
            cached = frame_cache.get(context.stream_id)
            if (
                cached is not None
                and cached.source_token == active_sources[context.stream_id]
            ):
                image_placeholder.image(
                    cached.jpeg,
                    width="stretch",
                )
                waiting_placeholder.empty()
            else:
                waiting_placeholder.caption("Waiting for frames")
            stream_placeholders[context.stream_id] = {
                "image": image_placeholder,
                "waiting": waiting_placeholder,
                "metrics": st.empty(),
                "error": st.empty(),
            }

    if selected_id and selected_id in dashboard_stream_ids:
        try:
            detail_context = engine.get(selected_id)
        except KeyError:
            detail_context = None
        if detail_context is not None:
            detail_stream_id = selected_id
            st.subheader(f"Details · {detail_context.source.display_name}")
            metric_columns = st.columns(8)
            detail_metric_placeholders = [column.empty() for column in metric_columns]
            detail_runtime_placeholder = st.empty()
            _render_detail_controls(detail_context)

@st.fragment(run_every=0.01)
def render_stream_images() -> None:
    for stream_id in dashboard_stream_ids:
        try:
            context = engine.get(stream_id)
        except KeyError:
            continue
        placeholders = stream_placeholders.get(stream_id)
        if placeholders is None:
            continue
        cached = frame_cache.get(stream_id)
        snapshot = snapshot_stream_context(
            context,
            cached_frame=cached,
            include_frame=True,
        )
        update: StreamFrameUpdate = update_stream_frame_cache(frame_cache, snapshot)

        image_placeholder = placeholders["image"]
        waiting_placeholder = placeholders["waiting"]
        if update.clear_image:
            image_placeholder.empty()
        if update.render_jpeg is not None:
            image_placeholder.image(
                update.render_jpeg,
                width="stretch",
            )
        if update.show_waiting:
            waiting_placeholder.caption("Waiting for frames")
        else:
            waiting_placeholder.empty()


@st.fragment(run_every=0.25)
def render_stream_metrics() -> None:
    snapshots: dict[str, StreamUISnapshot] = {}
    for stream_id in dashboard_stream_ids:
        try:
            context = engine.get(stream_id)
        except KeyError:
            continue
        snapshots[stream_id] = snapshot_stream_context(
            context,
            cached_frame=None,
            include_frame=False,
        )

    for stream_id, placeholders in stream_placeholders.items():
        snapshot = snapshots.get(stream_id)
        if snapshot is None:
            continue

        placeholders["metrics"].caption(_stream_metrics_caption(snapshot))
        if snapshot.error:
            placeholders["error"].error(snapshot.error)
        else:
            placeholders["error"].empty()

    if (
        detail_stream_id
        and detail_runtime_placeholder is not None
        and len(detail_metric_placeholders) == 8
    ):
        snapshot = snapshots.get(detail_stream_id)
        if snapshot is None:
            return
        detail_metric_placeholders[0].metric("Status", snapshot.state.value)
        detail_metric_placeholders[1].metric("FPS", _metric_value(snapshot.fps))
        detail_metric_placeholders[2].metric(
            "Inference", _metric_value(snapshot.inference_latency_ms, " ms")
        )
        detail_metric_placeholders[3].metric(
            "End-to-end", _metric_value(snapshot.end_to_end_latency_ms, " ms")
        )
        detail_metric_placeholders[4].metric("Dropped", f"{snapshot.dropped_rate:.1%}")
        detail_metric_placeholders[5].metric("In", snapshot.in_count)
        detail_metric_placeholders[6].metric("Out", snapshot.out_count)
        detail_metric_placeholders[7].metric("Occupancy", snapshot.occupancy)
        detail_runtime_placeholder.caption(
            runtime_backend_summary(
                snapshot,
                requested_backend=dashboard_requested_backend,
                requested_device=dashboard_requested_device,
            )
            + f" · source `{snapshot.source_type.value}`"
        )


render_stream_images()
render_stream_metrics()
