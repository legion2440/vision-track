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
    StreamMetricsSnapshot,
    StreamFrameUpdate,
    clear_stream_frame_cache,
    prune_stream_frame_cache,
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    snapshot_stream_frame,
    snapshot_stream_metrics,
    stream_grid_columns,
    stream_source_token,
    update_stream_frame_cache,
    waiting_slot_transition,
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
        frame_cache.clear()
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

    @st.fragment
    def render_sidebar_stream_controls(
        selected_stream_id: str | None,
        control_stream_ids: tuple[str, ...],
    ) -> None:
        selected_context = None
        if selected_stream_id:
            try:
                selected_context = engine.get(selected_stream_id)
            except KeyError:
                selected_context = None

        if selected_context is not None:
            stream_id = selected_context.stream_id
            st.caption(
                runtime_backend_summary(
                    selected_context,
                    requested_backend=engine.detector.name,
                    requested_device=engine.device.kind,
                )
            )
            with st.form(key=f"sidebar-settings-{stream_id}"):
                confidence = st.slider(
                    "Confidence",
                    0.05,
                    0.95,
                    float(selected_context.options.confidence),
                    0.05,
                    key=f"sidebar-confidence-{stream_id}",
                )
                iou = st.slider(
                    "IoU threshold",
                    0.10,
                    0.90,
                    float(selected_context.options.iou),
                    0.05,
                    key=f"sidebar-iou-{stream_id}",
                )
                detection_enabled = st.toggle(
                    "Detection",
                    value=selected_context.options.detection_enabled,
                    key=f"sidebar-detection-{stream_id}",
                )
                tracking_enabled = st.toggle(
                    "Tracking",
                    value=selected_context.options.tracking_enabled,
                    key=f"sidebar-tracking-{stream_id}",
                )
                counting_enabled = st.toggle(
                    "Counting",
                    value=selected_context.options.counting_enabled,
                    key=f"sidebar-counting-{stream_id}",
                )
                activation = st.slider(
                    "Track activation",
                    0.05,
                    0.90,
                    float(selected_context.tracker.settings.track_activation_threshold),
                    0.05,
                    key=f"sidebar-track-activation-{stream_id}",
                )
                lost_buffer = st.number_input(
                    "Lost track buffer",
                    min_value=1,
                    max_value=300,
                    value=int(selected_context.tracker.settings.lost_track_buffer),
                    key=f"sidebar-lost-buffer-{stream_id}",
                )
                matching = st.slider(
                    "Matching threshold",
                    0.10,
                    0.99,
                    float(selected_context.tracker.settings.minimum_matching_threshold),
                    0.01,
                    key=f"sidebar-matching-{stream_id}",
                )
                apply_settings = st.form_submit_button(
                    "Apply stream settings",
                    use_container_width=True,
                )
            if apply_settings:
                engine.update_options(
                    stream_id,
                    confidence=confidence,
                    iou=iou,
                    detection_enabled=detection_enabled,
                    tracking_enabled=tracking_enabled,
                    counting_enabled=counting_enabled,
                )
                if (
                    activation
                    != selected_context.tracker.settings.track_activation_threshold
                    or lost_buffer != selected_context.tracker.settings.lost_track_buffer
                    or matching
                    != selected_context.tracker.settings.minimum_matching_threshold
                ):
                    engine.update_tracker(
                        stream_id,
                        track_activation_threshold=activation,
                        lost_track_buffer=int(lost_buffer),
                        minimum_matching_threshold=matching,
                    )

            first, second = st.columns(2)
            if first.button(
                "Start",
                key=f"sidebar-start-{stream_id}",
                use_container_width=True,
            ):
                engine.start(stream_id)
            if second.button(
                "Stop",
                key=f"sidebar-stop-{stream_id}",
                use_container_width=True,
            ):
                engine.stop(stream_id)
            if first.button(
                replay_button_label(selected_context),
                key=f"sidebar-restart-{stream_id}",
                use_container_width=True,
            ):
                engine.restart(stream_id)
            if second.button(
                "Reset counters",
                key=f"sidebar-reset-{stream_id}",
                use_container_width=True,
            ):
                engine.reset_counters(stream_id)

        if control_stream_ids:
            left, right = st.columns(2)
            if left.button(
                "Start all",
                key="sidebar-start-all",
                use_container_width=True,
            ):
                engine.start_all()
            if right.button(
                "Stop all",
                key="sidebar-stop-all",
                use_container_width=True,
            ):
                engine.stop_all()

    render_sidebar_stream_controls(selected_id, tuple(stream_ids))

    if selected_id and st.button(
        "Remove stream",
        key=f"sidebar-remove-{selected_id}",
        use_container_width=True,
    ):
        clear_stream_frame_cache(frame_cache, selected_id)
        engine.remove(selected_id)
        st.rerun()


st.title("VisionTrack")
st.caption("Multi-stream person detection, ByteTrack tracking, line counting, and ROI occupancy")


def _metric_value(value: float, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if value else f"0.0{suffix}"


def _stream_metrics_caption(snapshot: StreamMetricsSnapshot) -> str:
    return (
        f"{snapshot.state.value} · FPS {_metric_value(snapshot.fps)} · "
        f"inference {_metric_value(snapshot.inference_latency_ms, ' ms')} · "
        f"end-to-end {_metric_value(snapshot.end_to_end_latency_ms, ' ms')} · "
        f"dropped {snapshot.dropped_rate:.1%} · "
        f"IN {snapshot.in_count} · OUT {snapshot.out_count} · OCC {snapshot.occupancy}"
    )


@st.fragment
def render_detail_controls(stream_id: str) -> None:
    try:
        context = engine.get(stream_id)
    except KeyError:
        return
    control_columns = st.columns(4)
    if control_columns[0].button(
        "Start",
        key=f"detail-start-{stream_id}",
        use_container_width=True,
    ):
        engine.start(stream_id)
    if control_columns[1].button(
        "Stop",
        key=f"detail-stop-{stream_id}",
        use_container_width=True,
    ):
        engine.stop(stream_id)
    if control_columns[2].button(
        replay_button_label(context),
        key=f"detail-restart-{stream_id}",
        use_container_width=True,
    ):
        engine.restart(stream_id)
    if control_columns[3].button(
        "Reset counters",
        key=f"detail-reset-{stream_id}",
        use_container_width=True,
    ):
        engine.reset_counters(stream_id)


dashboard_contexts = contexts
dashboard_stream_ids = [context.stream_id for context in dashboard_contexts]
dashboard_requested_backend = engine.detector.name
dashboard_requested_device = engine.device.kind
stream_placeholders: dict[str, dict[str, object]] = {}
waiting_visible: dict[str, bool] = {}
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
                waiting_visible[context.stream_id] = False
            else:
                waiting_placeholder.caption("Waiting for frames")
                waiting_visible[context.stream_id] = True
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
            render_detail_controls(detail_stream_id)

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
        snapshot = snapshot_stream_frame(
            context,
            cached_frame=cached,
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
        transition = waiting_slot_transition(
            waiting_visible[stream_id],
            update.show_waiting,
        )
        if transition is True:
            waiting_placeholder.caption("Waiting for frames")
            waiting_visible[stream_id] = True
        elif transition is False:
            waiting_placeholder.empty()
            waiting_visible[stream_id] = False


@st.fragment(run_every=0.25)
def render_stream_metrics() -> None:
    snapshots: dict[str, StreamMetricsSnapshot] = {}
    for stream_id in dashboard_stream_ids:
        try:
            context = engine.get(stream_id)
        except KeyError:
            continue
        snapshots[stream_id] = snapshot_stream_metrics(context)

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


if dashboard_stream_ids:
    render_stream_images()
    render_stream_metrics()
