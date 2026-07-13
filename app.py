from __future__ import annotations

import hashlib
import secrets
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.detector import available_backends
from vision_track.engine import ProcessingEngine
from vision_track.preview import build_preview_component_html, get_preview_runtime
from vision_track.streamlit_state import ENGINE_KEY
from vision_track.ui import (
    StreamMetricsSnapshot,
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    snapshot_stream_controls,
    snapshot_stream_identity,
    snapshot_stream_metrics,
    stream_grid_columns,
)


st.set_page_config(page_title="VisionTrack", page_icon="🎯", layout="wide")
config = load_config()
preview_runtime = get_preview_runtime()
PREVIEW_SESSION_TOKEN_KEY = "vision_preview_session_token_v1"
session_token = st.session_state.get(PREVIEW_SESSION_TOKEN_KEY)
if not isinstance(session_token, str) or not session_token:
    session_token = secrets.token_urlsafe(24)
    st.session_state[PREVIEW_SESSION_TOKEN_KEY] = session_token


def _engine_for_backend(backend_name: str) -> ProcessingEngine:
    existing = st.session_state.get(ENGINE_KEY)
    existing_backend = st.session_state.get("vision_backend")
    if existing is None or existing_backend != backend_name or getattr(existing, "_shutdown", False):
        snapshots = []
        if existing is not None:
            snapshots = existing.snapshot_for_rebuild()
        if existing is not None:
            existing.shutdown()
        existing = ProcessingEngine(config, backend_name=backend_name)
        for snapshot in snapshots:
            existing.add_stream(
                snapshot.source,
                stream_id=snapshot.stream_id,
                options=snapshot.options,
                tracker_settings=snapshot.tracker_settings,
            )
            if snapshot.was_running:
                existing.start(snapshot.stream_id)
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

    contexts = engine.contexts()
    preview_runtime.registry.replace_session(session_token, contexts)
    identities = [
        snapshot_stream_identity(context)
        for context in contexts
    ]
    identity_by_id = {identity.stream_id: identity for identity in identities}
    identity_labels = {
        identity.stream_id: f"{identity.stream_id} · {identity.display_name}"
        for identity in identities
    }

    uploads = st.file_uploader(
        "Add local videos",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        accept_multiple_files=True,
    )
    if st.button("Add uploaded videos", use_container_width=True):
        for upload in uploads or []:
            path = _save_upload(upload)
            if not any(identity.source.uri == str(path) for identity in identities):
                engine.add_stream(str(path))
        st.rerun()

    remote_url = st.text_input("HTTP or RTSP URL", type="password")
    if st.button("Add URL", use_container_width=True, disabled=not remote_url.strip()):
        engine.add_stream(remote_url.strip())
        st.rerun()

    stream_ids = [identity.stream_id for identity in identities]
    selected_id = st.selectbox(
        "Selected stream",
        stream_ids,
        format_func=lambda item: identity_labels.get(item, item),
        index=0 if stream_ids else None,
        placeholder="No streams",
    )

    @st.fragment
    def render_sidebar_stream_controls(
        selected_stream_id: str | None,
        control_stream_ids: tuple[str, ...],
    ) -> None:
        control = None
        if selected_stream_id:
            try:
                control = snapshot_stream_controls(engine.get(selected_stream_id))
            except KeyError:
                control = None

        if control is not None:
            stream_id = control.stream_id
            st.caption(
                runtime_backend_summary(
                    control,
                    requested_backend=engine.detector.name,
                    requested_device=engine.device.kind,
                )
            )
            with st.form(key=f"sidebar-settings-{stream_id}"):
                confidence = st.slider(
                    "Confidence",
                    0.05,
                    0.95,
                    float(control.confidence),
                    0.05,
                    key=f"sidebar-confidence-{stream_id}",
                )
                iou = st.slider(
                    "IoU threshold",
                    0.10,
                    0.90,
                    float(control.iou),
                    0.05,
                    key=f"sidebar-iou-{stream_id}",
                )
                detection_enabled = st.toggle(
                    "Detection",
                    value=control.detection_enabled,
                    key=f"sidebar-detection-{stream_id}",
                )
                tracking_enabled = st.toggle(
                    "Tracking",
                    value=control.tracking_enabled,
                    key=f"sidebar-tracking-{stream_id}",
                )
                counting_enabled = st.toggle(
                    "Counting",
                    value=control.counting_enabled,
                    key=f"sidebar-counting-{stream_id}",
                )
                activation = st.slider(
                    "Track activation",
                    0.05,
                    0.90,
                    float(control.track_activation_threshold),
                    0.05,
                    key=f"sidebar-track-activation-{stream_id}",
                )
                lost_buffer = st.number_input(
                    "Lost track buffer",
                    min_value=1,
                    max_value=300,
                    value=int(control.lost_track_buffer),
                    key=f"sidebar-lost-buffer-{stream_id}",
                )
                matching = st.slider(
                    "Matching threshold",
                    0.10,
                    0.99,
                    float(control.minimum_matching_threshold),
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
                    != control.track_activation_threshold
                    or lost_buffer != control.lost_track_buffer
                    or matching != control.minimum_matching_threshold
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
                replay_button_label(control),
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
        preview_runtime.registry.remove_stream(session_token, selected_id)
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
        control = snapshot_stream_controls(engine.get(stream_id))
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
        replay_button_label(control),
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


def render_stream_metrics_card(stream_id: str) -> None:
    try:
        snapshot = snapshot_stream_metrics(engine.get(stream_id))
    except KeyError:
        return
    st.caption(_stream_metrics_caption(snapshot))
    if snapshot.error:
        st.error(snapshot.error)


def render_detail_metrics(
    stream_id: str,
    requested_backend: str,
    requested_device: str,
) -> None:
    try:
        snapshot = snapshot_stream_metrics(engine.get(stream_id))
    except KeyError:
        return
    metric_columns = st.columns(8)
    metric_columns[0].metric("Status", snapshot.state.value)
    metric_columns[1].metric("FPS", _metric_value(snapshot.fps))
    metric_columns[2].metric(
        "Inference", _metric_value(snapshot.inference_latency_ms, " ms")
    )
    metric_columns[3].metric(
        "End-to-end", _metric_value(snapshot.end_to_end_latency_ms, " ms")
    )
    metric_columns[4].metric("Dropped", f"{snapshot.dropped_rate:.1%}")
    metric_columns[5].metric("In", snapshot.in_count)
    metric_columns[6].metric("Out", snapshot.out_count)
    metric_columns[7].metric("Occupancy", snapshot.occupancy)
    st.caption(
        runtime_backend_summary(
            snapshot,
            requested_backend=requested_backend,
            requested_device=requested_device,
        )
        + f" · source `{snapshot.source_type.value}`"
    )
    if snapshot.error:
        st.error(snapshot.error)


# Keep stream metrics in a single 4 Hz fragment.
# Multiple fragments or an 8 Hz combined rerun rate caused sustained
# CUDA inference degradation on Windows: ~30 ms -> 2–3 s per forward.
@st.fragment(run_every=0.25)
def render_metrics_dashboard(
    stream_ids: tuple[str, ...],
    selected_stream_id: str | None,
    selected_display_name: str | None,
    requested_backend: str,
    requested_device: str,
) -> None:
    if len(stream_ids) == 1:
        _, middle, _ = st.columns(single_stream_column_weights(1))
        metric_columns = [middle]
    else:
        metric_columns = st.columns(stream_grid_columns(len(stream_ids)))

    for index, stream_id in enumerate(stream_ids):
        with metric_columns[index % len(metric_columns)]:
            render_stream_metrics_card(stream_id)

    if selected_stream_id and selected_display_name:
        st.subheader(f"Details · {selected_display_name}")
        render_detail_metrics(
            selected_stream_id,
            requested_backend,
            requested_device,
        )


dashboard_identities = identities
dashboard_stream_ids = [identity.stream_id for identity in dashboard_identities]
dashboard_requested_backend = engine.detector.name
dashboard_requested_device = engine.device.kind

if not dashboard_identities:
    st.info("Add a local video or an HTTP/RTSP source from the sidebar.")
else:
    st.subheader("Streams")
    if len(dashboard_identities) == 1:
        _, middle, _ = st.columns(single_stream_column_weights(1))
        stream_columns = [middle]
    else:
        stream_columns = st.columns(stream_grid_columns(len(dashboard_identities)))

    for index, identity in enumerate(dashboard_identities):
        with stream_columns[index % len(stream_columns)]:
            st.markdown(f"**{identity.display_name}**")
            components.html(
                build_preview_component_html(
                    port=preview_runtime.server.port,
                    session_token=session_token,
                    stream_id=identity.stream_id,
                ),
                height=330,
                scrolling=False,
            )

    detail_identity = (
        identity_by_id.get(selected_id)
        if selected_id and selected_id in dashboard_stream_ids
        else None
    )
    render_metrics_dashboard(
        tuple(dashboard_stream_ids),
        detail_identity.stream_id if detail_identity is not None else None,
        detail_identity.display_name if detail_identity is not None else None,
        dashboard_requested_backend,
        dashboard_requested_device,
    )
    if detail_identity is not None:
        render_detail_controls(detail_identity.stream_id)
