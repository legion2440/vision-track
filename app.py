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


st.set_page_config(page_title="VisionTrack", page_icon="🎯", layout="wide")
config = load_config()


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
        f"Device: `{engine.device.kind}` · {engine.device.name}\n\n"
        f"Backend: `{engine.detector.name}`"
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
        if first.button("Restart", use_container_width=True):
            engine.restart(selected_id)
        if second.button("Reset counters", use_container_width=True):
            engine.reset_counters(selected_id)
        if st.button("Remove stream", use_container_width=True):
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


@st.fragment(run_every="500ms")
def render_dashboard() -> None:
    current_contexts = engine.contexts()
    if not current_contexts:
        st.info("Add a local video or an HTTP/RTSP source from the sidebar.")
        return

    st.subheader("Streams")
    columns = st.columns(min(2, len(current_contexts)))
    for index, context in enumerate(current_contexts):
        with columns[index % len(columns)]:
            st.markdown(f"**{context.source.display_name}**")
            if context.latest_rendered_frame is not None:
                st.image(
                    context.latest_rendered_frame,
                    channels="BGR",
                    use_container_width=True,
                )
            else:
                st.caption("Waiting for frames")
            st.caption(
                f"{context.state.value} · FPS {_metric_value(context.metrics.fps)} · "
                f"inference {_metric_value(context.metrics.inference_latency_ms, ' ms')} · "
                f"dropped {context.queue.dropped_rate:.1%}"
            )
            if context.error:
                st.error(context.error)

    if selected_id and selected_id in [item.stream_id for item in current_contexts]:
        context = engine.get(selected_id)
        st.subheader(f"Details · {context.source.display_name}")
        if context.latest_rendered_frame is not None:
            st.image(
                context.latest_rendered_frame,
                channels="BGR",
                use_container_width=True,
            )
        counter = context.counter
        metric_columns = st.columns(8)
        metric_columns[0].metric("Status", context.state.value)
        metric_columns[1].metric("FPS", _metric_value(context.metrics.fps))
        metric_columns[2].metric(
            "Inference", _metric_value(context.metrics.inference_latency_ms, " ms")
        )
        metric_columns[3].metric(
            "End-to-end", _metric_value(context.metrics.end_to_end_latency_ms, " ms")
        )
        metric_columns[4].metric("Dropped", f"{context.queue.dropped_rate:.1%}")
        metric_columns[5].metric("In", counter.in_count)
        metric_columns[6].metric("Out", counter.out_count)
        metric_columns[7].metric("Occupancy", counter.occupancy)
        st.caption(
            f"Device `{engine.device.kind}` · backend `{engine.detector.name}` · "
            f"source `{context.source.source_type.value}`"
        )


render_dashboard()
