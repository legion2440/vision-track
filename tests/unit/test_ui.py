from __future__ import annotations

from vision_track.context import StreamContext
from vision_track.lifecycle import StreamState
from vision_track.sources import VideoSource
from vision_track.ui import (
    replay_button_label,
    runtime_backend_summary,
    single_stream_column_weights,
    stream_grid_columns,
)


def test_stream_grid_uses_one_bounded_column_for_single_stream() -> None:
    assert stream_grid_columns(1) == 1
    assert single_stream_column_weights(1) == [1.0, 1.6, 1.0]


def test_stream_grid_uses_two_columns_for_multiple_streams() -> None:
    assert stream_grid_columns(2) == 2
    assert stream_grid_columns(5) == 2


def test_runtime_backend_summary_is_pending_before_first_inference() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    summary = runtime_backend_summary(
        context,
        requested_backend="pytorch",
        requested_device="cuda",
    )
    assert "Requested backend `pytorch`" in summary
    assert "pending first inference" in summary


def test_runtime_backend_summary_reports_actual_provider() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.actual_backend = "onnxruntime"
    context.actual_device = "cpu"
    context.actual_provider = "CPUExecutionProvider"
    summary = runtime_backend_summary(
        context,
        requested_backend="onnxruntime",
        requested_device="cuda",
    )
    assert "Actual backend `onnxruntime`" in summary
    assert "provider `CPUExecutionProvider`" in summary


def test_local_completed_stream_uses_replay_label() -> None:
    context = StreamContext("stream-1", VideoSource.from_uri("video.mp4"))
    context.force_state(StreamState.EOF)
    context.metrics.processed_frames = 1
    assert replay_button_label(context) == "Replay"
