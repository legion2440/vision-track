from __future__ import annotations

from .context import StreamContext
from .lifecycle import StreamState
from .sources import SourceType


def stream_grid_columns(stream_count: int) -> int:
    return 1 if stream_count <= 1 else 2


def single_stream_column_weights(stream_count: int) -> list[float]:
    return [1.0, 1.6, 1.0] if stream_count == 1 else [1.0] * stream_grid_columns(stream_count)


def replay_button_label(context: StreamContext) -> str:
    if (
        context.source.source_type is SourceType.LOCAL
        and context.state in {StreamState.EOF, StreamState.FAILED, StreamState.STOPPED}
        and context.metrics.processed_frames > 0
    ):
        return "Replay"
    return "Restart"


def runtime_backend_summary(
    context: StreamContext | None,
    *,
    requested_backend: str,
    requested_device: str,
) -> str:
    if context is None or context.actual_backend is None or context.actual_device is None:
        return (
            f"Requested backend `{requested_backend}` · requested device `{requested_device}` · "
            "actual runtime pending first inference"
        )
    provider = f" · provider `{context.actual_provider}`" if context.actual_provider else ""
    return (
        f"Actual backend `{context.actual_backend}` · actual device `{context.actual_device}`"
        f"{provider}"
    )
