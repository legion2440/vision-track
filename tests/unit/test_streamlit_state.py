from __future__ import annotations

from vision_track.streamlit_state import get_or_create_engine


class Engine:
    _shutdown = False


def test_repeated_reruns_reuse_engine() -> None:
    state = {}
    first = get_or_create_engine(state, Engine)
    second = get_or_create_engine(state, Engine)
    assert first is second

