from __future__ import annotations

from typing import MutableMapping

from .engine import ProcessingEngine


ENGINE_KEY = "vision_track_engine"


def get_or_create_engine(
    session_state: MutableMapping,
    factory=ProcessingEngine,
) -> ProcessingEngine:
    engine = session_state.get(ENGINE_KEY)
    if engine is None or getattr(engine, "_shutdown", False):
        engine = factory()
        session_state[ENGINE_KEY] = engine
    return engine

