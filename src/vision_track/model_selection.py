from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any


MODEL_STATE_KEY = "vision_model_id"
MODEL_SELECT_KEY = "vision_model_select_v1"
MODEL_SELECT_OVERRIDE_KEY = "vision_model_select_override_v1"


def sync_model_selector_before_widget(
    session_state: MutableMapping[str, Any],
    *,
    default_model_id: str,
    selectable_model_ids: Sequence[str],
) -> str:
    selectable = set(selectable_model_ids)
    override = session_state.pop(MODEL_SELECT_OVERRIDE_KEY, None)
    remembered = (
        override
        if override in selectable
        else session_state.get(
            MODEL_SELECT_KEY,
            session_state.get(MODEL_STATE_KEY, default_model_id),
        )
    )
    if remembered not in selectable:
        remembered = default_model_id
    session_state[MODEL_SELECT_KEY] = remembered
    return str(remembered)


def record_loaded_model(
    session_state: MutableMapping[str, Any],
    *,
    model_id: str,
) -> None:
    session_state[MODEL_STATE_KEY] = model_id


def record_model_fallback(
    session_state: MutableMapping[str, Any],
    *,
    fallback_model_id: str,
) -> None:
    session_state[MODEL_STATE_KEY] = fallback_model_id
    session_state[MODEL_SELECT_OVERRIDE_KEY] = fallback_model_id
