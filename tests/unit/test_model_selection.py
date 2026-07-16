from __future__ import annotations

from vision_track.model_selection import (
    MODEL_SELECT_KEY,
    MODEL_SELECT_OVERRIDE_KEY,
    MODEL_STATE_KEY,
    record_loaded_model,
    record_model_fallback,
    sync_model_selector_before_widget,
)


def test_fallback_syncs_widget_key_on_next_rerun() -> None:
    state = {
        MODEL_SELECT_KEY: "pretrained_l",
        MODEL_STATE_KEY: "pretrained_l",
    }

    record_model_fallback(state, fallback_model_id="fine_tuned_n")

    assert state[MODEL_STATE_KEY] == "fine_tuned_n"
    assert state[MODEL_SELECT_OVERRIDE_KEY] == "fine_tuned_n"

    remembered = sync_model_selector_before_widget(
        state,
        default_model_id="fine_tuned_n",
        selectable_model_ids=["fine_tuned_n", "pretrained_l"],
    )

    assert remembered == "fine_tuned_n"
    assert state[MODEL_SELECT_KEY] == "fine_tuned_n"
    assert MODEL_SELECT_OVERRIDE_KEY not in state


def test_stale_selector_value_is_replaced_by_active_model() -> None:
    state = {
        MODEL_SELECT_KEY: "pretrained_x",
        MODEL_STATE_KEY: "fine_tuned_n",
        MODEL_SELECT_OVERRIDE_KEY: "fine_tuned_n",
    }

    remembered = sync_model_selector_before_widget(
        state,
        default_model_id="fine_tuned_n",
        selectable_model_ids=["fine_tuned_n", "pretrained_x"],
    )

    assert remembered == "fine_tuned_n"
    assert state[MODEL_SELECT_KEY] == "fine_tuned_n"


def test_loaded_model_records_active_model_without_selector_override() -> None:
    state = {}

    record_loaded_model(state, model_id="pretrained_m")

    assert state == {MODEL_STATE_KEY: "pretrained_m"}
