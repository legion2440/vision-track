from __future__ import annotations

import pytest

from vision_track.lifecycle import StreamState, can_transition, validate_transition
from vision_track.logging_utils import mask_sensitive


def test_state_transitions() -> None:
    assert can_transition(StreamState.CREATED, StreamState.CONNECTING)
    assert can_transition(StreamState.ACTIVE, StreamState.EOF)
    with pytest.raises(ValueError):
        validate_transition(StreamState.CREATED, StreamState.EOF)


def test_credentials_are_masked() -> None:
    value = mask_sensitive("rtsp://user:pass@example.test/live?token=abc123")
    assert "user" not in value
    assert "pass" not in value
    assert "abc123" not in value
    assert "***" in value

