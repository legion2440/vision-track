from __future__ import annotations

from logging.handlers import RotatingFileHandler

import pytest

from vision_track.configuration import ROOT
from vision_track.lifecycle import StreamState, can_transition, validate_transition
from vision_track.logging_utils import configure_logging, mask_sensitive


def test_state_transitions() -> None:
    assert can_transition(StreamState.CREATED, StreamState.PREPARING)
    assert can_transition(StreamState.PREPARING, StreamState.CONNECTING)
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


def test_test_logger_uses_temporary_log_path(tmp_path) -> None:
    log_path = tmp_path / "app_errors.log"
    logger = configure_logging(log_path)

    handler_paths = [
        handler.baseFilename
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
    ]

    assert str(log_path.resolve()) in handler_paths
    assert str((ROOT / "logs" / "app_errors.log").resolve()) not in handler_paths
