from __future__ import annotations

from enum import Enum


class StreamState(str, Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    CONNECTING = "CONNECTING"
    ACTIVE = "ACTIVE"
    EOF = "EOF"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


_ALLOWED_TRANSITIONS = {
    StreamState.CREATED: {
        StreamState.PREPARING,
        StreamState.CONNECTING,
        StreamState.STOPPED,
    },
    StreamState.PREPARING: {
        StreamState.CONNECTING,
        StreamState.FAILED,
        StreamState.STOPPED,
    },
    StreamState.CONNECTING: {
        StreamState.ACTIVE,
        StreamState.RECONNECTING,
        StreamState.FAILED,
        StreamState.STOPPED,
    },
    StreamState.ACTIVE: {
        StreamState.EOF,
        StreamState.RECONNECTING,
        StreamState.FAILED,
        StreamState.STOPPED,
    },
    StreamState.EOF: {StreamState.CONNECTING, StreamState.STOPPED},
    StreamState.RECONNECTING: {
        StreamState.ACTIVE,
        StreamState.FAILED,
        StreamState.STOPPED,
    },
    StreamState.FAILED: {StreamState.CONNECTING, StreamState.STOPPED},
    StreamState.STOPPED: {StreamState.PREPARING, StreamState.CONNECTING},
}


def can_transition(current: StreamState, target: StreamState) -> bool:
    return target == current or target in _ALLOWED_TRANSITIONS[current]


def validate_transition(current: StreamState, target: StreamState) -> None:
    if not can_transition(current, target):
        raise ValueError(f"Invalid stream transition: {current.value} -> {target.value}")
