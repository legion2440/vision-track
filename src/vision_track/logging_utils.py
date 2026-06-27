from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


_TOKEN_PATTERN = re.compile(
    r"(?i)(token|api[_-]?key|password|passwd|secret)=([^&\s]+)"
)


def mask_sensitive(value: str) -> str:
    if not value:
        return value
    masked = _TOKEN_PATTERN.sub(r"\1=***", value)
    try:
        parts = urlsplit(masked)
        if parts.scheme and parts.hostname and (parts.username or parts.password):
            host = parts.hostname
            if parts.port:
                host = f"{host}:{parts.port}"
            masked = urlunsplit((parts.scheme, f"***:***@{host}", parts.path, parts.query, parts.fragment))
    except ValueError:
        pass
    return masked


def configure_logging(log_path: str | Path) -> logging.Logger:
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vision_track")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    has_requested_handler = False
    for handler in list(logger.handlers):
        if not isinstance(handler, RotatingFileHandler):
            continue
        handler_path = Path(handler.baseFilename).resolve()
        if handler_path == path:
            has_requested_handler = True
            continue
        logger.removeHandler(handler)
        handler.close()
    if not has_requested_handler:
        handler = RotatingFileHandler(
            path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | stream_id=%(stream_id)s | source_type=%(source_type)s "
                "| state=%(state)s | %(levelname)s | %(message)s"
            )
        )
        logger.addHandler(handler)
    return logger


def log_stream_error(
    logger: logging.Logger,
    *,
    stream_id: str,
    source_type: str,
    state: str,
    exc: BaseException,
    unexpected: bool = True,
) -> None:
    extra = {
        "stream_id": stream_id,
        "source_type": source_type,
        "state": state,
    }
    message = f"{type(exc).__name__}: {mask_sensitive(str(exc))}"
    logger.error(message, exc_info=unexpected, extra=extra)
