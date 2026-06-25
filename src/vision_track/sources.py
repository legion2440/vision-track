from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from .logging_utils import mask_sensitive


class SourceType(str, Enum):
    LOCAL = "local"
    HTTP = "http"
    RTSP = "rtsp"


@dataclass(frozen=True)
class VideoSource:
    uri: str
    source_type: SourceType
    display_name: str

    @classmethod
    def from_uri(cls, uri: str, display_name: str | None = None) -> "VideoSource":
        value = uri.strip()
        if not value:
            raise ValueError("Video source cannot be empty")
        scheme = urlsplit(value).scheme.lower()
        if scheme in {"http", "https"}:
            source_type = SourceType.HTTP
        elif scheme in {"rtsp", "rtsps"}:
            source_type = SourceType.RTSP
        else:
            source_type = SourceType.LOCAL
        if display_name is None:
            display_name = (
                Path(value).name
                if source_type is SourceType.LOCAL
                else mask_sensitive(value)
            )
        return cls(value, source_type, display_name)

    @property
    def is_remote(self) -> bool:
        return self.source_type in {SourceType.HTTP, SourceType.RTSP}

    @property
    def safe_uri(self) -> str:
        return mask_sensitive(self.uri)


def new_stream_id(prefix: str = "stream") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"

