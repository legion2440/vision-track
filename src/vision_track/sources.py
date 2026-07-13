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
    WEBCAM = "webcam"


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
        elif scheme == "webcam":
            parsed = urlsplit(value)
            device = parsed.netloc
            if (
                not device.isascii()
                or not device.isdecimal()
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Webcam source must use webcam://<non-negative index>")
            value = f"webcam://{int(device)}"
            source_type = SourceType.WEBCAM
        else:
            source_type = SourceType.LOCAL
        if display_name is None:
            if source_type is SourceType.LOCAL:
                display_name = Path(value).name
            elif source_type is SourceType.WEBCAM:
                display_name = f"Camera {urlsplit(value).netloc}"
            else:
                display_name = mask_sensitive(value)
        return cls(value, source_type, display_name)

    @classmethod
    def webcam(cls, device_index: int) -> "VideoSource":
        if isinstance(device_index, bool) or not isinstance(device_index, int):
            raise TypeError("Webcam device index must be an integer")
        if device_index < 0:
            raise ValueError("Webcam device index must be non-negative")
        return cls.from_uri(f"webcam://{device_index}")

    @property
    def is_remote(self) -> bool:
        return self.source_type in {SourceType.HTTP, SourceType.RTSP}

    @property
    def is_reconnectable(self) -> bool:
        return self.is_remote or self.source_type is SourceType.WEBCAM

    @property
    def webcam_index(self) -> int:
        if self.source_type is not SourceType.WEBCAM:
            raise ValueError("Video source is not a webcam")
        return int(urlsplit(self.uri).netloc)

    @property
    def safe_uri(self) -> str:
        return mask_sensitive(self.uri)


def new_stream_id(prefix: str = "stream") -> str:
    return f"{prefix}-{uuid4().hex[:8]}"
