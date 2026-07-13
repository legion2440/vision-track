from __future__ import annotations

import pytest

from vision_track.sources import SourceType, VideoSource


def test_webcam_source_is_canonical_and_reconnectable() -> None:
    source = VideoSource.from_uri("  webcam://007/  ")

    assert source.uri == "webcam://7"
    assert source.source_type is SourceType.WEBCAM
    assert source.display_name == "Camera 7"
    assert source.webcam_index == 7
    assert source.is_reconnectable
    assert not source.is_remote


def test_webcam_factory_validates_device_index() -> None:
    assert VideoSource.webcam(0) == VideoSource.from_uri("webcam://0")

    with pytest.raises(TypeError):
        VideoSource.webcam(True)
    with pytest.raises(TypeError):
        VideoSource.webcam(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VideoSource.webcam(-1)


@pytest.mark.parametrize(
    "uri",
    [
        "webcam://",
        "webcam://-1",
        "webcam://camera",
        "webcam://0/path",
        "webcam://0?option=value",
        "webcam://0#fragment",
        "webcam:0",
    ],
)
def test_invalid_webcam_uri_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="webcam://"):
        VideoSource.from_uri(uri)


def test_only_live_sources_are_reconnectable() -> None:
    assert VideoSource.from_uri("https://example.com/live").is_reconnectable
    assert VideoSource.from_uri("rtsp://example.com/live").is_reconnectable
    assert not VideoSource.from_uri("video.mp4").is_reconnectable


def test_webcam_index_rejects_non_webcam_source() -> None:
    with pytest.raises(ValueError, match="not a webcam"):
        _ = VideoSource.from_uri("video.mp4").webcam_index
