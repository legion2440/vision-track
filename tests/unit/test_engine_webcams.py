from __future__ import annotations

import pytest

from vision_track.sources import SourceType, VideoSource


def test_engine_rejects_duplicate_webcam_in_same_session(fake_engine) -> None:
    first = fake_engine.add_stream(VideoSource.webcam(0))

    with pytest.raises(ValueError, match="Camera 0 is already added"):
        fake_engine.add_stream("webcam://00")

    second = fake_engine.add_stream(VideoSource.webcam(1))
    assert fake_engine.get(first).source.source_type is SourceType.WEBCAM
    assert fake_engine.get(second).source.webcam_index == 1


def test_replace_source_rejects_another_streams_webcam_before_stop(
    fake_engine,
    synthetic_video,
) -> None:
    camera = fake_engine.add_stream(VideoSource.webcam(0))
    local = fake_engine.add_stream(str(synthetic_video))
    local_context = fake_engine.get(local)
    original_generation = local_context.runtime_generation

    with pytest.raises(ValueError, match="Camera 0 is already added"):
        fake_engine.replace_source(local, "webcam://0")

    assert local_context.source.uri == str(synthetic_video)
    assert local_context.runtime_generation == original_generation

    fake_engine.replace_source(camera, "webcam://0")
    assert fake_engine.get(camera).source.webcam_index == 0
