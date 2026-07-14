from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import promote_model
from scripts.promote_model import promote_checkpoint, verify_yolo_checkpoint
from vision_track.baseline import file_sha256


def test_promotion_cli_requires_source_destination_and_sha() -> None:
    parser = promote_model.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--source", "selected.pt", "--expected-sha256", "a" * 64])

    args = parser.parse_args(
        [
            "--source",
            "selected.pt",
            "--destination",
            "runtime.pt",
            "--expected-sha256",
            "a" * 64,
        ]
    )
    assert args.source == Path("selected.pt")
    assert args.destination == Path("runtime.pt")


def test_verify_yolo_checkpoint_requires_single_person_detection_class(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[tuple[str, str]] = []

    def loader(path: str, *, task: str):
        calls.append((path, task))
        return SimpleNamespace(task="detect", names={0: "person"})

    verification = verify_yolo_checkpoint(checkpoint, loader=loader)

    assert verification == {
        "status": "passed",
        "task": "detect",
        "names": {"0": "person"},
    }
    assert calls == [(str(checkpoint), "detect")]

    with pytest.raises(RuntimeError, match="exactly class 0=person"):
        verify_yolo_checkpoint(
            checkpoint,
            loader=lambda *_args, **_kwargs: SimpleNamespace(
                task="detect", names={0: "person", 1: "car"}
            ),
        )


def test_promote_checkpoint_verifies_and_atomically_replaces_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.pt"
    destination = tmp_path / "runtime.pt"
    source.write_bytes(b"selected-model")
    destination.write_bytes(b"previous-model")
    expected = file_sha256(source)
    verified_paths: list[Path] = []

    def verifier(path: Path) -> dict:
        assert path.read_bytes() == b"selected-model"
        verified_paths.append(path)
        return {"status": "passed", "task": "detect", "names": {"0": "person"}}

    result = promote_checkpoint(
        source,
        destination,
        expected_sha256=expected,
        verify_model=verifier,
    )

    assert destination.read_bytes() == b"selected-model"
    assert result["destination_sha256"] == expected
    assert result["previous_destination"]["sha256"] != expected
    assert result["publication"]["atomic"] is True
    assert verified_paths[0] == source
    assert verified_paths[1] != destination
    assert len(verified_paths) == 2
    assert not list(tmp_path.glob(".runtime.promotion-*.pt"))


def test_promote_checkpoint_sha_failure_preserves_destination(tmp_path: Path) -> None:
    source = tmp_path / "selected.pt"
    destination = tmp_path / "runtime.pt"
    source.write_bytes(b"selected-model")
    destination.write_bytes(b"previous-model")

    with pytest.raises(RuntimeError, match="Source SHA-256 mismatch"):
        promote_checkpoint(
            source,
            destination,
            expected_sha256="0" * 64,
            verify_model=lambda _path: {"status": "passed"},
        )

    assert destination.read_bytes() == b"previous-model"


def test_promote_checkpoint_staged_verification_failure_preserves_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.pt"
    destination = tmp_path / "runtime.pt"
    source.write_bytes(b"selected-model")
    destination.write_bytes(b"previous-model")
    calls = 0

    def verifier(_path: Path) -> dict:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("staged model rejected")
        return {"status": "passed"}

    with pytest.raises(RuntimeError, match="staged model rejected"):
        promote_checkpoint(
            source,
            destination,
            expected_sha256=file_sha256(source),
            verify_model=verifier,
        )

    assert destination.read_bytes() == b"previous-model"
    assert not list(tmp_path.glob(".runtime.promotion-*.pt"))


def test_promotion_run_writes_separate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "promotion.json"
    promotion = {
        "source": "selected.pt",
        "destination": "runtime.pt",
        "destination_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        promote_model,
        "promote_checkpoint",
        lambda *_args, **_kwargs: promotion,
    )
    monkeypatch.setattr(
        promote_model,
        "environment_payload",
        lambda _packages: {"python": {"version": "test"}},
    )
    args = Namespace(
        source=tmp_path / "selected.pt",
        destination=tmp_path / "runtime.pt",
        expected_sha256="a" * 64,
        report=report_path,
        run_id="promotion_test",
    )

    result = promote_model.run(args)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert result["promotion"] == promotion
    assert persisted == result
    assert persisted["report"] == report_path.relative_to(promote_model.ROOT).as_posix()
