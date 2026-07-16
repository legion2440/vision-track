from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import promote_model
from scripts.promote_model import promote_checkpoint, verify_yolo_checkpoint
from vision_track.baseline import file_sha256


class _FakeTensor:
    def __init__(self, value: np.ndarray, *, device: str = "cpu") -> None:
        self._value = value
        self.device = device

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._value


def _passed_verification() -> dict:
    return {
        "status": "passed",
        "task": "detect",
        "names": {"0": "person"},
        "inference_smoke": {"status": "passed"},
    }


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
    assert args.verification_device == "auto"


def test_verify_yolo_checkpoint_accepts_person_class_in_multiclass_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls: list[tuple[str, str]] = []

    predict_calls: list[dict] = []

    class FakeModel:
        task = "detect"
        names = {0: "person"}

        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            boxes = SimpleNamespace(
                xyxy=_FakeTensor(np.array([[1, 2, 10, 20]], dtype=np.float32)),
                conf=_FakeTensor(np.array([0.9], dtype=np.float32)),
                cls=_FakeTensor(np.array([0], dtype=np.float32)),
            )
            return [
                SimpleNamespace(
                    boxes=boxes,
                    speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
                )
            ]

    def loader(path: str, *, task: str):
        calls.append((path, task))
        return FakeModel()

    verification = verify_yolo_checkpoint(checkpoint, loader=loader, image_size=64)

    assert verification["status"] == "passed"
    assert verification["task"] == "detect"
    assert verification["names"] == {"0": "person"}
    assert verification["class_count"] == 1
    assert verification["person_class_id"] == 0
    assert verification["person_class_name"] == "person"
    assert verification["multiclass_checkpoint"] is False
    assert verification["inference_smoke"]["status"] == "passed"
    assert verification["inference_smoke"]["detection_count"] == 1
    assert verification["inference_smoke"]["input"]["shape"] == [64, 64, 3]
    assert verification["inference_smoke"]["outputs_finite"] is True
    assert calls == [(str(checkpoint), "detect")]
    assert len(predict_calls) == 1
    assert predict_calls[0]["source"].shape == (64, 64, 3)
    assert predict_calls[0]["imgsz"] == 64
    assert predict_calls[0]["device"] == "cpu"
    assert predict_calls[0]["classes"] == [0]

    class MulticlassFakeModel(FakeModel):
        names = {0: "person", 1: "bicycle", 2: "car"}

    multiclass = verify_yolo_checkpoint(
        checkpoint,
        loader=lambda *_args, **_kwargs: MulticlassFakeModel(),
        image_size=64,
    )
    assert multiclass["status"] == "passed"
    assert multiclass["class_count"] == 3
    assert multiclass["person_class_id"] == 0
    assert multiclass["person_class_name"] == "person"
    assert multiclass["multiclass_checkpoint"] is True

    with pytest.raises(RuntimeError, match="class 0=person"):
        verify_yolo_checkpoint(
            checkpoint,
            loader=lambda *_args, **_kwargs: SimpleNamespace(
                task="detect", names={0: "vehicle", 1: "person"}
            ),
        )


def test_verify_yolo_checkpoint_rejects_non_finite_inference_output(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    boxes = SimpleNamespace(
        xyxy=_FakeTensor(
            np.array([[1, 2, np.nan, 20]], dtype=np.float32)
        ),
        conf=_FakeTensor(np.array([0.9], dtype=np.float32)),
        cls=_FakeTensor(np.array([0], dtype=np.float32)),
    )
    model = SimpleNamespace(
        task="detect",
        names={0: "person"},
        predict=lambda **_kwargs: [SimpleNamespace(boxes=boxes, speed={})],
    )

    with pytest.raises(RuntimeError, match="non-finite outputs"):
        verify_yolo_checkpoint(
            checkpoint,
            loader=lambda *_args, **_kwargs: model,
            image_size=64,
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
        return _passed_verification()

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
    assert result["publication"]["staged_inference_verified_before_replace"] is True
    assert result["staged_verification"]["inference_smoke"]["status"] == "passed"
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
        return _passed_verification()

    with pytest.raises(RuntimeError, match="staged model rejected"):
        promote_checkpoint(
            source,
            destination,
            expected_sha256=file_sha256(source),
            verify_model=verifier,
        )

    assert destination.read_bytes() == b"previous-model"
    assert not list(tmp_path.glob(".runtime.promotion-*.pt"))


def test_promote_checkpoint_rejects_missing_staged_inference_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "selected.pt"
    destination = tmp_path / "runtime.pt"
    source.write_bytes(b"selected-model")
    destination.write_bytes(b"previous-model")

    with pytest.raises(RuntimeError, match="inference smoke must both pass"):
        promote_checkpoint(
            source,
            destination,
            expected_sha256=file_sha256(source),
            verify_model=lambda _path: {
                "status": "passed",
                "task": "detect",
                "names": {"0": "person"},
            },
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
    monkeypatch.setattr(
        promote_model,
        "select_device",
        lambda force=None: SimpleNamespace(
            kind="cuda",
            torch_device="0",
            name="Test GPU",
            backend="PyTorch CUDA",
        ),
    )
    args = Namespace(
        source=tmp_path / "selected.pt",
        destination=tmp_path / "runtime.pt",
        expected_sha256="a" * 64,
        config=promote_model.ROOT / "configs" / "app.yaml",
        verification_device="auto",
        report=report_path,
        run_id="promotion_test",
    )

    result = promote_model.run(args)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["status"] == "complete"
    assert result["schema_version"] == 2
    assert result["promotion"] == promotion
    assert result["verification_config"]["image_size"] == 640
    assert result["verification_config"]["device_request"] == "auto"
    assert result["verification_config"]["selected_device"]["kind"] == "cuda"
    assert persisted == result
    assert persisted["report"] == report_path.relative_to(promote_model.ROOT).as_posix()
