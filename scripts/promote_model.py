from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.baseline import environment_payload, file_sha256, utc_timestamp
from vision_track.configuration import load_config
from vision_track.device import select_device


ModelVerifier = Callable[[Path], dict]


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _normalized_names(names: object) -> dict[int, str]:
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise RuntimeError(f"Checkpoint exposes unsupported class names: {names!r}")


def _synthetic_verification_frame(image_size: int) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("Verification image size must be positive")
    frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    frame[:, :, :] = (31, 63, 95)
    inset = max(1, image_size // 4)
    far_edge = image_size - inset
    if inset < far_edge:
        frame[inset:far_edge, inset:far_edge, :] = (191, 127, 63)
    return frame


def _tensor_numpy(value: object) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_value = getattr(value, "numpy", None)
    if callable(numpy_value):
        value = numpy_value()
    return np.asarray(value)


def _inference_smoke(
    model: object,
    *,
    image_size: int,
    confidence: float,
    iou: float,
    person_class_id: int,
    device: str,
) -> dict:
    frame = _synthetic_verification_frame(image_size)
    started = time.perf_counter()
    predictions = list(
        model.predict(
            source=frame,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            classes=[person_class_id],
            device=device,
            verbose=False,
        )
    )
    wall_time_ms = (time.perf_counter() - started) * 1000.0
    if len(predictions) != 1:
        raise RuntimeError(
            f"Checkpoint inference must return exactly one result, got {len(predictions)}"
        )
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise RuntimeError("Checkpoint inference result does not expose detection boxes")

    xyxy_value = getattr(boxes, "xyxy", None)
    confidence_value = getattr(boxes, "conf", None)
    class_value = getattr(boxes, "cls", None)
    if xyxy_value is None or confidence_value is None or class_value is None:
        raise RuntimeError("Checkpoint inference boxes are missing xyxy/conf/cls outputs")
    observed_device = str(getattr(xyxy_value, "device", device))
    xyxy = _tensor_numpy(xyxy_value)
    confidences = _tensor_numpy(confidence_value).reshape(-1)
    classes = _tensor_numpy(class_value).reshape(-1)
    if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
        raise RuntimeError(f"Checkpoint inference returned invalid box shape {xyxy.shape}")
    if len(xyxy) != len(confidences) or len(xyxy) != len(classes):
        raise RuntimeError("Checkpoint inference box/confidence/class counts do not match")
    if not (
        np.isfinite(xyxy).all()
        and np.isfinite(confidences).all()
        and np.isfinite(classes).all()
    ):
        raise RuntimeError("Checkpoint inference returned non-finite outputs")
    if len(classes) and not np.all(classes.astype(np.int64) == person_class_id):
        raise RuntimeError("Checkpoint inference returned a class other than person")

    speed = getattr(result, "speed", None)
    speed_ms = {
        str(stage): float(value)
        for stage, value in (speed.items() if isinstance(speed, dict) else ())
        if value is not None and np.isfinite(float(value))
    }
    return {
        "status": "passed",
        "input": {
            "kind": "deterministic_synthetic_frame",
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "color_space": "BGR",
            "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
        },
        "parameters": {
            "image_size": image_size,
            "confidence": confidence,
            "iou": iou,
            "classes": [person_class_id],
            "requested_device": device,
        },
        "observed_output_device": observed_device,
        "result_count": 1,
        "detection_count": int(len(xyxy)),
        "outputs_finite": True,
        "wall_time_ms": round(wall_time_ms, 3),
        "ultralytics_speed_ms": speed_ms,
    }


def verify_yolo_checkpoint(
    checkpoint: Path,
    *,
    loader: Callable[..., object] | None = None,
    image_size: int = 640,
    confidence: float = 0.35,
    iou: float = 0.50,
    person_class_id: int = 0,
    device: str = "cpu",
) -> dict:
    if loader is None:
        from ultralytics import YOLO

        loader = YOLO
    model = loader(str(checkpoint), task="detect")
    task = str(getattr(model, "task", ""))
    names = _normalized_names(getattr(model, "names", None))
    if task != "detect":
        raise RuntimeError(f"Checkpoint task must be detect, got {task!r}")
    person_name = names.get(person_class_id)
    if person_name != "person":
        raise RuntimeError(
            f"Runtime checkpoint must contain class {person_class_id}=person, "
            f"got {names!r}"
        )
    inference_smoke = _inference_smoke(
        model,
        image_size=image_size,
        confidence=confidence,
        iou=iou,
        person_class_id=person_class_id,
        device=device,
    )
    return {
        "status": "passed",
        "task": task,
        "names": {str(index): name for index, name in names.items()},
        "class_count": len(names),
        "person_class_id": person_class_id,
        "person_class_name": person_name,
        "multiclass_checkpoint": len(names) > 1,
        "inference_smoke": inference_smoke,
    }


def promote_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str,
    verify_model: ModelVerifier = verify_yolo_checkpoint,
) -> dict:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("Expected SHA-256 must contain exactly 64 hexadecimal characters")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == destination_path:
        raise ValueError("Promotion source and destination must be different files")
    if source_path.suffix.lower() != destination_path.suffix.lower():
        raise ValueError("Promotion source and destination must use the same model format")

    source_sha256 = file_sha256(source_path)
    if source_sha256 != expected:
        raise RuntimeError(
            f"Source SHA-256 mismatch: expected {expected}, got {source_sha256}"
        )
    source_verification = verify_model(source_path)
    previous_destination = (
        {
            "existed": True,
            "sha256": file_sha256(destination_path),
            "bytes": destination_path.stat().st_size,
        }
        if destination_path.is_file()
        else {"existed": False, "sha256": None, "bytes": None}
    )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.stem}.promotion-",
        suffix=destination_path.suffix,
        dir=destination_path.parent,
    )
    staged_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as staged_handle:
            with source_path.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, staged_handle, length=1024 * 1024)
                staged_handle.flush()
                os.fsync(staged_handle.fileno())
        staged_sha256 = file_sha256(staged_path)
        if staged_sha256 != expected:
            raise RuntimeError(
                f"Staged SHA-256 mismatch: expected {expected}, got {staged_sha256}"
            )
        staged_verification = verify_model(staged_path)
        if staged_verification.get("status") != "passed" or (
            staged_verification.get("inference_smoke", {}).get("status") != "passed"
        ):
            raise RuntimeError(
                "Staged checkpoint verification and inference smoke must both pass"
            )
        os.replace(staged_path, destination_path)
        destination_sha256 = file_sha256(destination_path)
        if destination_sha256 != expected:
            raise RuntimeError(
                "Published destination SHA-256 differs from the verified staged file"
            )
    finally:
        if staged_path.exists():
            staged_path.unlink()

    return {
        "source": _display_path(source_path),
        "destination": _display_path(destination_path),
        "expected_sha256": expected,
        "source_sha256": source_sha256,
        "destination_sha256": destination_sha256,
        "bytes": destination_path.stat().st_size,
        "previous_destination": previous_destination,
        "source_verification": source_verification,
        "staged_verification": staged_verification,
        "publication": {
            "atomic": True,
            "method": "verified same-directory temporary file followed by os.replace",
            "staged_inference_verified_before_replace": True,
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    target = path.resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=target.suffix,
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict:
    started_at = utc_timestamp()
    config = load_config(args.config)
    selected_device = select_device(
        force=None if args.verification_device == "auto" else args.verification_device
    )
    report_path = (
        args.report.resolve()
        if args.report is not None
        else ROOT / "reports" / "model_promotions" / f"{args.run_id}.json"
    )
    if report_path.exists():
        raise FileExistsError(report_path)
    promotion = promote_checkpoint(
        args.source,
        args.destination,
        expected_sha256=args.expected_sha256,
        verify_model=lambda checkpoint: verify_yolo_checkpoint(
            checkpoint,
            image_size=config.model.image_size,
            confidence=config.model.confidence,
            iou=config.model.iou,
            person_class_id=config.model.person_class_id,
            device=selected_device.torch_device,
        ),
    )
    report = {
        "schema_version": 2,
        "status": "complete",
        "run_id": args.run_id,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "promotion": promotion,
        "verification_config": {
            "project_config": _display_path(args.config),
            "image_size": config.model.image_size,
            "confidence": config.model.confidence,
            "iou": config.model.iou,
            "person_class_id": config.model.person_class_id,
            "device_request": args.verification_device,
            "selected_device": {
                "kind": selected_device.kind,
                "torch_device": selected_device.torch_device,
                "name": selected_device.name,
                "backend": selected_device.backend,
            },
        },
        "environment": environment_payload(
            ("vision-track", "torch", "ultralytics", "numpy")
        ),
        "report": _display_path(report_path),
    }
    _write_json_atomic(report_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicitly verify and atomically promote a selected model"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument(
        "--verification-device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--run-id",
        default=f"promotion_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception:
        report_path = (
            args.report.resolve()
            if args.report is not None
            else ROOT / "reports" / "model_promotions" / f"{args.run_id}.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        error_path = report_path.with_suffix(report_path.suffix + ".error.log")
        error = traceback.format_exc()
        error_path.write_text(error, encoding="utf-8", newline="\n")
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
