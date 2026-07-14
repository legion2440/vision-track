from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.baseline import environment_payload, file_sha256, utc_timestamp


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


def verify_yolo_checkpoint(
    checkpoint: Path,
    *,
    loader: Callable[..., object] | None = None,
) -> dict:
    if loader is None:
        from ultralytics import YOLO

        loader = YOLO
    model = loader(str(checkpoint), task="detect")
    task = str(getattr(model, "task", ""))
    names = _normalized_names(getattr(model, "names", None))
    if task != "detect":
        raise RuntimeError(f"Checkpoint task must be detect, got {task!r}")
    if names != {0: "person"}:
        raise RuntimeError(
            "Runtime checkpoint must contain exactly class 0=person, "
            f"got {names!r}"
        )
    return {
        "status": "passed",
        "task": task,
        "names": {str(index): name for index, name in names.items()},
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
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "run_id": args.run_id,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "promotion": promotion,
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
