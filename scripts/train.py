from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.baseline import (
    checkpoint_artifact,
    create_linked_dataset,
    dataset_fingerprint,
    environment_payload,
    evaluator_arguments,
    evaluator_metrics,
    inspect_dataset_contract,
    training_arguments,
    utc_timestamp,
)
from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import select_device


EXPECTED_SPLIT_COUNTS = {"train": 64115, "val": 1346, "test": 1347}


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _split_images(root: Path, split: str) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in (root / "images" / split).iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _git_ignored(path: Path) -> bool:
    import subprocess

    return (
        subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def run(args: argparse.Namespace) -> dict:
    if not args.confirm_full_run:
        raise RuntimeError(
            "Full A training requires --confirm-full-run after the baseline readiness review"
        )
    from ultralytics import YOLO

    config = load_config(args.config)
    selected = select_device(force=None if args.device == "auto" else args.device)
    dataset_yaml = resolve_project_path(config.raw["training"]["dataset_yaml"])
    contract = inspect_dataset_contract(
        dataset_yaml,
        expected_counts=EXPECTED_SPLIT_COUNTS,
    )
    control_root = Path(contract["root"])
    before = dataset_fingerprint(control_root)
    run_root = ROOT / "models" / "training_runs" / "full_a" / args.run_id
    report_root = ROOT / "reports" / "training_runs" / args.run_id
    if run_root.exists() or report_root.exists():
        raise FileExistsError(f"Run ID already exists: {args.run_id}")
    run_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    linked = create_linked_dataset(
        control_root,
        run_root / "dataset",
        {
            "train": _split_images(control_root, "train"),
            "val": _split_images(control_root, "val"),
        },
    )
    _write_json(
        run_root / "linked_dataset_summary.json",
        {
            "dataset_yaml": linked["dataset_yaml"],
            "split_counts": linked["split_counts"],
            "link_methods": linked["link_methods"],
            "contains_test": False,
        },
    )
    effective_args = training_arguments(
        config,
        data=linked["dataset_yaml"],
        device=selected.torch_device,
        project=run_root / "ultralytics",
        name="a_full",
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    model_source = ROOT / config.model.pretrained
    source = model_source if model_source.is_file() else config.model.pretrained
    model = YOLO(str(source), task="detect")
    started_at = utc_timestamp()
    wall_started = time.perf_counter()
    results = model.train(**effective_args)
    wall_seconds = time.perf_counter() - wall_started
    save_dir = Path(results.save_dir)
    checkpoints = {
        "best": save_dir / "weights" / "best.pt",
        "last": save_dir / "weights" / "last.pt",
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise RuntimeError(f"Ultralytics did not produce best/last: {checkpoints}")

    inference_source = str(
        sorted((Path(linked["root"]) / "images" / "val").glob("*"))[0].absolute()
    )
    reload_verification: dict[str, dict] = {}
    for checkpoint_name, checkpoint in checkpoints.items():
        if not _git_ignored(checkpoint):
            raise RuntimeError(f"Checkpoint is not ignored by Git: {checkpoint}")
        reloaded = YOLO(str(checkpoint), task="detect")
        predictions = reloaded.predict(
            source=inference_source,
            imgsz=config.model.image_size,
            conf=config.model.confidence,
            iou=config.model.iou,
            classes=[config.model.person_class_id],
            device=selected.torch_device,
            verbose=False,
        )
        validation = reloaded.val(
            **evaluator_arguments(
                config,
                data=linked["dataset_yaml"],
                device=selected.torch_device,
                project=run_root / "ultralytics",
                name=f"{checkpoint_name}_reload_val",
                plots=False,
            )
        )
        reload_verification[checkpoint_name] = {
            "status": "passed",
            "checkpoint": checkpoint_artifact(checkpoint, ROOT),
            "git_ignored": True,
            "inference_images": len(predictions),
            "inference_detections": sum(
                len(item.boxes) if item.boxes is not None else 0
                for item in predictions
            ),
            "validation_split": "val",
            "validation_metrics": evaluator_metrics(validation),
        }

    destination = resolve_project_path(config.model.checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not _git_ignored(destination):
        raise RuntimeError(f"Runtime checkpoint destination is not ignored: {destination}")
    shutil.copy2(checkpoints["best"], destination)
    after = dataset_fingerprint(control_root)
    if before != after:
        raise RuntimeError("Control coco_person changed during full A training")

    args_yaml = save_dir / "args.yaml"
    report = {
        "status": "complete",
        "run_id": args.run_id,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "scope": {
            "full_a_training": True,
            "test_used": False,
            "validation_split": "val",
            "confidence_sweep_performed": False,
        },
        "pretrained_model": config.model.pretrained,
        "dataset": contract,
        "dataset_fingerprint_before": before,
        "dataset_fingerprint_after": after,
        "dataset_unchanged": True,
        "device": {
            "kind": selected.kind,
            "torch_device": selected.torch_device,
            "name": selected.name,
        },
        "environment": environment_payload(
            ("vision-track", "torch", "torchvision", "ultralytics", "numpy")
        ),
        "seed": config.seed,
        "effective_arguments": effective_args,
        "ultralytics_effective_arguments": (
            yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
            if args_yaml.is_file()
            else None
        ),
        "training_wall_seconds": wall_seconds,
        "training_metrics": evaluator_metrics(results),
        "checkpoints": reload_verification,
        "runtime_best_checkpoint": checkpoint_artifact(destination, ROOT),
        "ultralytics_save_dir": _display_path(save_dir),
    }
    _write_json(report_root / "training_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the approved full A fine-tuning with val-only model selection"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--run-id",
        default=f"a_full_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
    )
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required acknowledgement that the multi-hour full A run is approved",
    )
    args = parser.parse_args()
    report_root = ROOT / "reports" / "training_runs" / args.run_id
    try:
        report = run(args)
    except Exception:
        report_root.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (report_root / "error.log").write_text(
            error,
            encoding="utf-8",
            newline="\n",
        )
        print(error, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_id": report["run_id"],
                "report": _display_path(report_root / "training_report.json"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
