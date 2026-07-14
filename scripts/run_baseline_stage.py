from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
    EVALUATOR_CONFIDENCE,
    EVALUATOR_NMS_IOU,
    OPERATIONAL_MATCH_IOU,
    SMOKE_TRAIN_IMAGES,
    SMOKE_VAL_IMAGES,
    checkpoint_artifact,
    create_linked_dataset,
    dataset_fingerprint,
    deterministic_smoke_selection,
    environment_payload,
    evaluator_arguments,
    evaluator_metrics,
    file_sha256,
    inspect_dataset_contract,
    load_contamination_evidence,
    match_detection_counts,
    pytorch_runtime_device,
    read_yolo_xyxy,
    selected_object_count,
    training_arguments,
    utc_timestamp,
)
from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import select_device


EXPECTED_SPLIT_COUNTS = {"train": 64115, "val": 1346, "test": 1347}
ENVIRONMENT_PACKAGES = (
    "vision-track",
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "opencv-python",
    "PyYAML",
)


def _display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _stage(message: str) -> None:
    print(f"[baseline] {message}", flush=True)


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


def _cuda_smoke(selected_device: Any) -> dict:
    import torch

    started = time.perf_counter()
    device = pytorch_runtime_device(selected_device)
    left = torch.randn((512, 512), device=device)
    right = torch.randn((512, 512), device=device)
    result = left @ right
    if selected_device.kind == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "status": "passed",
        "device_kind": selected_device.kind,
        "torch_device": device,
        "device_name": selected_device.name,
        "matrix_shape": [512, 512],
        "elapsed_ms": elapsed_ms,
        "finite": bool(torch.isfinite(result).all().item()),
    }


def _pretrained_path(model_name: str) -> Path | str:
    local = ROOT / model_name
    return local.resolve() if local.is_file() else model_name


def _all_split_images(dataset_root: Path, split: str) -> list[str]:
    return sorted(
        path.relative_to(dataset_root).as_posix()
        for path in (dataset_root / "images" / split).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _run_operational_metrics(
    model: Any,
    *,
    config: Any,
    linked_root: Path,
    selected_device: Any,
) -> tuple[dict, list[dict]]:
    image_directory = linked_root / "images" / "val"
    if not any(path.is_file() for path in image_directory.iterdir()):
        raise RuntimeError("Operational val image list is empty")
    predictions = model.predict(
        source=str(image_directory.absolute()),
        stream=True,
        batch=int(config.raw["training"]["batch_size"]),
        imgsz=config.model.image_size,
        conf=config.model.confidence,
        iou=config.model.iou,
        classes=[config.model.person_class_id],
        device=selected_device.torch_device,
        verbose=False,
    )
    totals = {"tp": 0, "fp": 0, "fn": 0, "detections": 0, "objects": 0}
    speed_totals: dict[str, float] = {}
    per_image: list[dict] = []
    wall_started = time.perf_counter()
    for result in predictions:
        image_name = Path(result.path).name
        height, width = map(int, result.orig_shape)
        label_path = linked_root / "labels" / "val" / f"{Path(image_name).stem}.txt"
        expected = read_yolo_xyxy(label_path, width, height)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            predicted = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
        else:
            predicted = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
        tp, fp, fn = match_detection_counts(
            predicted,
            confidences,
            expected,
            iou_threshold=OPERATIONAL_MATCH_IOU,
        )
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["detections"] += len(predicted)
        totals["objects"] += len(expected)
        for key, value in result.speed.items():
            speed_totals[key] = speed_totals.get(key, 0.0) + float(value)
        per_image.append(
            {
                "image": f"images/val/{image_name}",
                "ground_truth_objects": len(expected),
                "detections": len(predicted),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "preprocess_ms": float(result.speed.get("preprocess", 0.0)),
                "inference_ms": float(result.speed.get("inference", 0.0)),
                "postprocess_ms": float(result.speed.get("postprocess", 0.0)),
            }
        )
    wall_seconds = time.perf_counter() - wall_started
    image_count = len(per_image)
    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    metrics = {
        "semantics": (
            "fixed project runtime confidence; detections matched in descending "
            "confidence order to the best unmatched ground truth at IoU>=0.50"
        ),
        "matching_algorithm": "confidence_descending_best_unmatched_gt",
        "matching_algorithm_version": 2,
        "computed_at": utc_timestamp(),
        "split": "val",
        "confidence": config.model.confidence,
        "nms_iou": config.model.iou,
        "matching_iou": OPERATIONAL_MATCH_IOU,
        "images": image_count,
        "objects": totals["objects"],
        "detections": totals["detections"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "false_negatives": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "false_positives_per_image": totals["fp"] / image_count,
        "wall_seconds": wall_seconds,
        "wall_ms_per_image": wall_seconds * 1000 / image_count,
        "speed_ms_per_image": {
            key: value / image_count for key, value in speed_totals.items()
        },
    }
    return metrics, per_image


def _write_operational_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_ignored(path: Path) -> bool:
    result = __import__("subprocess").run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=ROOT,
        check=False,
    )
    return result.returncode == 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _provisional_full_estimate(
    *,
    smoke_wall_seconds: float,
    smoke_epoch_seconds: float,
    smoke_run_bytes: int,
    checkpoint_bytes: int,
    full_epochs: int,
) -> dict:
    linear_seconds = (
        smoke_epoch_seconds
        * (EXPECTED_SPLIT_COUNTS["train"] / SMOKE_TRAIN_IMAGES)
        * full_epochs
    )
    return {
        "method": (
            "Linear extrapolation from Ultralytics results.csv epoch time for the "
            "256-image smoke. The range is wide because fixed startup and validation "
            "costs dominate a tiny run."
        ),
        "smoke_training_wall_seconds": smoke_wall_seconds,
        "smoke_ultralytics_epoch_seconds": smoke_epoch_seconds,
        "linear_extrapolation_hours": linear_seconds / 3600,
        "estimated_hours_range": [linear_seconds / 3600 * 0.20, linear_seconds / 3600],
        "smoke_run_bytes": smoke_run_bytes,
        "estimated_disk_bytes_range": [
            max(2 * checkpoint_bytes, smoke_run_bytes),
            max(512 * 1024 * 1024, smoke_run_bytes * 3),
        ],
        "confidence": "low until the first full-data epoch is authorized and measured",
    }


def _ultralytics_epoch_seconds(save_dir: Path) -> float:
    results_path = save_dir / "results.csv"
    with results_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[-1].get("time"):
        raise RuntimeError(f"Ultralytics epoch timing is missing: {results_path}")
    return float(rows[-1]["time"])


def _write_metrics_csv(path: Path, summary: dict) -> None:
    rows: list[dict] = []
    if summary.get("z0"):
        pretrained = summary["model"]["pretrained"]
        evaluator = summary["z0"]["evaluator_metrics"]
        rows.append(
            {
                "run": "Z0",
                "metric_semantics": "ultralytics_evaluator",
                "checkpoint": pretrained["path"],
                "checkpoint_sha256": pretrained["sha256"],
                "config_artifact": summary["artifacts"]["tracked_report_root"]
                + "/effective_config.json",
                "split": "val",
                "confidence": EVALUATOR_CONFIDENCE,
                "images": summary["z0"]["images"],
                "objects": summary["z0"]["objects"],
                "precision": evaluator["precision"],
                "recall": evaluator["recall"],
                "mAP50": evaluator["mAP50"],
                "mAP50_95": evaluator["mAP50_95"],
                "detections": "",
                "false_positives": "",
                "false_negatives": "",
                "false_positives_per_image": "",
                "inference_ms_per_image": evaluator["speed_ms_per_image"].get(
                    "inference", ""
                ),
                "wall_ms_per_image": "",
            }
        )
        operational = summary["z0"]["operational_metrics"]
        rows.append(
            {
                "run": "Z0",
                "metric_semantics": "project_runtime_threshold",
                "checkpoint": pretrained["path"],
                "checkpoint_sha256": pretrained["sha256"],
                "config_artifact": summary["artifacts"]["tracked_report_root"]
                + "/effective_config.json",
                "split": "val",
                "confidence": operational["confidence"],
                "images": operational["images"],
                "objects": operational["objects"],
                "precision": operational["precision"],
                "recall": operational["recall"],
                "mAP50": "",
                "mAP50_95": "",
                "detections": operational["detections"],
                "false_positives": operational["false_positives"],
                "false_negatives": operational["false_negatives"],
                "false_positives_per_image": operational["false_positives_per_image"],
                "inference_ms_per_image": operational["speed_ms_per_image"].get(
                    "inference", ""
                ),
                "wall_ms_per_image": operational["wall_ms_per_image"],
            }
        )
    for checkpoint_name, verification in summary.get("a_smoke", {}).get(
        "reload_verification", {}
    ).items():
        metrics = verification["validation_metrics"]
        checkpoint = verification["checkpoint"]
        rows.append(
            {
                "run": f"A-smoke-{checkpoint_name}",
                "metric_semantics": "ultralytics_evaluator",
                "checkpoint": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "config_artifact": summary["artifacts"]["tracked_report_root"]
                + "/effective_config.json",
                "split": "val-smoke-64",
                "confidence": EVALUATOR_CONFIDENCE,
                "images": verification["validation_images"],
                "objects": summary["a_smoke"]["val_objects"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mAP50": metrics["mAP50"],
                "mAP50_95": metrics["mAP50_95"],
                "detections": "",
                "false_positives": "",
                "false_negatives": "",
                "false_positives_per_image": "",
                "inference_ms_per_image": metrics["speed_ms_per_image"].get(
                    "inference", ""
                ),
                "wall_ms_per_image": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict) -> None:
    z0 = summary["z0"]
    operational = z0["operational_metrics"]
    evaluator = z0["evaluator_metrics"]
    smoke = summary["a_smoke"]
    estimate = summary["full_a_plan"]["estimate"]
    crowd = summary["contamination"]["unlabeled_crowd_val"]
    lines = [
        "# Detector baseline Z0/A readiness",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        "Status: **complete**",
        "",
        "## Scope",
        "",
        "- Z0: pretrained `yolo26n.pt`, full control validation split.",
        "- A smoke: one epoch on deterministic 256 train / 64 val linked samples.",
        "- Full A training: **not started**.",
        "- Test split: **not used**.",
        "- Control dataset: unchanged by before/after fingerprint.",
        "",
        "## Z0 standard evaluator",
        "",
        f"- Evaluator confidence: {EVALUATOR_CONFIDENCE}",
        f"- Precision: {evaluator['precision']:.6f}",
        f"- Recall: {evaluator['recall']:.6f}",
        f"- mAP50: {evaluator['mAP50']:.6f}",
        f"- mAP50-95: {evaluator['mAP50_95']:.6f}",
        f"- Inference: {evaluator['speed_ms_per_image']['inference']:.3f} ms/image",
        f"- Images / objects: {z0['images']} / {z0['objects']}",
        "",
        "## Z0 project runtime threshold",
        "",
        f"- Confidence / NMS IoU / match IoU: {operational['confidence']} / "
        f"{operational['nms_iou']} / {operational['matching_iou']}",
        f"- Matching: `{operational['matching_algorithm']}` "
        f"(v{operational['matching_algorithm_version']})",
        f"- Precision: {operational['precision']:.6f}",
        f"- Recall: {operational['recall']:.6f}",
        f"- Detections: {operational['detections']}",
        f"- False positives/image: {operational['false_positives_per_image']:.6f}",
        f"- Wall time: {operational['wall_ms_per_image']:.3f} ms/image",
        "",
        "## A smoke",
        "",
        f"- Training wall time: {smoke['training_wall_seconds']:.2f} s",
        f"- Train / val objects: {smoke['train_objects']} / {smoke['val_objects']}",
        f"- best.pt reload: {smoke['reload_verification']['best']['status']}",
        f"- last.pt reload: {smoke['reload_verification']['last']['status']}",
        "- Both checkpoints remain in the ignored local training-run directory.",
        "",
        "## Full A plan (not executed)",
        "",
        f"- Configured epochs / batch / patience: "
        f"{summary['full_a_plan']['effective_arguments']['epochs']} / "
        f"{summary['full_a_plan']['effective_arguments']['batch']} / "
        f"{summary['full_a_plan']['effective_arguments']['patience']}",
        f"- Provisional time range: {estimate['estimated_hours_range'][0]:.2f}–"
        f"{estimate['estimated_hours_range'][1]:.2f} hours.",
        f"- Provisional disk range: {estimate['estimated_disk_bytes_range'][0] / 2**30:.2f}–"
        f"{estimate['estimated_disk_bytes_range'][1] / 2**30:.2f} GiB.",
        "- Runtime promotion: not performed; it is a separate explicit stage.",
        "",
        "## Known control limitation",
        "",
        "Exact SHA leakage is zero. Five manually reviewed cross-split same-scene "
        "pHash clusters remain in the unchanged control split. Z0/A results should "
        "be interpreted with this small contamination; remediation belongs only to "
        "dataset_v2 or a separate deduplicated materialization.",
        "",
        f"The validation split also contains {crowd['regions']} unlabeled `iscrowd` "
        f"regions across {crowd['images_with_regions']} images. Real crowded people "
        "omitted from YOLO labels can be counted as operational false positives and "
        "can suppress measured recall. Crowd policy remains deferred to dataset_v2.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    started_at = utc_timestamp()
    _stage(f"starting run {args.run_id}")
    config = load_config(args.config)
    selected = select_device(force=None if args.device == "auto" else args.device)
    run_id = args.run_id
    local_root = ROOT / "models" / "training_runs" / "baseline" / run_id
    report_root = ROOT / "reports" / "baseline_runs" / run_id
    if local_root.exists() or report_root.exists():
        raise FileExistsError(f"Run ID already exists: {run_id}")
    local_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    _stage("validating control dataset contract")
    dataset_yaml = resolve_project_path(config.raw["training"]["dataset_yaml"])
    contract = inspect_dataset_contract(
        dataset_yaml,
        expected_counts=EXPECTED_SPLIT_COUNTS,
    )
    control_root = Path(contract["root"])
    _stage("fingerprinting control dataset before evaluation")
    before_fingerprint = dataset_fingerprint(control_root)
    _stage(
        f"control fingerprint complete ({before_fingerprint['file_count']} files)"
    )
    contamination = load_contamination_evidence(ROOT / "reports" / "dataset_audit.json")
    environment = environment_payload(ENVIRONMENT_PACKAGES)
    device_smoke = _cuda_smoke(selected)
    model_source = _pretrained_path(config.model.pretrained)
    pretrained_record = {
        "name": config.model.pretrained,
        "path": _display_path(model_source) if isinstance(model_source, Path) else model_source,
        "sha256": file_sha256(model_source) if isinstance(model_source, Path) else None,
    }
    summary: dict = {
        "schema_version": 2,
        "run_id": run_id,
        "started_at": started_at,
        "status": "running",
        "scope": {
            "z0_full_val": True,
            "operational_metrics_full_val": True,
            "a_smoke_only": True,
            "full_a_training_started": False,
            "test_used": False,
            "confidence_sweep_performed": False,
            "dataset_mutation_allowed": False,
        },
        "dataset": contract,
        "dataset_fingerprint_before": before_fingerprint,
        "contamination": contamination,
        "environment": environment,
        "device_smoke": device_smoke,
        "model": {"pretrained": pretrained_record},
    }
    _write_json(report_root / "environment.json", environment)

    _stage("materializing linked Z0 validation view")
    z0_relative_images = _all_split_images(control_root, "val")
    z0_linked = create_linked_dataset(
        control_root,
        local_root / "z0_dataset",
        {"val": z0_relative_images},
    )
    _write_json(local_root / "z0_sources.json", z0_linked)
    z0_model = YOLO(str(model_source), task="detect")
    z0_eval_args = evaluator_arguments(
        config,
        data=z0_linked["dataset_yaml"],
        device=selected.torch_device,
        project=local_root / "ultralytics",
        name="z0_val",
    )
    _stage("running Z0 standard evaluator on full val split")
    z0_started = time.perf_counter()
    z0_result = z0_model.val(**z0_eval_args)
    z0_wall_seconds = time.perf_counter() - z0_started
    standard_metrics = evaluator_metrics(z0_result)
    _stage("running Z0 operational metrics on full val split")
    operational_metrics, operational_rows = _run_operational_metrics(
        z0_model,
        config=config,
        linked_root=Path(z0_linked["root"]),
        selected_device=selected,
    )
    _write_operational_csv(report_root / "z0_operational_per_image.csv", operational_rows)
    summary["z0"] = {
        "status": "complete",
        "split": "val",
        "images": contract["splits"]["val"]["images"],
        "objects": contract["splits"]["val"]["objects"],
        "evaluator_confidence": EVALUATOR_CONFIDENCE,
        "evaluator_nms_iou": EVALUATOR_NMS_IOU,
        "effective_arguments": z0_eval_args,
        "wall_seconds": z0_wall_seconds,
        "evaluator_metrics": standard_metrics,
        "operational_metrics": operational_metrics,
        "ultralytics_save_dir": _display_path(z0_result.save_dir),
    }

    _stage("materializing deterministic 256/64 A-smoke view")
    selection = deterministic_smoke_selection(control_root, seed=config.seed)
    smoke_objects = {
        split: selected_object_count(control_root, paths)
        for split, paths in selection.items()
    }
    selection_digest = hashlib.sha256(
        "\n".join(path for split in ("train", "val") for path in selection[split]).encode()
    ).hexdigest()
    smoke_linked = create_linked_dataset(
        control_root,
        local_root / "smoke_dataset",
        selection,
    )
    smoke_selection_artifact = {
        "seed": config.seed,
        "algorithm": "sorted paths + one random.Random(seed).sample pass per train,val",
        "counts": {split: len(paths) for split, paths in selection.items()},
        "objects": smoke_objects,
        "sha256": selection_digest,
        "paths": selection,
        "link_methods": smoke_linked["link_methods"],
        "contains_test": False,
    }
    _write_json(report_root / "smoke_selection.json", smoke_selection_artifact)
    _write_json(local_root / "smoke_sources.json", smoke_linked)
    smoke_args = training_arguments(
        config,
        data=smoke_linked["dataset_yaml"],
        device=selected.torch_device,
        project=local_root / "ultralytics",
        name="a_smoke",
        epochs=1,
    )
    smoke_model = YOLO(str(model_source), task="detect")
    _stage("running one-epoch A training smoke")
    smoke_started = time.perf_counter()
    smoke_result = smoke_model.train(**smoke_args)
    smoke_wall_seconds = time.perf_counter() - smoke_started
    smoke_save_dir = Path(smoke_result.save_dir)
    checkpoints = {
        "best": smoke_save_dir / "weights" / "best.pt",
        "last": smoke_save_dir / "weights" / "last.pt",
    }
    if not all(path.is_file() for path in checkpoints.values()):
        raise RuntimeError(f"Smoke checkpoints missing: {checkpoints}")
    reloaded: dict[str, dict] = {}
    inference_source = str(
        sorted((Path(smoke_linked["root"]) / "images" / "val").glob("*"))[0].absolute()
    )
    for checkpoint_name, checkpoint in checkpoints.items():
        _stage(f"reloading and validating {checkpoint_name}.pt")
        if not _git_ignored(checkpoint):
            raise RuntimeError(f"Checkpoint is not ignored by Git: {checkpoint}")
        loaded = YOLO(str(checkpoint), task="detect")
        prediction = loaded.predict(
            source=inference_source,
            imgsz=config.model.image_size,
            conf=config.model.confidence,
            iou=config.model.iou,
            classes=[config.model.person_class_id],
            device=selected.torch_device,
            verbose=False,
        )
        reload_args = evaluator_arguments(
            config,
            data=smoke_linked["dataset_yaml"],
            device=selected.torch_device,
            project=local_root / "ultralytics",
            name=f"a_smoke_{checkpoint_name}_reload_val",
            plots=False,
        )
        validation = loaded.val(**reload_args)
        reloaded[checkpoint_name] = {
            "status": "passed",
            "checkpoint": checkpoint_artifact(checkpoint, ROOT),
            "git_ignored": True,
            "reload_inference_images": len(prediction),
            "reload_inference_detections": sum(
                len(item.boxes) if item.boxes is not None else 0 for item in prediction
            ),
            "validation_images": SMOKE_VAL_IMAGES,
            "validation_metrics": evaluator_metrics(validation),
            "validation_save_dir": _display_path(validation.save_dir),
        }
    args_yaml_path = smoke_save_dir / "args.yaml"
    effective_ultralytics_args = (
        yaml.safe_load(args_yaml_path.read_text(encoding="utf-8"))
        if args_yaml_path.is_file()
        else None
    )
    summary["a_smoke"] = {
        "status": "complete",
        "seed": config.seed,
        "train_images": SMOKE_TRAIN_IMAGES,
        "val_images": SMOKE_VAL_IMAGES,
        "train_objects": smoke_objects["train"],
        "val_objects": smoke_objects["val"],
        "epochs": 1,
        "effective_arguments": smoke_args,
        "ultralytics_effective_arguments": effective_ultralytics_args,
        "training_wall_seconds": smoke_wall_seconds,
        "training_metrics": evaluator_metrics(smoke_result),
        "ultralytics_save_dir": _display_path(smoke_save_dir),
        "reload_verification": reloaded,
    }

    full_args = training_arguments(
        config,
        data=dataset_yaml,
        device=selected.torch_device,
        project=ROOT / "models" / "training_runs" / "full_a",
        name="a_full_seed42",
    )
    smoke_run_bytes = _directory_bytes(smoke_save_dir)
    estimate = _provisional_full_estimate(
        smoke_wall_seconds=smoke_wall_seconds,
        smoke_epoch_seconds=_ultralytics_epoch_seconds(smoke_save_dir),
        smoke_run_bytes=smoke_run_bytes,
        checkpoint_bytes=max(path.stat().st_size for path in checkpoints.values()),
        full_epochs=full_args["epochs"],
    )
    summary["full_a_plan"] = {
        "status": "not_started_requires_explicit_approval",
        "expected_command": (
            "python scripts/train.py --device cuda --confirm-full-run"
        ),
        "effective_arguments": full_args,
        "estimate": estimate,
        "expected_artifacts": [
            "best.pt",
            "last.pt",
            "args.yaml",
            "results.csv",
            "plots and validation diagnostics",
            "checkpoint SHA-256/reload report",
        ],
        "runtime_promotion": {
            "performed": False,
            "stage": "scripts/promote_model.py",
        },
    }
    _stage("fingerprinting control dataset after evaluation")
    after_fingerprint = dataset_fingerprint(control_root)
    summary["dataset_fingerprint_after"] = after_fingerprint
    summary["dataset_unchanged"] = before_fingerprint == after_fingerprint
    if not summary["dataset_unchanged"]:
        raise RuntimeError("Control coco_person changed during baseline run")
    summary["completed_at"] = utc_timestamp()
    summary["status"] = "complete"
    summary["artifacts"] = {
        "local_ignored_run_root": _display_path(local_root),
        "tracked_report_root": _display_path(report_root),
        "summary_json": _display_path(report_root / "summary.json"),
        "metrics_csv": _display_path(report_root / "metrics.csv"),
        "operational_per_image_csv": _display_path(
            report_root / "z0_operational_per_image.csv"
        ),
        "smoke_selection": _display_path(report_root / "smoke_selection.json"),
    }
    effective_config = {
        "run_id": run_id,
        "seed": config.seed,
        "source_config": _display_path(args.config),
        "source_config_sha256": file_sha256(args.config),
        "dataset_yaml": _display_path(dataset_yaml),
        "dataset_yaml_sha256": file_sha256(dataset_yaml),
        "z0_evaluator_arguments": z0_eval_args,
        "operational": {
            "confidence": config.model.confidence,
            "nms_iou": config.model.iou,
            "matching_iou": OPERATIONAL_MATCH_IOU,
            "matching_algorithm": "confidence_descending_best_unmatched_gt",
            "matching_algorithm_version": 2,
        },
        "a_smoke_arguments": smoke_args,
        "full_a_arguments_not_executed": full_args,
    }
    _write_json(report_root / "effective_config.json", effective_config)
    _write_json(report_root / "summary.json", summary)
    _write_metrics_csv(report_root / "metrics.csv", summary)
    commands = (
        "# Baseline commands\n\n"
        "Executed from Git Bash:\n\n"
        "```bash\n"
        "PYTHONUTF8=1 .venv/Scripts/python.exe scripts/run_baseline_stage.py "
        f"--device {args.device} --run-id {run_id}\n"
        "```\n\n"
        "Full A (not executed; requires separate approval):\n\n"
        "```bash\npython scripts/train.py --device cuda --confirm-full-run\n```\n"
    )
    (report_root / "commands.md").write_text(
        commands,
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown(report_root / "report.md", summary)
    _stage("run complete")
    return summary


def recompute_operational(args: argparse.Namespace) -> dict:
    from ultralytics import YOLO

    report_root = ROOT / "reports" / "baseline_runs" / args.run_id
    summary_path = report_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Completed baseline summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"Baseline run is not complete: {args.run_id}")

    _stage(f"verifying control fingerprint for {args.run_id}")
    current_fingerprint = dataset_fingerprint(Path(summary["dataset"]["root"]))
    recorded_fingerprint = summary["dataset_fingerprint_after"]
    if current_fingerprint != recorded_fingerprint:
        raise RuntimeError("Control dataset fingerprint changed since the baseline run")

    config = load_config(args.config)
    selected = select_device(force=None if args.device == "auto" else args.device)
    model_source = _pretrained_path(config.model.pretrained)
    if not isinstance(model_source, Path):
        raise FileNotFoundError(
            "Operational recompute requires the recorded local pretrained checkpoint"
        )
    recorded_model = summary["model"]["pretrained"]
    if file_sha256(model_source) != recorded_model["sha256"]:
        raise RuntimeError("Pretrained checkpoint SHA-256 changed since the baseline run")

    local_root = ROOT / summary["artifacts"]["local_ignored_run_root"]
    linked_root = local_root / "z0_dataset"
    if not linked_root.is_dir():
        raise FileNotFoundError(f"Linked Z0 dataset is unavailable: {linked_root}")

    standard_metrics = summary["z0"]["evaluator_metrics"]
    standard_metrics_sha256 = hashlib.sha256(
        json.dumps(standard_metrics, sort_keys=True).encode("utf-8")
    ).hexdigest()
    previous_operational = summary["z0"]["operational_metrics"]
    _stage("recomputing only Z0 operational metrics")
    model = YOLO(str(model_source), task="detect")
    operational_metrics, operational_rows = _run_operational_metrics(
        model,
        config=config,
        linked_root=linked_root,
        selected_device=selected,
    )

    corrected_at = operational_metrics["computed_at"]
    summary["schema_version"] = 2
    summary["updated_at"] = corrected_at
    summary["contamination"] = load_contamination_evidence(
        ROOT / "reports" / "dataset_audit.json"
    )
    summary["z0"]["operational_metrics"] = operational_metrics
    summary["z0"]["operational_recomputed_at"] = corrected_at
    summary["operational_recompute"] = {
        "status": "complete",
        "computed_at": corrected_at,
        "previous_matching_algorithm": previous_operational.get(
            "matching_algorithm", "global_iou_pair_greedy"
        ),
        "matching_algorithm": operational_metrics["matching_algorithm"],
        "matching_algorithm_version": operational_metrics[
            "matching_algorithm_version"
        ],
        "standard_evaluator_rerun": False,
        "a_smoke_rerun": False,
        "standard_evaluator_metrics_sha256": standard_metrics_sha256,
        "dataset_fingerprint_verified": True,
        "pretrained_checkpoint_sha256_verified": True,
    }

    effective_config_path = report_root / "effective_config.json"
    effective_config = json.loads(effective_config_path.read_text(encoding="utf-8"))
    effective_config["operational"] = {
        "confidence": config.model.confidence,
        "nms_iou": config.model.iou,
        "matching_iou": OPERATIONAL_MATCH_IOU,
        "matching_algorithm": operational_metrics["matching_algorithm"],
        "matching_algorithm_version": operational_metrics[
            "matching_algorithm_version"
        ],
        "recomputed_at": corrected_at,
    }

    _write_operational_csv(
        report_root / "z0_operational_per_image.csv", operational_rows
    )
    _write_json(effective_config_path, effective_config)
    _write_json(summary_path, summary)
    _write_metrics_csv(report_root / "metrics.csv", summary)
    _write_markdown(report_root / "report.md", summary)

    commands_path = report_root / "commands.md"
    commands = commands_path.read_text(encoding="utf-8")
    correction_heading = "## Operational matching correction"
    if correction_heading not in commands:
        commands += (
            f"\n{correction_heading}\n\n"
            "Executed from Git Bash without rerunning Z0 evaluator or A-smoke:\n\n"
            "```bash\n"
            "PYTHONUTF8=1 .venv/Scripts/python.exe scripts/run_baseline_stage.py "
            f"--device {args.device} --run-id {args.run_id} --operational-only\n"
            "```\n"
        )
        commands_path.write_text(commands, encoding="utf-8", newline="\n")
    _stage("operational correction complete")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the val-only Z0 evaluation and deterministic A training smoke"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument(
        "--run-id",
        default=f"baseline_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
    )
    parser.add_argument(
        "--operational-only",
        action="store_true",
        help="Recompute only operational metrics for an existing completed run",
    )
    args = parser.parse_args()
    report_root = ROOT / "reports" / "baseline_runs" / args.run_id
    try:
        summary = recompute_operational(args) if args.operational_only else run(args)
    except Exception:
        report_root.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        error_name = (
            "operational_recompute_error.log" if args.operational_only else "error.log"
        )
        (report_root / error_name).write_text(
            error,
            encoding="utf-8",
            newline="\n",
        )
        print(error, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "run_id": summary["run_id"],
                "report": summary["artifacts"]["summary_json"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
