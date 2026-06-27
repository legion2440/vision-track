from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.detector import create_backend
from vision_track.device import DeviceInfo, select_device
from vision_track.metrics import detection_scores


PERSON_CLASS_ID = 0


def yolo_labels_to_xyxy(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    if not label_path.exists():
        return np.empty((0, 4), dtype=np.float32)
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())
        if int(class_id) != PERSON_CLASS_ID:
            continue
        boxes.append(
            [
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                (x_center + box_width / 2) * width,
                (y_center + box_height / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def image_paths_for_split(
    image_dir: Path,
    *,
    limit: int | None,
) -> list[Path]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    return image_paths[:limit] if limit else image_paths


def smoke_match_scores(
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    *,
    iou_threshold: float,
):
    return detection_scores(predictions, ground_truth, iou_threshold=iou_threshold)


def _read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def _warmup_backend(backend, image_paths: list[Path], warmup_count: int) -> None:
    if not image_paths:
        return
    for index in range(warmup_count):
        image = _read_image(image_paths[index % len(image_paths)])
        if image is not None:
            backend.infer(image)


def evaluate_artifact(
    *,
    name: str,
    model_path: Path,
    backend_name: str,
    device: DeviceInfo,
    image_paths: list[Path],
    label_dir: Path,
    image_size: int,
    confidence: float,
    nms_iou: float,
    gt_iou: float,
    warmup_count: int,
) -> dict:
    backend = create_backend(
        backend_name,
        model_path,
        device,
        image_size=image_size,
        confidence=confidence,
        iou=nms_iou,
        person_class_id=PERSON_CLASS_ID,
    )
    backend.load()
    _warmup_backend(backend, image_paths, warmup_count)
    predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []
    latencies: list[float] = []
    last_backend = backend.name
    last_device = device.torch_device
    last_provider = getattr(backend, "actual_provider", None) or None
    started = time.perf_counter()
    for image_path in image_paths:
        image = _read_image(image_path)
        if image is None:
            continue
        result = backend.infer(image)
        last_backend = result.backend
        last_device = result.device
        last_provider = result.provider
        predictions.append(result.detections.xyxy)
        ground_truth.append(
            yolo_labels_to_xyxy(
                label_dir / f"{image_path.stem}.txt",
                image.shape[1],
                image.shape[0],
            )
        )
        latencies.append(result.latency_ms)
    elapsed = time.perf_counter() - started
    scores = smoke_match_scores(predictions, ground_truth, iou_threshold=gt_iou)
    measured = len(predictions)
    return {
        "name": name,
        "status": "measured" if measured else "no_images",
        "artifact": str(model_path),
        "backend": last_backend,
        "device": last_device,
        "provider": last_provider,
        "file_size_mb": model_path.stat().st_size / 1_000_000,
        "detection_precision": scores.precision,
        "detection_recall": scores.recall,
        "f1_score": scores.f1,
        "true_positives": scores.true_positives,
        "false_positives": scores.false_positives,
        "false_negatives": scores.false_negatives,
        "inference_latency_ms": float(np.mean(latencies)) if latencies else None,
        "throughput_fps": measured / elapsed if elapsed > 0 and measured else None,
        "warmup_images": warmup_count,
        "measured_images": measured,
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "ground_truth_iou_threshold": gt_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained, pruned, and INT8 models")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, help="Maximum measured images per artifact")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--gt-iou", type=float, default=0.5)
    parser.add_argument(
        "--acknowledge-test-isolation",
        action="store_true",
        help="Required for final test evaluation after all choices are frozen",
    )
    args = parser.parse_args()
    if args.split == "test" and not args.acknowledge_test_isolation:
        raise SystemExit(
            "Refusing to inspect the isolated test split without --acknowledge-test-isolation"
        )

    config = load_config()
    dataset_yaml = resolve_project_path(config.raw["training"]["dataset_yaml"])
    dataset_root = dataset_yaml.parent
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label split not found: {label_dir}")
    image_paths = image_paths_for_split(image_dir, limit=args.limit)
    device = select_device()
    artifacts = [
        ("fine_tuned", resolve_project_path(config.model.checkpoint), "pytorch", device),
        ("pruned", resolve_project_path(config.model.pruned_checkpoint), "pytorch", device),
        (
            "quantized_int8",
            resolve_project_path(config.model.quantized_checkpoint),
            "onnxruntime",
            device,
        ),
    ]
    models = []
    for name, path, backend_name, artifact_device in artifacts:
        if not path.exists():
            models.append({"name": name, "status": "missing", "artifact": str(path)})
            continue
        started = time.perf_counter()
        try:
            result = evaluate_artifact(
                name=name,
                model_path=path,
                backend_name=backend_name,
                device=artifact_device,
                image_paths=image_paths,
                label_dir=label_dir,
                image_size=config.model.image_size,
                confidence=args.confidence,
                nms_iou=args.nms_iou,
                gt_iou=args.gt_iou,
                warmup_count=args.warmup,
            )
            result["evaluation_seconds"] = time.perf_counter() - started
        except Exception as exc:
            result = {
                "name": name,
                "status": "failed",
                "artifact": str(path),
                "error": str(exc),
            }
        models.append(result)
    payload = {
        "protocol": {
            "split": args.split,
            "image_size": config.model.image_size,
            "confidence_threshold": args.confidence,
            "nms_iou_threshold": args.nms_iou,
            "ground_truth_iou_threshold": args.gt_iou,
            "image_limit": args.limit,
            "warmup_count": args.warmup,
            "measured_image_count": len(image_paths),
            "matching": "greedy descending IoU, one prediction per ground-truth box",
        },
        "test_isolation_acknowledged": args.acknowledge_test_isolation,
        "models": models,
    }
    output = ROOT / "reports" / "artifact_comparison.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
