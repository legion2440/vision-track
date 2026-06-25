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
from vision_track.detector import OnnxRuntimeBackend
from vision_track.device import DeviceInfo, select_device
from vision_track.metrics import detection_scores


def yolo_labels_to_xyxy(label_path: Path, width: int, height: int) -> np.ndarray:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, x_center, y_center, box_width, box_height = map(float, line.split())
        boxes.append(
            [
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                (x_center + box_width / 2) * width,
                (y_center + box_height / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def evaluate_onnx(
    model_path: Path,
    image_dir: Path,
    label_dir: Path,
    image_size: int,
    limit: int | None,
) -> dict:
    backend = OnnxRuntimeBackend(
        model_path,
        DeviceInfo("cpu", "cpu", "CPU", "ONNX Runtime CPU"),
        image_size=image_size,
        confidence=0.35,
        iou=0.5,
    )
    backend.load()
    predictions = []
    ground_truth = []
    latencies = []
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    if limit:
        image_paths = image_paths[:limit]
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = backend.infer(image)
        predictions.append(result.detections.xyxy)
        ground_truth.append(
            yolo_labels_to_xyxy(
                label_dir / f"{image_path.stem}.txt",
                image.shape[1],
                image.shape[0],
            )
        )
        latencies.append(result.latency_ms)
    scores = detection_scores(predictions, ground_truth)
    return {
        "detection_precision": scores.precision,
        "detection_recall": scores.recall,
        "f1_score": scores.f1,
        "mAP50": None,
        "mAP50_95": None,
        "inference_latency_ms": float(np.mean(latencies)) if latencies else None,
        "evaluated_images": len(predictions),
        "backend": backend.actual_provider,
    }


def evaluate_pytorch(
    model_path: Path,
    dataset_yaml: Path,
    split: str,
    image_size: int,
    device: str,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="detect")
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=image_size,
        classes=[0],
        device=device,
        plots=False,
        verbose=False,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    try:
        _, parameters, _, flops = model.info(verbose=False)
    except Exception:
        parameters, flops = None, None
    return {
        "detection_precision": precision,
        "detection_recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "inference_latency_ms": float(metrics.speed.get("inference", 0.0)),
        "parameter_count": parameters,
        "flops": flops,
        "backend": "pytorch",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained, pruned, and INT8 models")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int)
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
    device = select_device()
    artifacts = [
        ("fine_tuned", resolve_project_path(config.model.checkpoint), "pytorch"),
        ("pruned", resolve_project_path(config.model.pruned_checkpoint), "pytorch"),
        ("quantized_int8", resolve_project_path(config.model.quantized_checkpoint), "onnxruntime"),
    ]
    models = []
    for name, path, backend in artifacts:
        if not path.exists():
            models.append(
                {"name": name, "status": "missing", "artifact": str(path)}
            )
            continue
        started = time.perf_counter()
        if backend == "pytorch":
            result = evaluate_pytorch(
                path,
                dataset_yaml,
                args.split,
                config.model.image_size,
                device.torch_device,
            )
        else:
            result = evaluate_onnx(
                path,
                dataset_root / "images" / args.split,
                dataset_root / "labels" / args.split,
                config.model.image_size,
                args.limit,
            )
        result.update(
            {
                "name": name,
                "status": "measured",
                "artifact": str(path),
                "file_size_mb": path.stat().st_size / 1_000_000,
                "evaluation_seconds": time.perf_counter() - started,
            }
        )
        models.append(result)
    payload = {
        "split": args.split,
        "test_isolation_acknowledged": args.acknowledge_test_isolation,
        "models": models,
    }
    output = ROOT / "reports" / "artifact_comparison.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

