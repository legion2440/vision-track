from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import select_device


def metric_payload(metrics, model_name: str, split: str, device: str) -> dict:
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "model_name": model_name,
        "split": split,
        "device": device,
        "detection_precision": precision,
        "detection_recall": recall,
        "f1_score": f1,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO artifact")
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from ultralytics import YOLO

    config = load_config()
    selected = select_device(force=None if args.device == "auto" else args.device)
    model_path = args.model or config.model.pretrained
    data_path = args.data or str(resolve_project_path(config.raw["training"]["dataset_yaml"]))
    model = YOLO(model_path, task="detect")
    metrics = model.val(
        data=data_path,
        split=args.split,
        imgsz=config.model.image_size,
        classes=[config.model.person_class_id],
        device=selected.torch_device,
        plots=True,
        verbose=False,
    )
    payload = metric_payload(metrics, str(model_path), args.split, selected.kind)
    output = args.output or ROOT / "reports" / f"{Path(str(model_path)).stem}_{args.split}_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

