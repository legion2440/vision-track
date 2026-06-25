from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import select_device
from vision_track.metrics import software_versions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26n for person detection")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Skip the one-time isolated test evaluation while iterating on training",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    config = load_config(args.config)
    training = config.raw["training"]
    selected = select_device(force=None if args.device == "auto" else args.device)
    dataset_yaml = resolve_project_path(training["dataset_yaml"])
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset configuration not found: {dataset_yaml}. Run prepare_coco_person.py first."
        )
    run_dir = ROOT / "models" / "training_runs"
    model = YOLO(config.model.pretrained, task="detect")
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs or int(training["epochs"]),
        batch=args.batch_size or int(training["batch_size"]),
        imgsz=config.model.image_size,
        patience=int(training["patience"]),
        optimizer=training["optimizer"],
        lr0=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        workers=int(training["workers"]),
        seed=config.seed,
        deterministic=True,
        device=selected.torch_device,
        project=str(run_dir),
        name="person_yolo26n",
        exist_ok=True,
        val=True,
        plots=True,
        classes=[config.model.person_class_id],
        **training["augmentations"],
    )
    source_best = Path(results.save_dir) / "weights" / "best.pt"
    if not source_best.exists():
        raise RuntimeError(f"Ultralytics did not produce a best checkpoint: {source_best}")
    destination = resolve_project_path(config.model.checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_best, destination)

    report = {
        "status": "trained",
        "pretrained_model": config.model.pretrained,
        "best_checkpoint": str(destination),
        "dataset": str(dataset_yaml),
        "device": selected.kind,
        "parameters": training,
        "seed": config.seed,
        "software_versions": software_versions(
            ["torch", "ultralytics", "numpy", "opencv-python"]
        ),
        "final_test": None,
    }
    if not args.skip_final_test:
        best_model = YOLO(str(destination), task="detect")
        metrics = best_model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=config.model.image_size,
            classes=[config.model.person_class_id],
            device=selected.torch_device,
            plots=True,
            verbose=False,
        )
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        report["final_test"] = {
            "detection_precision": precision,
            "detection_recall": recall,
            "f1_score": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        }
    report_path = ROOT / "reports" / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
