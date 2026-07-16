from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from .configuration import AppConfig
from .dataset_validation import IMAGE_EXTENSIONS
from .metrics import box_iou_matrix


EVALUATOR_CONFIDENCE = 0.001
EVALUATOR_NMS_IOU = 0.70
OPERATIONAL_MATCH_IOU = 0.50
SMOKE_TRAIN_IMAGES = 256
SMOKE_VAL_IMAGES = 64


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def pytorch_runtime_device(device: object) -> str:
    kind = str(getattr(device, "kind"))
    configured = str(getattr(device, "torch_device"))
    if kind == "cuda" and configured.isdigit():
        return f"cuda:{configured}"
    return configured


def assert_no_test_paths(paths: Iterable[str | Path]) -> None:
    for value in paths:
        normalized = Path(value).as_posix().lower()
        parts = tuple(part for part in normalized.split("/") if part)
        if "test" in parts:
            raise ValueError(f"Test split is forbidden in baseline development: {value}")


def _resolve_dataset_root(dataset_yaml: Path, payload: dict) -> Path:
    configured = Path(str(payload.get("path") or dataset_yaml.parent))
    if configured.is_absolute():
        return configured.resolve()
    return (dataset_yaml.parent / configured).resolve()


def inspect_dataset_contract(
    dataset_yaml: str | Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    yaml_path = Path(dataset_yaml).resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    root = _resolve_dataset_root(yaml_path, payload)
    names = payload.get("names")
    if names not in ({0: "person"}, {"0": "person"}, ["person"]):
        raise ValueError(f"Expected a single person class in {yaml_path}, got {names!r}")

    splits: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        configured = payload.get(split)
        if not isinstance(configured, str):
            raise ValueError(f"Dataset YAML must define string path for {split}")
        image_dir = (root / configured).resolve()
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing {split} image/label directories")
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(
                f"Image/label mismatch in {split}: "
                f"missing_labels={len(image_stems - label_stems)}, "
                f"missing_images={len(label_stems - image_stems)}"
            )
        object_count = sum(
            len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
            for path in labels
        )
        splits[split] = {
            "image_dir": image_dir.as_posix(),
            "label_dir": label_dir.resolve().as_posix(),
            "images": len(images),
            "labels": len(labels),
            "objects": object_count,
        }
    if expected_counts:
        actual_counts = {split: item["images"] for split, item in splits.items()}
        if actual_counts != expected_counts:
            raise ValueError(
                f"Dataset split counts changed: expected={expected_counts}, "
                f"actual={actual_counts}"
            )
    return {
        "dataset_yaml": yaml_path.as_posix(),
        "root": root.as_posix(),
        "names": {"0": "person"},
        "splits": splits,
    }


def dataset_fingerprint(dataset_root: str | Path) -> dict:
    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    content_hashed_files = 0
    files: list[Path] = []
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        files.extend(Path(directory) / file_name for file_name in file_names)
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if path.suffix.lower() in {".txt", ".yaml", ".yml", ".json"}:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            content_hashed_files += 1
        file_count += 1
        total_bytes += stat.st_size
    return {
        "algorithm": "sha256(path,size,mtime_ns,full_text_content)",
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "content_hashed_files": content_hashed_files,
    }


def deterministic_smoke_selection(
    dataset_root: str | Path,
    *,
    seed: int = 42,
    train_images: int = SMOKE_TRAIN_IMAGES,
    val_images: int = SMOKE_VAL_IMAGES,
) -> dict[str, list[str]]:
    root = Path(dataset_root).resolve()
    candidates: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        candidates[split] = sorted(
            path
            for path in (root / "images" / split).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    requested = {"train": train_images, "val": val_images}
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    for split in ("train", "val"):
        if len(candidates[split]) < requested[split]:
            raise ValueError(
                f"Not enough {split} images for smoke selection: "
                f"{len(candidates[split])} < {requested[split]}"
            )
        sampled = rng.sample(candidates[split], requested[split])
        selected[split] = sorted(path.relative_to(root).as_posix() for path in sampled)
    assert_no_test_paths(path for paths in selected.values() for path in paths)
    return selected


def selected_object_count(
    dataset_root: str | Path,
    relative_images: Sequence[str],
) -> int:
    root = Path(dataset_root).resolve()
    total = 0
    for relative_image in relative_images:
        relative = Path(relative_image)
        if len(relative.parts) < 3 or relative.parts[0] != "images":
            raise ValueError(f"Unexpected selected image path: {relative_image}")
        split = relative.parts[1]
        if split not in {"train", "val"}:
            raise ValueError(f"Forbidden selected split: {relative_image}")
        label = root / "labels" / split / f"{relative.stem}.txt"
        total += sum(
            bool(line.strip())
            for line in label.read_text(encoding="utf-8").splitlines()
        )
    return total


def _link_to_control(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source.resolve(), destination)
        return "symlink"
    except OSError:
        os.link(source.resolve(), destination)
        return "hardlink"


def create_linked_dataset(
    control_root: str | Path,
    destination: str | Path,
    split_images: dict[str, Sequence[str]],
) -> dict:
    control = Path(control_root).resolve()
    output = Path(destination).resolve()
    if output == control or control in output.parents:
        raise ValueError("Linked dataset must be outside the control dataset")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    methods: dict[str, int] = {}
    source_manifest: dict[str, list[dict]] = {}
    for split, relative_images in split_images.items():
        if split not in {"train", "val"}:
            raise ValueError(f"Forbidden linked split: {split}")
        assert_no_test_paths(relative_images)
        list_lines: list[str] = []
        source_manifest[split] = []
        for relative_image in relative_images:
            source_image = (control / relative_image).resolve()
            expected_parent = (control / "images" / split).resolve()
            if expected_parent not in source_image.parents:
                raise ValueError(f"Image is outside control {split}: {source_image}")
            source_label = control / "labels" / split / f"{source_image.stem}.txt"
            if not source_image.is_file() or not source_label.is_file():
                missing = source_image if not source_image.is_file() else source_label
                raise FileNotFoundError(missing)
            linked_image = output / "images" / split / source_image.name
            linked_label = output / "labels" / split / source_label.name
            image_method = _link_to_control(source_image, linked_image)
            label_method = _link_to_control(source_label, linked_label)
            methods[image_method] = methods.get(image_method, 0) + 1
            methods[label_method] = methods.get(label_method, 0) + 1
            list_lines.append(linked_image.absolute().as_posix())
            source_manifest[split].append(
                {
                    "linked_image": linked_image.relative_to(output).as_posix(),
                    "control_image": source_image.relative_to(control).as_posix(),
                    "control_label": source_label.relative_to(control).as_posix(),
                    "link_method": image_method,
                }
            )
        list_path = output / "lists" / f"{split}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8", newline="\n")

    train_list = "lists/train.txt" if "train" in split_images else "lists/val.txt"
    yaml_payload = {
        "path": output.as_posix(),
        "train": train_list,
        "val": "lists/val.txt",
        "names": {0: "person"},
    }
    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(yaml_payload, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "dataset_yaml": yaml_path.as_posix(),
        "root": output.as_posix(),
        "split_counts": {split: len(paths) for split, paths in split_images.items()},
        "link_methods": methods,
        "sources": source_manifest,
        "contains_test": False,
    }


def training_arguments(
    config: AppConfig,
    *,
    data: str | Path,
    device: str,
    project: str | Path,
    name: str,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> dict:
    training = config.raw["training"]
    return {
        "data": str(Path(data).resolve()),
        "epochs": int(epochs if epochs is not None else training["epochs"]),
        "batch": int(batch_size if batch_size is not None else training["batch_size"]),
        "imgsz": config.model.image_size,
        "patience": int(training["patience"]),
        "optimizer": str(training["optimizer"]),
        "lr0": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "workers": int(training["workers"]),
        "seed": config.seed,
        "deterministic": True,
        "device": device,
        "project": str(Path(project).resolve()),
        "name": name,
        "exist_ok": False,
        "val": True,
        "plots": True,
        "save": True,
        "classes": [config.model.person_class_id],
        **training["augmentations"],
    }


def evaluator_arguments(
    config: AppConfig,
    *,
    data: str | Path,
    device: str,
    project: str | Path,
    name: str,
    plots: bool = True,
) -> dict:
    return {
        "data": str(Path(data).resolve()),
        "split": "val",
        "imgsz": config.model.image_size,
        "batch": int(config.raw["training"]["batch_size"]),
        "conf": EVALUATOR_CONFIDENCE,
        "iou": EVALUATOR_NMS_IOU,
        "classes": [config.model.person_class_id],
        "device": device,
        "plots": plots,
        "verbose": False,
        "project": str(Path(project).resolve()),
        "name": name,
        "exist_ok": False,
    }


def evaluator_metrics(metrics: object) -> dict:
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
        "results_dict": {
            str(key): float(value)
            for key, value in getattr(metrics, "results_dict", {}).items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
    }


def read_yolo_xyxy(label_path: str | Path, width: int, height: int) -> np.ndarray:
    boxes: list[list[float]] = []
    for raw_line in Path(label_path).read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if len(fields) != 5 or int(float(fields[0])) != 0:
            raise ValueError(f"Invalid person label in {label_path}: {raw_line}")
        x_center, y_center, box_width, box_height = map(float, fields[1:])
        boxes.append(
            [
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                (x_center + box_width / 2) * width,
                (y_center + box_height / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def match_detection_counts(
    predicted_xyxy: np.ndarray,
    predicted_confidences: np.ndarray,
    expected_xyxy: np.ndarray,
    *,
    iou_threshold: float = OPERATIONAL_MATCH_IOU,
) -> tuple[int, int, int]:
    predicted = np.asarray(predicted_xyxy, dtype=np.float32).reshape(-1, 4)
    confidences = np.asarray(predicted_confidences, dtype=np.float32).reshape(-1)
    expected = np.asarray(expected_xyxy, dtype=np.float32).reshape(-1, 4)
    if len(confidences) != len(predicted):
        raise ValueError(
            "Predicted confidence count must match predicted box count: "
            f"{len(confidences)} != {len(predicted)}"
        )
    ious = box_iou_matrix(predicted, expected)
    matched_ground_truth: set[int] = set()
    true_positives = 0
    for pred_index in np.argsort(-confidences, kind="stable"):
        available = [
            gt_index
            for gt_index in range(len(expected))
            if gt_index not in matched_ground_truth
        ]
        if not available:
            continue
        best_ground_truth = max(
            available,
            key=lambda gt_index: float(ious[int(pred_index), gt_index]),
        )
        if float(ious[int(pred_index), best_ground_truth]) < iou_threshold:
            continue
        matched_ground_truth.add(best_ground_truth)
        true_positives += 1
    return (
        true_positives,
        len(predicted) - true_positives,
        len(expected) - len(matched_ground_truth),
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_artifact(path: str | Path, repository_root: str | Path) -> dict:
    checkpoint = Path(path).resolve()
    root = Path(repository_root).resolve()
    try:
        display_path = checkpoint.relative_to(root).as_posix()
    except ValueError:
        display_path = checkpoint.as_posix()
    return {
        "path": display_path,
        "sha256": file_sha256(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }


def environment_payload(packages: Sequence[str]) -> dict:
    package_versions: dict[str, str] = {}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    import torch

    cuda_available = bool(torch.cuda.is_available())
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    cuda = {
        "available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": cudnn_version,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count() if cuda_available else 0)
        ],
    }
    try:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free,pstate,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        nvidia_smi = None
    return {
        "captured_at": utc_timestamp(),
        "python": {
            "version": sys.version,
            "executable": Path(sys.executable).resolve().as_posix(),
            "required": "==3.13.*",
        },
        "platform": platform.platform(),
        "packages": package_versions,
        "cuda": cuda,
        "nvidia_smi": nvidia_smi,
    }


def load_contamination_evidence(audit_report: str | Path) -> dict:
    payload = json.loads(Path(audit_report).read_text(encoding="utf-8"))
    duplicates = payload["prepared_dataset"]["duplicates"]
    review = payload["manual_review"]
    prepared_val = payload["expected_current_preparer"]["val"]
    same_scene = sum(
        item["decision"] in {"duplicate", "near_duplicate_same_scene"}
        for item in review["phash_clusters"]
    )
    return {
        "control_dataset_unchanged": True,
        "exact_sha_cross_split_groups": duplicates["exact_cross_split_group_count"],
        "reviewed_cross_split_same_scene_phash_clusters": same_scene,
        "kept_in_control_splits": True,
        "unlabeled_crowd_val": {
            "images_with_regions": prepared_val[
                "images_with_unlabeled_crowd_regions"
            ],
            "regions": prepared_val["unlabeled_crowd_annotations"],
            "interpretation": (
                "Real crowded people omitted from YOLO labels can be counted as "
                "operational false positives and can suppress measured recall."
            ),
        },
        "interpretation": (
            "Z0/A control metrics include small known cross-split same-scene "
            "contamination; fixes belong only to dataset_v2 or a separate "
            "deduplicated materialization."
        ),
    }
