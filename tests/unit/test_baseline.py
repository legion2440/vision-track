from __future__ import annotations

import csv
import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.train import run as run_full_training
from vision_track.baseline import (
    assert_no_test_paths,
    create_linked_dataset,
    dataset_fingerprint,
    deterministic_smoke_selection,
    inspect_dataset_contract,
    load_contamination_evidence,
    match_detection_counts,
    pytorch_runtime_device,
    selected_object_count,
    training_arguments,
)
from vision_track.configuration import load_config
from scripts.run_baseline_stage import (
    _provisional_full_estimate,
    _run_operational_metrics,
    _ultralytics_epoch_seconds,
    _write_metrics_csv,
)


ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path, counts: dict[str, int]) -> tuple[Path, Path]:
    root = tmp_path / "control"
    for split, count in counts.items():
        for index in range(count):
            stem = f"{index + 1:012d}"
            image = root / "images" / split / f"{stem}.jpg"
            label = root / "labels" / split / f"{stem}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image-{split}-{index}".encode())
            label.write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    dataset_yaml = root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": root.as_posix(),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "person"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root, dataset_yaml


def test_test_path_guard_rejects_only_forbidden_split() -> None:
    assert_no_test_paths(["images/train/a.jpg", "images/val/b.jpg"])

    with pytest.raises(ValueError, match="Test split is forbidden"):
        assert_no_test_paths(["images/test/c.jpg"])


def test_dataset_contract_confirms_counts_and_objects(tmp_path: Path) -> None:
    _, dataset_yaml = _dataset(tmp_path, {"train": 4, "val": 3, "test": 2})

    contract = inspect_dataset_contract(
        dataset_yaml,
        expected_counts={"train": 4, "val": 3, "test": 2},
    )

    assert contract["splits"]["train"]["objects"] == 4
    assert contract["splits"]["val"]["images"] == 3
    assert contract["names"] == {"0": "person"}


def test_smoke_selection_is_seeded_and_never_uses_test(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path, {"train": 12, "val": 8, "test": 4})

    first = deterministic_smoke_selection(
        root,
        seed=42,
        train_images=5,
        val_images=3,
    )
    second = deterministic_smoke_selection(
        root,
        seed=42,
        train_images=5,
        val_images=3,
    )

    assert first == second
    assert len(first["train"]) == 5
    assert len(first["val"]) == 3
    assert all("/test/" not in path for paths in first.values() for path in paths)
    assert selected_object_count(root, first["train"]) == 5
    assert selected_object_count(root, first["val"]) == 3


def test_linked_dataset_points_to_control_and_keeps_cache_outside(
    tmp_path: Path,
) -> None:
    root, _ = _dataset(tmp_path, {"train": 3, "val": 2, "test": 1})
    selection = {
        "train": ["images/train/000000000001.jpg"],
        "val": ["images/val/000000000001.jpg"],
    }

    linked = create_linked_dataset(root, tmp_path / "linked", selection)
    linked_root = Path(linked["root"])
    linked_image = linked_root / "images" / "train" / "000000000001.jpg"

    assert os.path.samefile(linked_image, root / selection["train"][0])
    assert Path(linked["dataset_yaml"]).is_file()
    payload = yaml.safe_load(Path(linked["dataset_yaml"]).read_text(encoding="utf-8"))
    assert "test" not in payload
    assert "linked/images/train" in (
        linked_root / "lists" / "train.txt"
    ).read_text(encoding="utf-8").replace("\\", "/")
    cache = linked_root / "labels" / "train.cache"
    cache.write_bytes(b"cache")
    assert cache.is_file()
    assert not (root / "labels" / "train.cache").exists()


def test_dataset_fingerprint_detects_added_cache(tmp_path: Path) -> None:
    root, _ = _dataset(tmp_path, {"train": 1, "val": 1, "test": 1})
    before = dataset_fingerprint(root)

    (root / "labels" / "val.cache").write_bytes(b"cache")
    after = dataset_fingerprint(root)

    assert before["sha256"] != after["sha256"]
    assert after["file_count"] == before["file_count"] + 1


def test_training_arguments_use_project_config_and_explicit_smoke_override(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs" / "app.yaml")
    data = tmp_path / "dataset.yaml"
    data.write_text("train: x\nval: y\nnames: [person]\n", encoding="utf-8")

    arguments = training_arguments(
        config,
        data=data,
        device="0",
        project=tmp_path / "runs",
        name="smoke",
        epochs=1,
    )

    assert arguments["epochs"] == 1
    assert arguments["batch"] == 16
    assert arguments["patience"] == 12
    assert arguments["optimizer"] == "AdamW"
    assert arguments["lr0"] == 0.001
    assert arguments["mosaic"] == 0.5
    assert arguments["deterministic"] is True
    assert arguments["classes"] == [0]


def test_pytorch_runtime_device_adapts_ultralytics_cuda_index() -> None:
    cuda = SimpleNamespace(kind="cuda", torch_device="0")
    cpu = SimpleNamespace(kind="cpu", torch_device="cpu")

    assert pytorch_runtime_device(cuda) == "cuda:0"
    assert pytorch_runtime_device(cpu) == "cpu"


def test_operational_matching_uses_one_to_one_iou() -> None:
    expected = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    predicted = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [50, 50, 60, 60]],
        dtype=np.float32,
    )
    confidences = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert match_detection_counts(
        predicted,
        confidences,
        expected,
        iou_threshold=0.5,
    ) == (1, 2, 1)


def test_operational_matching_is_confidence_ordered() -> None:
    expected = np.array(
        [[0, 0, 10, 10], [4, 0, 14, 10]],
        dtype=np.float32,
    )
    predicted = np.array(
        [
            [0, 0, 10, 10],
            [1.5, 0, 11.5, 10],
        ],
        dtype=np.float32,
    )
    confidences = np.array([0.1, 0.9], dtype=np.float32)

    assert match_detection_counts(
        predicted,
        confidences,
        expected,
        iou_threshold=0.5,
    ) == (1, 1, 1)


def test_operational_matching_rejects_mismatched_confidences() -> None:
    with pytest.raises(ValueError, match="confidence count"):
        match_detection_counts(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((1,), dtype=np.float32),
            np.zeros((1, 4), dtype=np.float32),
        )


def test_operational_metrics_use_directory_source_for_real_batching(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "val" / "frame.jpg"
    label = tmp_path / "labels" / "val" / "frame.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    captured: dict = {}

    class Model:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return iter(
                [
                    SimpleNamespace(
                        path=str(image),
                        orig_shape=(100, 100),
                        boxes=None,
                        speed={},
                    )
                ]
            )

    config = SimpleNamespace(
        raw={"training": {"batch_size": 16}},
        model=SimpleNamespace(
            image_size=640,
            confidence=0.35,
            iou=0.5,
            person_class_id=0,
        ),
    )
    device = SimpleNamespace(torch_device="0")

    metrics, rows = _run_operational_metrics(
        Model(),
        config=config,
        linked_root=tmp_path,
        selected_device=device,
    )

    assert captured["source"] == str((tmp_path / "images" / "val").absolute())
    assert not isinstance(captured["source"], list)
    assert captured["batch"] == 16
    assert metrics["images"] == 1
    assert len(rows) == 1


def test_metrics_csv_links_metrics_to_checkpoint_and_config(tmp_path: Path) -> None:
    speed = {"inference": 2.5}
    summary = {
        "model": {
            "pretrained": {"path": "yolo26n.pt", "sha256": "pretrained-sha"}
        },
        "artifacts": {"tracked_report_root": "reports/baseline_runs/run"},
        "z0": {
            "images": 2,
            "objects": 3,
            "evaluator_metrics": {
                "precision": 0.8,
                "recall": 0.6,
                "mAP50": 0.7,
                "mAP50_95": 0.5,
                "speed_ms_per_image": speed,
            },
            "operational_metrics": {
                "confidence": 0.35,
                "images": 2,
                "objects": 3,
                "precision": 0.9,
                "recall": 0.5,
                "detections": 2,
                "false_positives": 0,
                "false_negatives": 1,
                "false_positives_per_image": 0.0,
                "speed_ms_per_image": speed,
                "wall_ms_per_image": 4.0,
            },
        },
        "a_smoke": {"val_objects": 1, "reload_verification": {}},
    }
    output = tmp_path / "metrics.csv"

    _write_metrics_csv(output, summary)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["checkpoint"] == "yolo26n.pt"
    assert rows[0]["checkpoint_sha256"] == "pretrained-sha"
    assert rows[0]["config_artifact"].endswith("/effective_config.json")
    assert rows[0]["images"] == "2"
    assert rows[0]["objects"] == "3"
    assert rows[0]["inference_ms_per_image"] == "2.5"


def test_full_estimate_uses_ultralytics_epoch_time_not_total_wall(
    tmp_path: Path,
) -> None:
    (tmp_path / "results.csv").write_text(
        "epoch,time,metric\n1,15.5,0.1\n",
        encoding="utf-8",
    )

    epoch_seconds = _ultralytics_epoch_seconds(tmp_path)
    estimate = _provisional_full_estimate(
        smoke_wall_seconds=40.0,
        smoke_epoch_seconds=epoch_seconds,
        smoke_run_bytes=100,
        checkpoint_bytes=10,
        full_epochs=80,
    )

    assert epoch_seconds == 15.5
    assert estimate["smoke_training_wall_seconds"] == 40.0
    assert estimate["smoke_ultralytics_epoch_seconds"] == 15.5
    assert estimate["linear_extrapolation_hours"] < 100


def test_contamination_evidence_counts_reviewed_same_scene_clusters(
    tmp_path: Path,
) -> None:
    report = {
        "prepared_dataset": {
            "duplicates": {"exact_cross_split_group_count": 0}
        },
        "manual_review": {
            "phash_clusters": [
                {"decision": "near_duplicate_same_scene"},
                {"decision": "similar_not_duplicate"},
                {"decision": "duplicate"},
            ]
        },
        "expected_current_preparer": {
            "val": {
                "images_with_unlabeled_crowd_regions": 117,
                "unlabeled_crowd_annotations": 117,
            }
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    evidence = load_contamination_evidence(path)

    assert evidence["exact_sha_cross_split_groups"] == 0
    assert evidence["reviewed_cross_split_same_scene_phash_clusters"] == 2
    assert evidence["unlabeled_crowd_val"]["regions"] == 117


def test_full_training_requires_explicit_confirmation() -> None:
    with pytest.raises(RuntimeError, match="requires --confirm-full-run"):
        run_full_training(Namespace(confirm_full_run=False))


def test_full_training_has_no_runtime_promotion_responsibility() -> None:
    source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")

    assert "shutil.copy2" not in source
    assert "runtime_best_checkpoint" not in source
    assert "config.model.checkpoint" not in source


def test_model_weights_and_training_runs_are_git_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "models/training_runs/" in ignore
    assert "models/checkpoints/*.pt" in ignore
    assert "!models/checkpoints/best.pt" not in ignore
