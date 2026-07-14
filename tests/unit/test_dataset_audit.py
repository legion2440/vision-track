from __future__ import annotations

import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from scripts.prepare_coco_person import convert_split
from vision_track.dataset_audit import (
    PreparedImageRecord,
    audit_dataset,
    build_exact_resolution_manifest,
    build_manual_review_evidence,
    cross_split_duplicate_summary,
    expected_prepared_splits,
    load_coco_person_index,
    load_prepared_records,
    numeric_summary,
    render_audit_markdown,
    validate_coco_inputs,
    write_annotation_contact_sheet,
    write_crowd_contact_sheet,
)


ROOT = Path(__file__).resolve().parents[2]


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((48, 64, 3), value, dtype=np.uint8)
    cv2.rectangle(image, (12, 8), (36, 42), (255 - value, 30, 180), -1)
    assert cv2.imwrite(str(path), image)


def _write_coco(
    path: Path,
    image_ids: list[int],
    annotations: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": f"{image_id:012d}.jpg",
                "width": 64,
                "height": 48,
            }
            for image_id in image_ids
        ],
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "chair"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _person(annotation_id: int, image_id: int, *, crowd: int = 0) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": crowd,
        "bbox": [10, 6, 24, 34],
    }


def _prepared_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    prepared = tmp_path / "prepared"
    _write_coco(
        raw / "annotations" / "instances_train2017.json",
        [1, 2, 3],
        [_person(1, 1), _person(2, 3, crowd=1)],
    )
    _write_coco(
        raw / "annotations" / "instances_val2017.json",
        [4, 5],
        [_person(3, 4), _person(4, 5)],
    )
    for image_id in (1, 2, 3):
        _write_image(raw / "train2017" / f"{image_id:012d}.jpg", 40 + image_id)
    for image_id in (4, 5):
        _write_image(raw / "val2017" / f"{image_id:012d}.jpg", 40 + image_id)
    train_index = load_coco_person_index(
        raw / "annotations" / "instances_train2017.json"
    )
    val_index = load_coco_person_index(
        raw / "annotations" / "instances_val2017.json"
    )
    split_ids = expected_prepared_splits(train_index, val_index)
    for split, image_ids in split_ids.items():
        for image_id in image_ids:
            value = 80 if image_id in {1, 5} else 160
            image_path = prepared / "images" / split / f"{image_id:012d}.jpg"
            _write_image(image_path, value)
            label_path = prepared / "labels" / split / f"{image_id:012d}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "0 0.34375 0.47916667 0.375 0.70833333\n",
                encoding="utf-8",
            )
    return raw, prepared


def test_numeric_summary_includes_tail_percentiles() -> None:
    summary = numeric_summary([0, 10, 20, 30, 40])
    assert summary["count"] == 5
    assert summary["p50"] == 20
    assert summary["p95"] == 38


def test_raw_coco_audit_distinguishes_empty_and_excluded_people(tmp_path: Path) -> None:
    raw, _ = _prepared_fixture(tmp_path)
    index = load_coco_person_index(
        raw / "annotations" / "instances_train2017.json"
    )

    assert index.summary["total_images"] == 3
    assert index.summary["images_with_usable_person"] == 1
    assert index.summary["images_without_person_annotation"] == 1
    assert index.summary["images_with_only_excluded_person_annotations"] == 1
    assert index.summary["crowd_annotations_excluded"] == 1


def test_raw_coco_audit_warns_when_retained_image_contains_crowd(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    annotation_path = raw / "annotations" / "instances_train2017.json"
    _write_coco(
        annotation_path,
        [1],
        [_person(1, 1), _person(2, 1, crowd=1)],
    )
    _write_coco(
        raw / "annotations" / "instances_val2017.json",
        [2],
        [_person(3, 2)],
    )

    index = load_coco_person_index(annotation_path)
    report = audit_dataset(raw, tmp_path / "prepared")

    assert index.summary["images_with_crowd_annotations"] == 1
    assert index.summary["retained_images_with_normal_person_and_crowd"] == 1
    assert index.summary["retained_crowd_annotations_unlabeled"] == 1
    assert index.summary["unlabeled_crowd_positive_risk"] is True
    assert report["warnings"][0]["code"] == "unlabeled_crowd_positive_risk"
    assert report["warnings"][0]["retained_images"] == 1


def test_current_converter_skips_images_without_person_objects(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_image(source / "positive.jpg", 80)
    _write_image(source / "empty.jpg", 120)
    images = {
        1: {"id": 1, "file_name": "positive.jpg", "width": 64, "height": 48},
        2: {"id": 2, "file_name": "empty.jpg", "width": 64, "height": 48},
    }
    written = convert_split(
        source,
        images,
        {1: [{"bbox": [10, 6, 24, 34]}]},
        [1, 2],
        tmp_path / "converted",
        "train",
        None,
    )

    assert written == 1
    assert (tmp_path / "converted/images/train/positive.jpg").is_file()
    assert not (tmp_path / "converted/images/train/empty.jpg").exists()


def test_prepared_audit_detects_cross_split_duplicates_and_writes_sheet(
    tmp_path: Path,
) -> None:
    _, prepared = _prepared_fixture(tmp_path)
    records, summary = load_prepared_records(prepared)
    duplicates = cross_split_duplicate_summary(records)
    sheet = tmp_path / "contact.jpg"

    assert summary["split_statistics"]["train"]["images"] == 1
    assert duplicates["exact_cross_split_group_count"] == 1
    assert write_annotation_contact_sheet(records, sheet)
    assert cv2.imread(str(sheet)) is not None


def test_perceptual_duplicate_leakage_uses_hamming_distance(tmp_path: Path) -> None:
    records = [
        PreparedImageRecord(
            "train", "images/train/a.jpg", tmp_path / "a.jpg", 1, 1, (), "a", 0b0000
        ),
        PreparedImageRecord(
            "val", "images/val/b.jpg", tmp_path / "b.jpg", 1, 1, (), "b", 0b0011
        ),
        PreparedImageRecord(
            "test",
            "images/test/c.jpg",
            tmp_path / "c.jpg",
            1,
            1,
            (),
            "c",
            0b1111_1111,
        ),
    ]

    duplicates = cross_split_duplicate_summary(records, perceptual_distance=2)

    assert duplicates["exact_cross_split_group_count"] == 0
    assert duplicates["perceptual_hash"]["cross_split_pair_count"] == 1
    assert duplicates["perceptual_hash"]["examples"][0]["hamming_distance"] == 2


def test_perceptual_machine_evidence_is_not_truncated_and_is_clustered(
    tmp_path: Path,
) -> None:
    records = [
        PreparedImageRecord(
            "train" if index % 2 == 0 else "val",
            f"images/{'train' if index % 2 == 0 else 'val'}/{index}.jpg",
            tmp_path / f"{index}.jpg",
            1,
            1,
            (),
            f"sha-{index}",
            0,
        )
        for index in range(22)
    ]

    duplicates = cross_split_duplicate_summary(records, perceptual_distance=0)
    phash = duplicates["perceptual_hash"]

    assert phash["cross_split_pair_count"] == 121
    assert len(phash["edges"]) == 121
    assert len(phash["examples"]) == 100
    assert phash["machine_evidence_truncated"] is False
    assert phash["cluster_count"] == 1
    assert len(phash["clusters"][0]["paths"]) == 22


def test_exact_resolution_prefers_test_then_smallest_image_id(tmp_path: Path) -> None:
    records = [
        PreparedImageRecord(
            split,
            path,
            tmp_path / Path(path).name,
            1,
            1,
            (),
            "same",
            0,
        )
        for split, path in (
            ("train", "images/train/000000000001.jpg"),
            ("test", "images/test/000000000009.jpg"),
            ("test", "images/test/000000000002.jpg"),
        )
    ]

    manifest = build_exact_resolution_manifest(cross_split_duplicate_summary(records))

    assert manifest["group_count"] == 1
    assert manifest["groups"][0]["keep"] == "images/test/000000000002.jpg"
    assert manifest["control_dataset_mutated"] is False


def test_manual_review_requires_categories_and_resolved_phash_clusters() -> None:
    sheets = [{"category": "random", "path": "reports/random.jpg"}]
    clusters = [{"cluster_id": "phash-00001", "paths": ["a", "b"]}]

    pending = build_manual_review_evidence(sheets, clusters)
    complete = build_manual_review_evidence(
        sheets,
        clusters,
        {
            "category_contact_sheets": [
                {
                    "category": "random",
                    "review_status": "reviewed",
                    "notes": "checked",
                }
            ],
            "phash_clusters": [
                {
                    "cluster_id": "phash-00001",
                    "decision": "similar_not_duplicate",
                    "notes": "different scene",
                }
            ],
        },
    )

    assert pending["status"] == "pending"
    assert complete["status"] == "complete"


def test_input_validation_decodes_all_expected_images(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_coco(
        raw / "annotations" / "instances_train2017.json",
        [1],
        [_person(1, 1)],
    )
    _write_coco(
        raw / "annotations" / "instances_val2017.json",
        [2],
        [_person(2, 2)],
    )
    _write_image(raw / "train2017" / "000000000001.jpg", 80)
    _write_image(raw / "val2017" / "000000000002.jpg", 120)
    train = load_coco_person_index(
        raw / "annotations" / "instances_train2017.json"
    )
    val = load_coco_person_index(raw / "annotations" / "instances_val2017.json")

    validation = validate_coco_inputs(raw, train, val)

    assert validation["status"] == "confirmed"
    assert validation["extracted_images"]["train2017"]["decode_failure_count"] == 0
    assert validation["archives"]["train2017.zip"]["status"] == "not_available"


def test_complete_audit_fails_when_raw_image_dimensions_do_not_match(
    tmp_path: Path,
) -> None:
    raw, prepared = _prepared_fixture(tmp_path)
    wrong_size = np.zeros((12, 12, 3), dtype=np.uint8)
    assert cv2.imwrite(str(raw / "train2017" / "000000000001.jpg"), wrong_size)

    report = audit_dataset(raw, prepared)

    assert report["status"] == "failed_integrity"
    assert (
        report["input_validation"]["extracted_images"]["train2017"][
            "dimension_mismatch_count"
        ]
        == 1
    )


def test_crowd_sheet_prefers_segmentation_mask(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    normal = _person(1, 1)
    crowd = _person(2, 1, crowd=1)
    crowd["segmentation"] = {"size": [48, 64], "counts": [0, 10, 3062]}
    _write_coco(
        raw / "annotations" / "instances_train2017.json",
        [1],
        [normal, crowd],
    )
    index = load_coco_person_index(
        raw / "annotations" / "instances_train2017.json"
    )
    image_path = tmp_path / "000000000001.jpg"
    _write_image(image_path, 80)
    record = PreparedImageRecord(
        "train",
        "images/train/000000000001.jpg",
        image_path,
        64,
        48,
        ((0.34375, 0.47916667, 0.375, 0.70833333),),
        "sha",
        0,
    )

    result = write_crowd_contact_sheet(
        [record],
        {"train": index},
        tmp_path / "crowd.jpg",
    )

    assert result is not None
    assert result["segmentation_mask_regions"] == 1
    assert result["bbox_fallback_regions"] == 0


def test_complete_audit_proves_raw_empty_images_are_not_prepared(tmp_path: Path) -> None:
    raw, prepared = _prepared_fixture(tmp_path)
    report = audit_dataset(raw, prepared, contact_sheet_dir=tmp_path / "sheets")

    assert report["status"] == "complete"
    assert report["preparer_empty_image_behavior"]["negative_images_preserved"] is False
    assert report["raw_vs_prepared"]["selection_matches_current_preparer"] is True
    assert report["raw_vs_prepared"]["labels_match_current_preparer"] is True
    assert report["raw_vs_prepared"]["conversion_matches_current_preparer"] is True
    assert report["raw_vs_prepared"]["raw_images_without_usable_person"] == 2
    assert report["raw_vs_prepared"]["raw_empty_images_copied_to_prepared"] == 0
    assert "Current preparer selection" in render_audit_markdown(report)


def test_label_comparison_detects_missing_and_coordinate_mismatches(
    tmp_path: Path,
) -> None:
    raw, prepared = _prepared_fixture(tmp_path)
    train_label = next((prepared / "labels" / "train").glob("*.txt"))
    train_label.write_text("0 0.5 0.5 0.375 0.70833333\n", encoding="utf-8")
    val_label = next((prepared / "labels" / "val").glob("*.txt"))
    val_label.unlink()

    report = audit_dataset(raw, prepared)
    comparison = report["raw_vs_prepared"]
    labels = comparison["label_comparison"]

    assert comparison["selection_matches_current_preparer"] is True
    assert comparison["labels_match_current_preparer"] is False
    assert comparison["conversion_matches_current_preparer"] is False
    assert labels["missing_expected_box_count"] == 2
    assert labels["extra_actual_box_count"] == 1
    assert labels["mismatched_box_pair_count"] == 1
    assert labels["splits"]["train"]["mismatched_box_pair_count"] == 1
    assert labels["splits"]["val"]["images_with_object_count_mismatch"] == 1


def test_missing_dataset_is_reported_without_fabricated_statistics(tmp_path: Path) -> None:
    report = audit_dataset(tmp_path / "raw", tmp_path / "prepared")

    assert report["status"] == "blocked_missing_data"
    assert "raw_coco" not in report
    assert "prepared_dataset" not in report
    assert len(report["missing_inputs"]) == 3


def test_domain_manifest_schema_accepts_mixed_and_label_free_negative() -> None:
    schema = json.loads(
        (ROOT / "configs" / "domain_manifest.schema.json").read_text(encoding="utf-8")
    )
    base = {
        "schema_version": 1,
        "sample_id": "camera0-000001",
        "image_path": "images/camera0-000001.jpg",
        "source_id": "camera0",
        "source_type": "webcam",
        "scene_id": "office",
        "time_block_id": "2026-07-14T12",
        "group_id": "camera0/office/2026-07-14T12",
        "split": "unassigned",
        "annotation_status": "verified",
        "distractors": ["reflection"],
    }
    negative = {**base, "role": "hard_negative"}
    mixed = {
        **base,
        "sample_id": "camera0-000002",
        "role": "mixed",
        "label_path": "labels/camera0-000002.txt",
    }

    jsonschema.validate(negative, schema)
    jsonschema.validate(mixed, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**base, "role": "positive"}, schema)
