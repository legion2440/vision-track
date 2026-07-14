from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence

import cv2
import numpy as np

from vision_track.dataset_validation import IMAGE_EXTENSIONS


DEFAULT_SPLITS = ("train", "val", "test")


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": fmean(ordered),
    }


def _box_summary(boxes: Iterable[tuple[float, float, float, float]]) -> dict:
    materialized = list(boxes)
    widths = [box[2] for box in materialized]
    heights = [box[3] for box in materialized]
    areas = [box[2] * box[3] for box in materialized]
    aspect_ratios = [box[2] / box[3] for box in materialized if box[3] > 0]
    return {
        "width_fraction": numeric_summary(widths),
        "height_fraction": numeric_summary(heights),
        "area_fraction": numeric_summary(areas),
        "aspect_ratio": numeric_summary(aspect_ratios),
        "area_bands": {
            "small_lt_1pct": sum(area < 0.01 for area in areas),
            "medium_1_to_10pct": sum(0.01 <= area < 0.10 for area in areas),
            "large_ge_10pct": sum(area >= 0.10 for area in areas),
        },
        "touches_frame_edge": sum(
            x_center - width / 2 <= 0.01
            or y_center - height / 2 <= 0.01
            or x_center + width / 2 >= 0.99
            or y_center + height / 2 >= 0.99
            for x_center, y_center, width, height in materialized
        ),
    }


def _valid_coco_box(
    annotation: dict,
    image: dict,
) -> tuple[float, float, float, float] | None:
    x, y, width, height = map(float, annotation["bbox"])
    if width <= 0 or height <= 0:
        return None
    image_width = float(image["width"])
    image_height = float(image["height"])
    normalized = (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )
    if any(value < 0 or value > 1 for value in normalized):
        return None
    return normalized


@dataclass(frozen=True)
class CocoIndex:
    source_split: str
    images: dict[int, dict]
    boxes_by_image: dict[int, tuple[tuple[float, float, float, float], ...]]
    crowd_annotations_by_image: dict[int, int]
    summary: dict

    @property
    def usable_image_ids(self) -> list[int]:
        return sorted(self.boxes_by_image)


def load_coco_person_index(annotation_path: str | Path) -> CocoIndex:
    path = Path(annotation_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = {item["name"]: int(item["id"]) for item in payload["categories"]}
    if "person" not in categories:
        raise ValueError(f"COCO annotation has no person category: {path}")
    person_id = categories["person"]
    images = {int(item["id"]): item for item in payload["images"]}
    any_person_images: set[int] = set()
    boxes_by_image_lists: dict[int, list[tuple[float, float, float, float]]] = (
        defaultdict(list)
    )
    crowd_annotations = 0
    crowd_annotations_by_image: dict[int, int] = defaultdict(int)
    invalid_annotations = 0
    person_annotations = 0
    for annotation in payload["annotations"]:
        if int(annotation["category_id"]) != person_id:
            continue
        person_annotations += 1
        image_id = int(annotation["image_id"])
        any_person_images.add(image_id)
        if annotation.get("iscrowd", 0):
            crowd_annotations += 1
            crowd_annotations_by_image[image_id] += 1
            continue
        normalized = _valid_coco_box(annotation, images[image_id])
        if normalized is None:
            invalid_annotations += 1
            continue
        boxes_by_image_lists[image_id].append(normalized)

    boxes_by_image = {
        image_id: tuple(boxes)
        for image_id, boxes in boxes_by_image_lists.items()
        if boxes
    }
    resolutions = [
        (int(image["width"]), int(image["height"])) for image in images.values()
    ]
    all_boxes = [box for boxes in boxes_by_image.values() for box in boxes]
    crowd_image_ids = set(crowd_annotations_by_image)
    retained_crowd_image_ids = crowd_image_ids & boxes_by_image.keys()
    retained_crowd_annotations = sum(
        crowd_annotations_by_image[image_id]
        for image_id in retained_crowd_image_ids
    )
    source_split = path.stem.removeprefix("instances_")
    summary = {
        "source_split": source_split,
        "total_images": len(images),
        "person_annotations": person_annotations,
        "usable_person_annotations": len(all_boxes),
        "crowd_annotations_excluded": crowd_annotations,
        "images_with_crowd_annotations": len(crowd_image_ids),
        "retained_images_with_normal_person_and_crowd": len(
            retained_crowd_image_ids
        ),
        "retained_crowd_annotations_unlabeled": retained_crowd_annotations,
        "unlabeled_crowd_positive_risk": bool(retained_crowd_image_ids),
        "invalid_person_annotations_excluded": invalid_annotations,
        "images_with_any_person_annotation": len(any_person_images),
        "images_with_usable_person": len(boxes_by_image),
        "images_without_person_annotation": len(images) - len(any_person_images),
        "images_with_only_excluded_person_annotations": len(
            any_person_images - boxes_by_image.keys()
        ),
        "people_per_image_all_images": numeric_summary(
            len(boxes_by_image.get(image_id, ())) for image_id in images
        ),
        "people_per_positive_image": numeric_summary(
            len(boxes) for boxes in boxes_by_image.values()
        ),
        "image_width": numeric_summary(width for width, _ in resolutions),
        "image_height": numeric_summary(height for _, height in resolutions),
        "image_aspect_ratio": numeric_summary(
            width / height for width, height in resolutions if height > 0
        ),
        "boxes": _box_summary(all_boxes),
    }
    return CocoIndex(
        source_split,
        images,
        boxes_by_image,
        dict(crowd_annotations_by_image),
        summary,
    )


def summarize_coco_selection(index: CocoIndex, image_ids: Sequence[int]) -> dict:
    selected = [image_id for image_id in image_ids if image_id in index.boxes_by_image]
    boxes = [box for image_id in selected for box in index.boxes_by_image[image_id]]
    resolutions = [
        (int(index.images[image_id]["width"]), int(index.images[image_id]["height"]))
        for image_id in selected
    ]
    selected_crowd_images = [
        image_id
        for image_id in selected
        if image_id in index.crowd_annotations_by_image
    ]
    return {
        "images": len(selected),
        "objects": len(boxes),
        "images_with_unlabeled_crowd_regions": len(selected_crowd_images),
        "unlabeled_crowd_annotations": sum(
            index.crowd_annotations_by_image[image_id]
            for image_id in selected_crowd_images
        ),
        "people_per_image": numeric_summary(
            len(index.boxes_by_image[image_id]) for image_id in selected
        ),
        "image_width": numeric_summary(width for width, _ in resolutions),
        "image_height": numeric_summary(height for _, height in resolutions),
        "image_aspect_ratio": numeric_summary(
            width / height for width, height in resolutions if height > 0
        ),
        "boxes": _box_summary(boxes),
    }


def expected_prepared_splits(
    train_index: CocoIndex,
    val_index: CocoIndex,
    *,
    seed: int = 42,
    validation_fraction: float = 0.5,
) -> dict[str, list[int]]:
    holdout_ids = val_index.usable_image_ids
    random.Random(seed).shuffle(holdout_ids)
    cut = int(len(holdout_ids) * validation_fraction)
    return {
        "train": train_index.usable_image_ids,
        "val": holdout_ids[:cut],
        "test": holdout_ids[cut:],
    }


def _parse_yolo_boxes(
    label_path: Path | None,
) -> tuple[list[tuple[float, float, float, float]], list[str]]:
    if label_path is None or not label_path.exists():
        return [], []
    boxes: list[tuple[float, float, float, float]] = []
    issues: list[str] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            issues.append(f"{label_path}:{line_number}: expected 5 fields")
            continue
        try:
            class_value, x_center, y_center, width, height = map(float, fields)
        except ValueError:
            issues.append(f"{label_path}:{line_number}: non-numeric annotation")
            continue
        if class_value != 0 or min(x_center, y_center, width, height) < 0:
            issues.append(f"{label_path}:{line_number}: invalid person annotation")
            continue
        if max(x_center, y_center, width, height) > 1 or width <= 0 or height <= 0:
            issues.append(f"{label_path}:{line_number}: invalid normalized box")
            continue
        boxes.append((x_center, y_center, width, height))
    return boxes, issues


def perceptual_hash(image: np.ndarray) -> int:
    if image is None or image.size == 0:
        raise ValueError("Cannot hash an empty image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(np.float32(resized))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    result = 0
    for bit, coefficient in enumerate(coefficients):
        if coefficient > median:
            result |= 1 << bit
    return result


@dataclass(frozen=True)
class PreparedImageRecord:
    split: str
    relative_path: str
    path: Path
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]
    sha256: str
    phash: int


class _BKTree:
    def __init__(self) -> None:
        self.root: dict | None = None

    def add(self, value: int, record_index: int) -> None:
        if self.root is None:
            self.root = {"value": value, "records": [record_index], "children": {}}
            return
        node = self.root
        while True:
            distance = (value ^ node["value"]).bit_count()
            if distance == 0:
                node["records"].append(record_index)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {
                    "value": value,
                    "records": [record_index],
                    "children": {},
                }
                return
            node = child

    def search(self, value: int, maximum_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = (value ^ node["value"]).bit_count()
            if distance <= maximum_distance:
                matches.extend(node["records"])
            lower = distance - maximum_distance
            upper = distance + maximum_distance
            pending.extend(
                child
                for edge, child in node["children"].items()
                if lower <= edge <= upper
            )
        return matches


def cross_split_duplicate_summary(
    records: Sequence[PreparedImageRecord],
    *,
    perceptual_distance: int = 6,
    maximum_examples: int = 100,
) -> dict:
    exact_groups: dict[str, list[PreparedImageRecord]] = defaultdict(list)
    for record in records:
        exact_groups[record.sha256].append(record)
    cross_exact = [
        group
        for group in exact_groups.values()
        if len({record.split for record in group}) > 1
    ]
    exact_examples = [
        [record.relative_path for record in group]
        for group in cross_exact[:maximum_examples]
    ]

    tree = _BKTree()
    near_count = 0
    near_examples: list[dict] = []
    for index, record in enumerate(records):
        for candidate_index in tree.search(record.phash, perceptual_distance):
            candidate = records[candidate_index]
            if candidate.split == record.split or candidate.sha256 == record.sha256:
                continue
            near_count += 1
            if len(near_examples) < maximum_examples:
                near_examples.append(
                    {
                        "left": candidate.relative_path,
                        "right": record.relative_path,
                        "hamming_distance": (
                            candidate.phash ^ record.phash
                        ).bit_count(),
                    }
                )
        tree.add(record.phash, index)
    return {
        "exact_cross_split_group_count": len(cross_exact),
        "exact_cross_split_examples": exact_examples,
        "perceptual_hash": {
            "algorithm": "64-bit DCT pHash",
            "maximum_hamming_distance": perceptual_distance,
            "cross_split_pair_count": near_count,
            "examples": near_examples,
        },
    }


def _prepared_split_summary(records: Sequence[PreparedImageRecord]) -> dict:
    boxes = [box for record in records for box in record.boxes]
    return {
        "images": len(records),
        "images_without_people": sum(not record.boxes for record in records),
        "objects": len(boxes),
        "people_per_image": numeric_summary(len(record.boxes) for record in records),
        "image_width": numeric_summary(record.width for record in records),
        "image_height": numeric_summary(record.height for record in records),
        "image_aspect_ratio": numeric_summary(
            record.width / record.height for record in records if record.height > 0
        ),
        "boxes": _box_summary(boxes),
    }


def load_prepared_records(
    dataset_root: str | Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> tuple[list[PreparedImageRecord], dict]:
    root = Path(dataset_root)
    records: list[PreparedImageRecord] = []
    issues: list[str] = []
    missing_label_files = 0
    empty_label_files = 0
    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            issues.append(f"Missing image directory: {image_dir}")
            continue
        for image_path in sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_label_files += 1
                selected_label: Path | None = None
            else:
                selected_label = label_path
                if not label_path.read_text(encoding="utf-8").strip():
                    empty_label_files += 1
            boxes, label_issues = _parse_yolo_boxes(selected_label)
            issues.extend(label_issues)
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                issues.append(f"OpenCV could not decode image: {image_path}")
                continue
            height, width = image.shape[:2]
            records.append(
                PreparedImageRecord(
                    split=split,
                    relative_path=image_path.relative_to(root).as_posix(),
                    path=image_path,
                    width=width,
                    height=height,
                    boxes=tuple(boxes),
                    sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    phash=perceptual_hash(image),
                )
            )
    summary = {
        "split_statistics": {
            split: _prepared_split_summary(
                [record for record in records if record.split == split]
            )
            for split in splits
        },
        "missing_label_files": missing_label_files,
        "empty_label_files": empty_label_files,
        "issues": issues,
    }
    return records, summary


def _contact_selection(
    records: Sequence[PreparedImageRecord],
    *,
    maximum_images: int,
    seed: int,
) -> list[PreparedImageRecord]:
    if len(records) <= maximum_images:
        return list(records)
    selected: list[PreparedImageRecord] = []

    def add(candidates: Iterable[PreparedImageRecord], count: int) -> None:
        for record in candidates:
            if record not in selected:
                selected.append(record)
            if len(selected) >= count or len(selected) >= maximum_images:
                break

    add(sorted(records, key=lambda record: len(record.boxes), reverse=True), 4)
    add(
        sorted(
            (record for record in records if record.boxes),
            key=lambda record: min(box[2] * box[3] for box in record.boxes),
        ),
        8,
    )
    add(
        sorted(
            (record for record in records if record.boxes),
            key=lambda record: max(box[2] * box[3] for box in record.boxes),
            reverse=True,
        ),
        12,
    )
    remainder = [record for record in records if record not in selected]
    random.Random(seed).shuffle(remainder)
    add(remainder, maximum_images)
    return selected


def write_annotation_contact_sheet(
    records: Sequence[PreparedImageRecord],
    destination: str | Path,
    *,
    maximum_images: int = 16,
    seed: int = 42,
) -> bool:
    selected = _contact_selection(records, maximum_images=maximum_images, seed=seed)
    if not selected:
        return False
    columns = 4
    tile_width, tile_height, caption_height = 320, 240, 28
    rows = (len(selected) + columns - 1) // columns
    canvas = np.zeros(
        (rows * (tile_height + caption_height), columns * tile_width, 3),
        dtype=np.uint8,
    )
    for index, record in enumerate(selected):
        image = cv2.imread(str(record.path))
        if image is None:
            continue
        source_height, source_width = image.shape[:2]
        for x_center, y_center, width, height in record.boxes:
            x1 = max(0, int((x_center - width / 2) * source_width))
            y1 = max(0, int((y_center - height / 2) * source_height))
            x2 = min(source_width - 1, int((x_center + width / 2) * source_width))
            y2 = min(source_height - 1, int((y_center + height / 2) * source_height))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        scale = min(tile_width / source_width, tile_height / source_height)
        resized_width = max(1, int(source_width * scale))
        resized_height = max(1, int(source_height * scale))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, columns)
        tile_x = column * tile_width
        tile_y = row * (tile_height + caption_height)
        x_offset = tile_x + (tile_width - resized_width) // 2
        y_offset = tile_y + (tile_height - resized_height) // 2
        canvas[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ] = resized
        caption = f"{record.split}/{record.path.name} people={len(record.boxes)}"
        cv2.putText(
            canvas,
            caption[:46],
            (tile_x + 4, tile_y + tile_height + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output), canvas))


def compare_raw_and_prepared(
    train_index: CocoIndex,
    val_index: CocoIndex,
    expected_ids: dict[str, list[int]],
    prepared_records: Sequence[PreparedImageRecord],
    *,
    box_tolerance: float = 1e-6,
    prepared_issues: Sequence[str] = (),
) -> dict:
    indexes = {"train": train_index, "val": val_index, "test": val_index}
    expected_names = {
        split: {
            str(indexes[split].images[image_id]["file_name"])
            for image_id in image_ids
        }
        for split, image_ids in expected_ids.items()
    }
    actual_names = {
        split: {
            Path(record.relative_path).name
            for record in prepared_records
            if record.split == split
        }
        for split in DEFAULT_SPLITS
    }
    actual_records = {
        split: {
            Path(record.relative_path).name: record
            for record in prepared_records
            if record.split == split
        }
        for split in DEFAULT_SPLITS
    }
    raw_empty_names = {
        str(image["file_name"])
        for index in (train_index, val_index)
        for image_id, image in index.images.items()
        if image_id not in index.boxes_by_image
    }
    all_actual = set().union(*actual_names.values())
    label_splits: dict[str, dict] = {}
    total_missing_boxes = 0
    total_extra_boxes = 0
    total_mismatched_box_pairs = 0
    for split in DEFAULT_SPLITS:
        index = indexes[split]
        expected_by_name = {
            str(index.images[image_id]["file_name"]): index.boxes_by_image[image_id]
            for image_id in expected_ids[split]
        }
        missing_boxes = 0
        extra_boxes = 0
        mismatched_box_pairs = 0
        count_mismatch_images = 0
        box_mismatch_images = 0
        checked_images = 0
        examples: list[dict] = []
        for file_name, expected_boxes in expected_by_name.items():
            record = actual_records[split].get(file_name)
            actual_boxes = record.boxes if record is not None else ()
            if record is not None:
                checked_images += 1
            unmatched_actual = list(actual_boxes)
            unmatched_expected: list[tuple[float, float, float, float]] = []
            for expected_box in expected_boxes:
                match_index = next(
                    (
                        candidate_index
                        for candidate_index, actual_box in enumerate(unmatched_actual)
                        if all(
                            abs(expected_value - actual_value) <= box_tolerance
                            for expected_value, actual_value in zip(
                                expected_box,
                                actual_box,
                            )
                        )
                    ),
                    None,
                )
                if match_index is None:
                    unmatched_expected.append(expected_box)
                else:
                    unmatched_actual.pop(match_index)
            if len(expected_boxes) != len(actual_boxes):
                count_mismatch_images += 1
            if unmatched_expected or unmatched_actual:
                box_mismatch_images += 1
                missing_boxes += len(unmatched_expected)
                extra_boxes += len(unmatched_actual)
                mismatched_box_pairs += min(
                    len(unmatched_expected),
                    len(unmatched_actual),
                )
                if len(examples) < 100:
                    examples.append(
                        {
                            "image": file_name,
                            "image_missing": record is None,
                            "expected_object_count": len(expected_boxes),
                            "actual_object_count": len(actual_boxes),
                            "missing_expected_boxes": unmatched_expected,
                            "extra_actual_boxes": unmatched_actual,
                        }
                    )
        for file_name in actual_names[split] - expected_names[split]:
            record = actual_records[split][file_name]
            if record.boxes:
                extra_boxes += len(record.boxes)
                box_mismatch_images += 1
                if len(examples) < 100:
                    examples.append(
                        {
                            "image": file_name,
                            "image_missing": False,
                            "expected_object_count": 0,
                            "actual_object_count": len(record.boxes),
                            "missing_expected_boxes": [],
                            "extra_actual_boxes": record.boxes,
                        }
                    )
        total_missing_boxes += missing_boxes
        total_extra_boxes += extra_boxes
        total_mismatched_box_pairs += mismatched_box_pairs
        label_splits[split] = {
            "expected_images": len(expected_by_name),
            "actual_images": len(actual_records[split]),
            "images_checked": checked_images,
            "expected_object_count": sum(
                len(boxes) for boxes in expected_by_name.values()
            ),
            "actual_object_count": sum(
                len(record.boxes) for record in actual_records[split].values()
            ),
            "images_with_object_count_mismatch": count_mismatch_images,
            "images_with_box_mismatch": box_mismatch_images,
            "missing_expected_box_count": missing_boxes,
            "extra_actual_box_count": extra_boxes,
            "mismatched_box_pair_count": mismatched_box_pairs,
            "examples": examples,
        }
    selection_matches = all(
        actual_names[split] == expected_names[split] for split in DEFAULT_SPLITS
    )
    labels_match = (
        total_missing_boxes == 0
        and total_extra_boxes == 0
        and not prepared_issues
    )
    return {
        "selection_matches_current_preparer": selection_matches,
        "labels_match_current_preparer": labels_match,
        "conversion_matches_current_preparer": selection_matches and labels_match,
        "expected_counts": {
            split: len(names) for split, names in expected_names.items()
        },
        "actual_counts": {split: len(names) for split, names in actual_names.items()},
        "missing_expected_images": {
            split: sorted(expected_names[split] - actual_names[split])[:100]
            for split in DEFAULT_SPLITS
        },
        "unexpected_images": {
            split: sorted(actual_names[split] - expected_names[split])[:100]
            for split in DEFAULT_SPLITS
        },
        "raw_images_without_usable_person": len(raw_empty_names),
        "raw_empty_images_copied_to_prepared": len(raw_empty_names & all_actual),
        "prepared_split_overlap": {
            "train_val": len(actual_names["train"] & actual_names["val"]),
            "train_test": len(actual_names["train"] & actual_names["test"]),
            "val_test": len(actual_names["val"] & actual_names["test"]),
        },
        "label_comparison": {
            "coordinate_tolerance": box_tolerance,
            "label_parse_issue_count": len(prepared_issues),
            "label_parse_issue_examples": list(prepared_issues[:100]),
            "missing_expected_box_count": total_missing_boxes,
            "extra_actual_box_count": total_extra_boxes,
            "mismatched_box_pair_count": total_mismatched_box_pairs,
            "splits": label_splits,
        },
    }


def render_audit_markdown(report: dict) -> str:
    def number(value: float | int | None, digits: int = 2) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"

    def percent(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:.2f}%"

    lines = [
        "# Dataset audit",
        "",
        f"Status: **{report['status']}**",
        "",
    ]
    missing = report.get("missing_inputs", [])
    if missing:
        lines.extend(["## Missing inputs", ""])
        lines.extend(f"- `{path}`" for path in missing)
        lines.append("")
    raw = report.get("raw_coco")
    if raw:
        lines.extend(["## Raw COCO person inventory", ""])
        lines.append(
            "| Source split | Images | Usable positives | No person | "
            "Only excluded person | Usable boxes | Crowd boxes excluded | Invalid boxes excluded |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split in ("train2017", "val2017"):
            item = raw[split]
            lines.append(
                f"| {split} | {item['total_images']} | {item['images_with_usable_person']} "
                f"| {item['images_without_person_annotation']} "
                f"| {item['images_with_only_excluded_person_annotations']} "
                f"| {item['usable_person_annotations']} "
                f"| {item['crowd_annotations_excluded']} "
                f"| {item['invalid_person_annotations_excluded']} |"
            )
        lines.append("")
        lines.append(
            "| Source split | Images with crowd | Retained normal+crowd images | "
            "Unlabeled retained crowd annotations |"
        )
        lines.append("|---|---:|---:|---:|")
        for split in ("train2017", "val2017"):
            item = raw[split]
            lines.append(
                f"| {split} | {item['images_with_crowd_annotations']} "
                f"| {item['retained_images_with_normal_person_and_crowd']} "
                f"| {item['retained_crowd_annotations_unlabeled']} |"
            )
        lines.append("")
        lines.append(
            "| Source split | People/image positive p50 / p95 | Median W×H | "
            "Median image AR | Bbox area p05 / p50 / p95 | Bboxes <1% | "
            "Edge-touching boxes |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for split in ("train2017", "val2017"):
            item = raw[split]
            people = item["people_per_positive_image"]
            area = item["boxes"]["area_fraction"]
            small = item["boxes"]["area_bands"]["small_lt_1pct"]
            small_fraction = small / area["count"] if area["count"] else None
            edge_fraction = (
                item["boxes"]["touches_frame_edge"] / area["count"]
                if area["count"]
                else None
            )
            lines.append(
                f"| {split} | {number(people['p50'])} / {number(people['p95'])} "
                f"| {number(item['image_width']['p50'], 0)}×"
                f"{number(item['image_height']['p50'], 0)} "
                f"| {number(item['image_aspect_ratio']['p50'])} "
                f"| {percent(area['p05'])} / {percent(area['p50'])} / "
                f"{percent(area['p95'])} | {percent(small_fraction)} "
                f"| {percent(edge_fraction)} |"
            )
        lines.append("")
    expected = report.get("expected_current_preparer")
    if expected:
        lines.extend(["## Current preparer selection", ""])
        lines.append(
            "The current converter selects only images with at least one usable, "
            "non-crowd person box. Images without such boxes are omitted rather than "
            "preserved as negative samples."
        )
        lines.append("")
        lines.append(
            "| Split | Selected images | Objects | People/image p50 / p95 | "
            "Bbox area p50 | Images / annotations with unlabeled crowd |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for split in DEFAULT_SPLITS:
            item = expected[split]
            people = item["people_per_image"]
            lines.append(
                f"| {split} | {item['images']} | {item['objects']} "
                f"| {number(people['p50'])} / {number(people['p95'])} "
                f"| {percent(item['boxes']['area_fraction']['p50'])} "
                f"| {item['images_with_unlabeled_crowd_regions']} / "
                f"{item['unlabeled_crowd_annotations']} |"
            )
        lines.append("")
    prepared = report.get("prepared_dataset")
    if prepared:
        lines.extend(["## Prepared dataset", ""])
        lines.append("| Split | Images | Empty/negative images | Objects |")
        lines.append("|---|---:|---:|---:|")
        for split in DEFAULT_SPLITS:
            item = prepared["split_statistics"][split]
            lines.append(
                f"| {split} | {item['images']} | {item['images_without_people']} "
                f"| {item['objects']} |"
            )
        lines.append("")
    comparison = report.get("raw_vs_prepared")
    if comparison:
        labels = comparison["label_comparison"]
        lines.extend(["## Raw vs prepared integrity", ""])
        lines.append(
            f"- Image selection matches current preparer: "
            f"**{comparison['selection_matches_current_preparer']}**"
        )
        lines.append(
            f"- Labels match current preparer: "
            f"**{comparison['labels_match_current_preparer']}**"
        )
        lines.append(
            f"- Missing expected boxes: {labels['missing_expected_box_count']}"
        )
        lines.append(f"- Extra actual boxes: {labels['extra_actual_box_count']}")
        lines.append(
            f"- Coordinate-mismatched box pairs: "
            f"{labels['mismatched_box_pair_count']}"
        )
        lines.append(f"- Coordinate tolerance: {labels['coordinate_tolerance']}")
        lines.append("")
    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["## Warnings", ""])
        for warning in warnings:
            lines.append(
                f"- **{warning['code']}**: {warning['message']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Manual review still required",
            "",
            "Lighting, indoor/outdoor context, body-part distractors, screens/posters, "
            "reflections, occlusion quality, and label correctness require review of "
            "the generated contact sheets. They are not inferred from COCO metadata.",
            "",
            "## Split recommendation",
            "",
            "Keep COCO general evaluation separate from domain webcam/CCTV and "
            "hard-negative results. Assign domain samples by source/scene/time-block "
            "group, never by individual frame. Freeze the final test groups before "
            "threshold selection and use only train data for augmentation/calibration.",
            "",
        ]
    )
    unexecuted = report.get("unexecuted_checks", [])
    if unexecuted:
        lines.extend(["## Checks not executed", ""])
        lines.extend(
            f"- **{item['check']}**: {item['reason']}" for item in unexecuted
        )
        lines.append("")
    return "\n".join(lines)


def audit_dataset(
    raw_dir: str | Path,
    prepared_dir: str | Path,
    *,
    contact_sheet_dir: str | Path | None = None,
    seed: int = 42,
    validation_fraction: float = 0.5,
    perceptual_distance: int = 6,
    box_tolerance: float = 1e-6,
) -> dict:
    raw_root = Path(raw_dir)
    prepared_root = Path(prepared_dir)
    train_annotation = raw_root / "annotations" / "instances_train2017.json"
    val_annotation = raw_root / "annotations" / "instances_val2017.json"
    missing_inputs = [
        path.as_posix()
        for path in (train_annotation, val_annotation)
        if not path.is_file()
    ]
    prepared_available = all(
        (prepared_root / "images" / split).is_dir() for split in DEFAULT_SPLITS
    )
    raw_available = train_annotation.is_file() and val_annotation.is_file()
    if not prepared_available:
        missing_inputs.append((prepared_root / "images/{train,val,test}").as_posix())

    report: dict = {
        "status": (
            "partial" if raw_available or prepared_available else "blocked_missing_data"
        ),
        "raw_dir": raw_root.as_posix(),
        "prepared_dir": prepared_root.as_posix(),
        "missing_inputs": missing_inputs,
        "annotation_policy": "docs/annotation_policy.md",
        "domain_manifest_schema": "configs/domain_manifest.schema.json",
        "preparer_empty_image_behavior": {
            "implementation": "scripts/prepare_coco_person.py::convert_split",
            "selection": "skip images with no usable non-crowd person boxes",
            "negative_images_preserved": False,
        },
        "unexecuted_checks": [],
        "warnings": [],
    }
    if raw_available:
        train_index = load_coco_person_index(train_annotation)
        val_index = load_coco_person_index(val_annotation)
        report["raw_coco"] = {
            "train2017": train_index.summary,
            "val2017": val_index.summary,
        }
        retained_crowd_images = sum(
            index.summary["retained_images_with_normal_person_and_crowd"]
            for index in (train_index, val_index)
        )
        retained_crowd_annotations = sum(
            index.summary["retained_crowd_annotations_unlabeled"]
            for index in (train_index, val_index)
        )
        if retained_crowd_images:
            report["warnings"].append(
                {
                    "severity": "warning",
                    "code": "unlabeled_crowd_positive_risk",
                    "retained_images": retained_crowd_images,
                    "unlabeled_crowd_annotations": retained_crowd_annotations,
                    "message": (
                        "The current preparer retains images containing normal "
                        "person boxes and iscrowd regions, but does not write the "
                        "crowd regions to YOLO labels. Real crowded people therefore "
                        "become unlabeled background and may suppress recall. Do not "
                        "choose exclusion, ignore-region handling, or manual review "
                        "until the dataset policy decision is made."
                    ),
                }
            )
        expected_ids = expected_prepared_splits(
            train_index,
            val_index,
            seed=seed,
            validation_fraction=validation_fraction,
        )
        report["expected_current_preparer"] = {
            "seed": seed,
            "validation_fraction_of_val2017": validation_fraction,
            **{
                split: summarize_coco_selection(
                    train_index if split == "train" else val_index,
                    image_ids,
                )
                for split, image_ids in expected_ids.items()
            },
        }
    else:
        train_index = val_index = None
        expected_ids = None
        report["unexecuted_checks"].append(
            {
                "check": "raw COCO annotation inventory",
                "reason": "instances_train2017.json and/or instances_val2017.json is missing",
            }
        )

    if prepared_available:
        records, prepared_summary = load_prepared_records(prepared_root)
        prepared_summary["duplicates"] = cross_split_duplicate_summary(
            records,
            perceptual_distance=perceptual_distance,
        )
        report["prepared_dataset"] = prepared_summary
        if contact_sheet_dir is not None:
            output_dir = Path(contact_sheet_dir)
            generated = []
            for split in DEFAULT_SPLITS:
                destination = output_dir / f"{split}_annotations.jpg"
                if write_annotation_contact_sheet(
                    [record for record in records if record.split == split],
                    destination,
                    seed=seed,
                ):
                    generated.append(destination.as_posix())
            report["contact_sheets"] = generated
        if train_index and val_index and expected_ids:
            report["raw_vs_prepared"] = compare_raw_and_prepared(
                train_index,
                val_index,
                expected_ids,
                records,
                box_tolerance=box_tolerance,
                prepared_issues=prepared_summary["issues"],
            )
    else:
        report["unexecuted_checks"].extend(
            [
                {
                    "check": "prepared YOLO statistics and raw/prepared comparison",
                    "reason": "prepared images/{train,val,test} is missing",
                },
                {
                    "check": "exact and perceptual duplicate leakage",
                    "reason": "image files for the materialized splits are missing",
                },
                {
                    "check": "annotation contact sheets and manual visual review",
                    "reason": "image files for the materialized splits are missing",
                },
            ]
        )

    if not missing_inputs:
        report["status"] = "complete"
    return report
