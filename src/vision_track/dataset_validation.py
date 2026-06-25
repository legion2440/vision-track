from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class DatasetValidationReport:
    valid: bool
    image_count: int
    annotation_count: int
    object_count: int
    split_counts: dict[str, int]
    issues: list[ValidationIssue]

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "object_count": self.object_count,
            "split_counts": self.split_counts,
            "issue_counts": dict(Counter(issue.code for issue in self.issues)),
            "issues": [asdict(issue) for issue in self.issues],
        }


def validate_annotation_lines(
    lines: Iterable[str],
    *,
    path: str = "<memory>",
    allowed_class_ids: set[int] | None = None,
    minimum_area: float = 0.0001,
) -> tuple[list[ValidationIssue], int]:
    allowed = allowed_class_ids or {0}
    issues: list[ValidationIssue] = []
    normalized = [line.strip() for line in lines if line.strip()]
    if not normalized:
        issues.append(
            ValidationIssue("error", "empty_annotation", path, "Annotation has no objects")
        )
        return issues, 0
    seen: set[tuple[float, ...]] = set()
    object_count = 0
    for line_number, line in enumerate(normalized, start=1):
        parts = line.split()
        if len(parts) != 5:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_field_count",
                    path,
                    f"Line {line_number}: expected class x_center y_center width height",
                )
            )
            continue
        try:
            class_value = float(parts[0])
            values = tuple(float(value) for value in parts[1:])
        except ValueError:
            issues.append(
                ValidationIssue(
                    "error",
                    "non_numeric_annotation",
                    path,
                    f"Line {line_number}: annotation contains non-numeric values",
                )
            )
            continue
        class_id = int(class_value)
        if class_value != class_id or class_id not in allowed:
            issues.append(
                ValidationIssue(
                    "error",
                    "unknown_class_id",
                    path,
                    f"Line {line_number}: unsupported class ID {parts[0]}",
                )
            )
        x_center, y_center, width, height = values
        if any(value < 0 or value > 1 for value in values):
            issues.append(
                ValidationIssue(
                    "error",
                    "coordinate_out_of_range",
                    path,
                    f"Line {line_number}: normalized coordinates must be in [0, 1]",
                )
            )
        if (
            x_center - width / 2 < 0
            or y_center - height / 2 < 0
            or x_center + width / 2 > 1
            or y_center + height / 2 > 1
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "box_outside_image",
                    path,
                    f"Line {line_number}: bounding box extends outside the image",
                )
            )
        if width <= 0 or height <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "non_positive_box",
                    path,
                    f"Line {line_number}: box width and height must be positive",
                )
            )
        if width * height < minimum_area:
            issues.append(
                ValidationIssue(
                    "warning",
                    "tiny_object",
                    path,
                    f"Line {line_number}: normalized box area {width * height:.8f}",
                )
            )
        record = (float(class_id), x_center, y_center, width, height)
        if record in seen:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_annotation",
                    path,
                    f"Line {line_number}: duplicate annotation",
                )
            )
        seen.add(record)
        object_count += 1
    return issues, object_count


def validate_yolo_dataset(
    dataset_root: str | Path,
    splits: tuple[str, ...] = ("train", "val", "test"),
    *,
    allowed_class_ids: set[int] | None = None,
    minimum_area: float = 0.0001,
) -> DatasetValidationReport:
    root = Path(dataset_root)
    issues: list[ValidationIssue] = []
    image_count = annotation_count = object_count = 0
    split_counts: dict[str, int] = {}

    for split in splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            issues.append(
                ValidationIssue("error", "missing_image_directory", str(image_dir), "Missing split image directory")
            )
            split_counts[split] = 0
            continue
        if not label_dir.is_dir():
            issues.append(
                ValidationIssue("error", "missing_label_directory", str(label_dir), "Missing split label directory")
            )
            split_counts[split] = 0
            continue
        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        split_counts[split] = len(images)
        image_count += len(images)
        annotation_count += len(labels)

        for stem in sorted(images.keys() - labels.keys()):
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_annotation",
                    str(images[stem]),
                    "Image has no matching annotation file",
                )
            )
        for stem in sorted(labels.keys() - images.keys()):
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_image",
                    str(labels[stem]),
                    "Annotation has no matching image",
                )
            )
        for stem in sorted(images.keys() & labels.keys()):
            image = cv2.imread(str(images[stem]))
            if image is None or image.size == 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "corrupt_image",
                        str(images[stem]),
                        "OpenCV could not decode image",
                    )
                )
            try:
                lines = labels[stem].read_text(encoding="utf-8").splitlines()
            except UnicodeError as exc:
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_annotation_encoding",
                        str(labels[stem]),
                        str(exc),
                    )
                )
                continue
            annotation_issues, count = validate_annotation_lines(
                lines,
                path=str(labels[stem]),
                allowed_class_ids=allowed_class_ids,
                minimum_area=minimum_area,
            )
            issues.extend(annotation_issues)
            object_count += count

    valid = not any(issue.severity == "error" for issue in issues)
    return DatasetValidationReport(
        valid=valid,
        image_count=image_count,
        annotation_count=annotation_count,
        object_count=object_count,
        split_counts=split_counts,
        issues=issues,
    )
