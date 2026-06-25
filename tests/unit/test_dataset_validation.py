from __future__ import annotations

from vision_track.dataset_validation import validate_annotation_lines


def codes(lines: list[str]) -> set[str]:
    issues, _ = validate_annotation_lines(lines)
    return {issue.code for issue in issues}


def test_valid_person_annotation() -> None:
    issues, count = validate_annotation_lines(["0 0.5 0.5 0.2 0.3"])
    assert not [issue for issue in issues if issue.severity == "error"]
    assert count == 1


def test_annotation_failures_are_reported() -> None:
    found = codes(
        [
            "1 0.5 0.5 0.2 0.3",
            "0 1.2 0.5 0.2 0.3",
            "0 0.95 0.5 0.2 0.3",
            "0 0.5 0.5 0 0.3",
            "0 0.5 0.5 0.2 0.3",
            "0 0.5 0.5 0.2 0.3",
        ]
    )
    assert {
        "unknown_class_id",
        "coordinate_out_of_range",
        "box_outside_image",
        "non_positive_box",
        "duplicate_annotation",
    } <= found


def test_empty_annotation_is_invalid() -> None:
    assert "empty_annotation" in codes([])
