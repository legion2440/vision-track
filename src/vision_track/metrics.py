from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import numpy as np


PERFORMANCE_SCHEMA = {
    "type": "object",
    "required": [
        "detection_precision",
        "detection_recall",
        "f1_score",
        "average_fps_per_stream",
        "average_latency_ms",
    ],
    "properties": {
        "detection_precision": {"type": ["number", "null"]},
        "detection_recall": {"type": ["number", "null"]},
        "f1_score": {"type": ["number", "null"]},
        "average_fps_per_stream": {"type": ["number", "null"]},
        "average_latency_ms": {"type": ["number", "null"]},
    },
}


@dataclass(frozen=True)
class DetectionScores:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0, bottom_right - top_left)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def detection_scores(
    predictions: Iterable[np.ndarray],
    ground_truth: Iterable[np.ndarray],
    iou_threshold: float = 0.5,
) -> DetectionScores:
    tp = fp = fn = 0
    pairs = list(zip(predictions, ground_truth))
    for predicted, expected in pairs:
        predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 4)
        expected = np.asarray(expected, dtype=np.float32).reshape(-1, 4)
        ious = box_iou_matrix(predicted, expected)
        matches: list[tuple[float, int, int]] = []
        for pred_index, gt_index in np.argwhere(ious >= iou_threshold):
            matches.append((float(ious[pred_index, gt_index]), int(pred_index), int(gt_index)))
        matched_pred: set[int] = set()
        matched_gt: set[int] = set()
        for _, pred_index, gt_index in sorted(matches, reverse=True):
            if pred_index in matched_pred or gt_index in matched_gt:
                continue
            matched_pred.add(pred_index)
            matched_gt.add(gt_index)
        tp += len(matched_pred)
        fp += len(predicted) - len(matched_pred)
        fn += len(expected) - len(matched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return DetectionScores(precision, recall, f1, tp, fp, fn)


def software_versions(packages: Iterable[str]) -> dict[str, str]:
    result = {"python": sys.version.split()[0], "os": platform.platform()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def write_performance_report(path: str | Path, payload: dict) -> None:
    from jsonschema import validate

    validate(instance=payload, schema=PERFORMANCE_SCHEMA)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def scores_dict(scores: DetectionScores) -> dict:
    return asdict(scores)
