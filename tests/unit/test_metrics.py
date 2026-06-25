from __future__ import annotations

import json

import numpy as np
from jsonschema import validate

from vision_track.metrics import PERFORMANCE_SCHEMA, detection_scores


def test_detection_metrics() -> None:
    scores = detection_scores(
        [np.array([[0, 0, 10, 10], [20, 20, 30, 30]])],
        [np.array([[0, 0, 10, 10]])],
    )
    assert scores.true_positives == 1
    assert scores.false_positives == 1
    assert scores.false_negatives == 0
    assert scores.precision == 0.5
    assert scores.recall == 1.0


def test_performance_report_schema() -> None:
    payload = json.loads(open("reports/performance_metrics.json", encoding="utf-8").read())
    validate(instance=payload, schema=PERFORMANCE_SCHEMA)

