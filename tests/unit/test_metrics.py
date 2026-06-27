from __future__ import annotations

import json

import numpy as np
from jsonschema import validate

from vision_track.context import StreamMetrics
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


def test_stream_metrics_keep_ewma_and_cumulative_totals(monkeypatch) -> None:
    import vision_track.context as context_module

    times = iter([10.0, 11.0])
    monkeypatch.setattr(context_module, "perf_counter", lambda: next(times))
    metrics = StreamMetrics()

    metrics.update(5.0, captured_at=9.9)
    metrics.update(15.0, captured_at=10.8)

    assert metrics.processed_frames == 2
    assert metrics.inference_latency_total_ms == 20.0
    assert round(metrics.end_to_end_latency_total_ms, 6) == 300.0
    assert metrics.inference_latency_ms != metrics.inference_latency_total_ms
