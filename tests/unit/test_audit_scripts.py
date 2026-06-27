from __future__ import annotations

import types
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark import build_performance_payload
from scripts.compare_artifacts import smoke_match_scores


def test_benchmark_top_level_fields_come_from_primary_scenario(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"model")
    config = types.SimpleNamespace(
        seed=42,
        model=types.SimpleNamespace(image_size=640),
    )
    primary = {
        "name": "primary_two_local_720p_pytorch_best",
        "status": "ok",
        "model": str(model_path),
        "actual_backend": "pytorch",
        "provider": None,
        "actual_device": "cuda:0",
        "gpu_name": "Test GPU",
        "streams": 2,
        "fps_per_stream": 21.0,
        "aggregate_fps": 42.0,
        "inference_latency_ms": 8.0,
        "end_to_end_latency_ms": 15.0,
        "dropped_frame_rate": 0.05,
    }
    comparison = {
        "name": "one_local_720p_cpu_best",
        "status": "ok",
        "fps_per_stream": 99.0,
        "end_to_end_latency_ms": 1.0,
    }

    payload = build_performance_payload(
        primary=primary,
        scenarios=[primary, comparison],
        detection={"detection_precision": 0.7, "detection_recall": 0.8, "f1_score": 0.75},
        config=config,
        resolution=(1280, 720),
        warmup_seconds=1.0,
        measured_seconds=2.0,
        parameter_count=123,
        flops=456.0,
    )

    assert payload["average_fps_per_stream"] == 21.0
    assert payload["fps_per_stream"] == 21.0
    assert payload["aggregate_fps"] == 42.0
    assert payload["average_latency_ms"] == 15.0
    assert payload["device"] == "cuda:0"
    assert payload["number_of_streams"] == 2


def test_unified_artifact_matching_counts_tp_fp_fn() -> None:
    predictions = [
        np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        np.empty((0, 4), dtype=np.float32),
    ]
    ground_truth = [
        np.array([[0, 0, 10, 10]], dtype=np.float32),
        np.array([[20, 20, 30, 30]], dtype=np.float32),
    ]

    scores = smoke_match_scores(predictions, ground_truth, iou_threshold=0.5)

    assert scores.true_positives == 1
    assert scores.false_positives == 1
    assert scores.false_negatives == 1
    assert scores.precision == 0.5
    assert scores.recall == 0.5
