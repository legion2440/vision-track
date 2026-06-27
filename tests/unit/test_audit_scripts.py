from __future__ import annotations

import types
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark import (
    build_performance_payload,
    run_processing_engine_scenario,
    summarize_primary_window,
)
from scripts.compare_artifacts import (
    artifact_specs,
    build_comparison_payload,
    evaluate_artifact,
    smoke_match_scores,
    validate_cpu_comparison_runtime,
)
from vision_track.detections import Detections
from vision_track.detector import InferenceResult
from vision_track.device import DeviceInfo


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


def test_primary_window_metrics_exclude_warmup_samples_and_drops() -> None:
    start = {
        "a": {
            "processed_frames": 10,
            "inference_latency_total_ms": 100.0,
            "end_to_end_latency_total_ms": 200.0,
            "queue_received": 50,
            "queue_dropped": 20,
        },
        "b": {
            "processed_frames": 5,
            "inference_latency_total_ms": 200.0,
            "end_to_end_latency_total_ms": 400.0,
            "queue_received": 30,
            "queue_dropped": 10,
        },
    }
    end = {
        "a": {
            "processed_frames": 14,
            "inference_latency_total_ms": 180.0,
            "end_to_end_latency_total_ms": 360.0,
            "queue_received": 60,
            "queue_dropped": 22,
        },
        "b": {
            "processed_frames": 7,
            "inference_latency_total_ms": 260.0,
            "end_to_end_latency_total_ms": 520.0,
            "queue_received": 38,
            "queue_dropped": 12,
        },
    }

    result = summarize_primary_window(start, end, measured_elapsed=2.0)

    assert result["per_stream"]["a"]["processed_frames"] == 4
    assert result["per_stream"]["a"]["fps"] == 2.0
    assert result["per_stream"]["a"]["average_inference_latency_ms"] == 20.0
    assert result["per_stream"]["a"]["dropped_frame_rate"] == 0.2
    assert result["per_stream"]["b"]["fps"] == 1.0
    assert result["aggregate_fps"] == 3.0
    assert result["fps_per_stream"] == 1.5
    assert result["inference_latency_ms"] == (80.0 + 60.0) / 6
    assert result["end_to_end_latency_ms"] == (160.0 + 120.0) / 6
    assert result["dropped_frame_rate"] == 4 / 18


def test_primary_window_zero_denominators_do_not_produce_nan() -> None:
    start = {
        "a": {
            "processed_frames": 1,
            "inference_latency_total_ms": 5.0,
            "end_to_end_latency_total_ms": 10.0,
            "queue_received": 2,
            "queue_dropped": 1,
        }
    }
    result = summarize_primary_window(start, start, measured_elapsed=0.0)

    assert result["aggregate_fps"] == 0.0
    assert result["inference_latency_ms"] == 0.0
    assert result["end_to_end_latency_ms"] == 0.0
    assert result["dropped_frame_rate"] == 0.0


def test_insufficient_known_primary_video_duration_fails_clearly(monkeypatch, tmp_path) -> None:
    from scripts import benchmark

    monkeypatch.setattr(benchmark, "_known_video_duration_seconds", lambda _path: 1.0)
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(confidence=0.35, iou=0.5, person_class_id=0),
    )

    result = run_processing_engine_scenario(
        name="primary",
        config=config,
        backend_name="pytorch",
        model_path=tmp_path / "best.pt",
        device_name=None,
        videos=[tmp_path / "video.avi"],
        image_size=640,
        resolution=(1280, 720),
        warmup_seconds=1.0,
        measured_seconds=2.0,
    )

    assert result["status"] == "failed"
    assert "shorter than required" in result["reason"]


def test_artifact_specs_use_same_forced_cpu_device() -> None:
    device = DeviceInfo("cpu", "cpu", "CPU", "PyTorch CPU")
    config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            checkpoint="models/checkpoints/best.pt",
            pruned_checkpoint="models/checkpoints/best_pruned.pt",
            quantized_checkpoint="models/checkpoints/best_quantized.onnx",
        )
    )

    specs = artifact_specs(config, device)

    assert {id(item[3]) for item in specs} == {id(device)}
    assert {item[3].kind for item in specs} == {"cpu"}


def test_onnx_cpu_provider_is_accepted_for_normalized_comparison() -> None:
    validate_cpu_comparison_runtime(
        backend_name="onnxruntime",
        result=InferenceResult(
            Detections.empty(),
            1.0,
            "onnxruntime",
            "cpu",
            "CPUExecutionProvider",
        ),
    )


def test_onnx_cuda_provider_is_rejected_for_normalized_comparison() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="CPUExecutionProvider"):
        validate_cpu_comparison_runtime(
            backend_name="onnxruntime",
            result=InferenceResult(
                Detections.empty(),
                1.0,
                "onnxruntime",
                "cuda",
                "CUDAExecutionProvider",
            ),
        )


def test_artifact_protocol_contains_comparison_device_cpu() -> None:
    payload = build_comparison_payload(
        split="val",
        image_size=640,
        confidence=0.35,
        nms_iou=0.5,
        gt_iou=0.5,
        image_limit=10,
        warmup_count=3,
        measured_image_count=10,
        comparison_device="cpu",
        test_isolation_acknowledged=False,
        models=[],
    )

    assert payload["protocol"]["comparison_device"] == "cpu"


def test_throughput_result_fields_identify_actual_device_and_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import compare_artifacts

    class FakeBackend:
        name = "onnxruntime"

        def load(self):
            pass

        def infer(self, _image):
            return InferenceResult(
                Detections.empty(),
                4.0,
                "onnxruntime",
                "cpu",
                "CPUExecutionProvider",
            )

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    image_path = tmp_path / "image.jpg"
    monkeypatch.setattr(compare_artifacts, "create_backend", lambda *_, **__: FakeBackend())
    monkeypatch.setattr(
        compare_artifacts,
        "_read_image",
        lambda _path: np.zeros((10, 10, 3), dtype=np.uint8),
    )

    result = evaluate_artifact(
        name="quantized_int8",
        model_path=model_path,
        backend_name="onnxruntime",
        device=DeviceInfo("cpu", "cpu", "CPU", "PyTorch CPU"),
        image_paths=[image_path],
        label_dir=tmp_path,
        image_size=640,
        confidence=0.35,
        nms_iou=0.5,
        gt_iou=0.5,
        warmup_count=0,
    )

    assert result["device"] == "cpu"
    assert result["provider"] == "CPUExecutionProvider"
    assert result["throughput_fps"] is not None
