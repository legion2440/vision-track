from __future__ import annotations

import types
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark import (
    build_performance_payload,
    run_scenario,
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
from vision_track.lifecycle import StreamState


class FakePrimaryMetrics:
    def __init__(self) -> None:
        self.processed_frames = 0
        self.inference_latency_total_ms = 0.0
        self.end_to_end_latency_total_ms = 0.0


class FakePrimaryQueue:
    def __init__(self) -> None:
        self.received = 0
        self.dropped = 0


class FakePrimaryContext:
    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self.lock = threading.RLock()
        self.reset_count = 0
        self.tracker_token = object()
        self.counter_token = object()
        self.latest_frame = None
        self.reset_runtime()

    def reset_runtime(self) -> None:
        self.state = StreamState.CREATED
        self.metrics = FakePrimaryMetrics()
        self.queue = FakePrimaryQueue()
        self.actual_backend = None
        self.actual_device = None
        self.actual_provider = None
        self.error = None
        self.tracker_token = object()
        self.counter_token = object()
        self.latest_frame = None
        self.reset_count += 1


class FakePrimaryClock:
    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def _advance_primary_context(context: FakePrimaryContext) -> None:
    context.metrics.processed_frames += 1
    context.metrics.inference_latency_total_ms += 4.0
    context.metrics.end_to_end_latency_total_ms += 8.0
    context.queue.received += 1


def _mark_primary_ready(
    context: FakePrimaryContext,
    *,
    backend: str = "pytorch",
    device: str = "cpu",
    provider: str | None = None,
) -> None:
    context.state = StreamState.ACTIVE
    context.metrics.processed_frames = max(context.metrics.processed_frames, 1)
    context.actual_backend = backend
    context.actual_device = device
    context.actual_provider = provider


def _run_fake_primary(
    monkeypatch,
    tmp_path: Path,
    updater,
    *,
    warmup_seconds: float = 0.1,
    measured_seconds: float = 0.1,
    startup_timeout_seconds: float = 1.0,
    snapshot_times: list[float] | None = None,
    atomic_snapshot_hook=None,
) -> tuple[dict, FakePrimaryClock, object]:
    from scripts import benchmark

    clock = FakePrimaryClock()
    engine_holder: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            self.contexts_by_id: dict[str, FakePrimaryContext] = {}
            self.start_all_calls = 0
            self.stop_all_calls = 0
            self.start_times: list[float] = []
            self.stop_times: list[float] = []
            self.all_stopped_before_restart: list[bool] = []
            engine_holder["engine"] = self

        def add_stream(self, _path: str) -> str:
            stream_id = f"stream-{len(self.contexts_by_id)}"
            self.contexts_by_id[stream_id] = FakePrimaryContext(stream_id)
            return stream_id

        def start_all(self) -> None:
            self.start_all_calls += 1
            self.start_times.append(clock.now)
            if self.start_all_calls > 1:
                self.all_stopped_before_restart.append(
                    all(
                        context.state is StreamState.STOPPED
                        for context in self.contexts_by_id.values()
                    )
                )
                for context in self.contexts_by_id.values():
                    context.reset_runtime()
            updater(self, clock.now, 0.0, "start_all")

        def stop_all(self) -> None:
            self.stop_all_calls += 1
            self.stop_times.append(clock.now)
            for context in self.contexts_by_id.values():
                context.state = StreamState.STOPPED
            updater(self, clock.now, 0.0, "stop_all")

        def get(self, stream_id: str) -> FakePrimaryContext:
            return self.contexts_by_id[stream_id]

        def shutdown(self) -> None:
            pass

    def sleep(seconds: float) -> None:
        clock.now += seconds
        updater(engine_holder["engine"], clock.now, seconds, "sleep")

    monkeypatch.setattr(benchmark, "_primary_duration_failure", lambda *_args: None)
    monkeypatch.setattr(
        benchmark,
        "select_device",
        lambda force=None: DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
    )
    monkeypatch.setattr(benchmark, "create_backend", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(benchmark, "ProcessingEngine", FakeEngine)
    monkeypatch.setattr(benchmark.time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(benchmark.time, "sleep", sleep)
    if snapshot_times is not None or atomic_snapshot_hook is not None:
        original_atomic_snapshot = benchmark._snapshot_contexts_atomic
        snapshot_call = 0

        def snapshot_contexts_atomic(contexts):
            nonlocal snapshot_call
            context_list = list(contexts)
            snapshot_call += 1
            if atomic_snapshot_hook is not None:
                atomic_snapshot_hook(
                    engine_holder["engine"],
                    clock,
                    context_list,
                    snapshot_call,
                )
            timestamp, snapshots = original_atomic_snapshot(context_list)
            if snapshot_times is not None:
                snapshot_times.extend([timestamp] * len(context_list))
            return timestamp, snapshots

        monkeypatch.setattr(benchmark, "_snapshot_contexts_atomic", snapshot_contexts_atomic)

    config = types.SimpleNamespace(
        model=types.SimpleNamespace(confidence=0.35, iou=0.5, person_class_id=0),
    )
    result = run_processing_engine_scenario(
        name="primary",
        config=config,
        backend_name="pytorch",
        model_path=tmp_path / "best.pt",
        device_name=None,
        videos=[tmp_path / "a.avi", tmp_path / "b.avi"],
        image_size=640,
        resolution=(1280, 720),
        warmup_seconds=warmup_seconds,
        measured_seconds=measured_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
    )
    return result, clock, engine_holder["engine"]


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


def test_atomic_primary_snapshots_lock_in_stream_id_order(monkeypatch) -> None:
    from scripts import benchmark

    acquired: list[str] = []

    class RecordingLock:
        def __init__(self, stream_id: str) -> None:
            self.stream_id = stream_id

        def __enter__(self):
            acquired.append(self.stream_id)
            return self

        def __exit__(self, _exc_type, _exc, _traceback) -> None:
            return None

    first = FakePrimaryContext("stream-b")
    second = FakePrimaryContext("stream-a")
    first.lock = RecordingLock(first.stream_id)
    second.lock = RecordingLock(second.stream_id)
    _mark_primary_ready(first)
    _mark_primary_ready(second)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 123.0)

    timestamp, snapshots = benchmark._snapshot_contexts_atomic([first, second])

    assert timestamp == 123.0
    assert acquired == ["stream-a", "stream-b"]
    assert list(snapshots) == ["stream-a", "stream-b"]


def test_primary_atomic_start_and_end_snapshots_share_timestamps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot_times: list[float] = []

    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if seconds > 0:
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        snapshot_times=snapshot_times,
    )

    assert result["status"] == "ok"
    assert snapshot_times[0] == snapshot_times[1]
    assert snapshot_times[2] == snapshot_times[3]
    assert snapshot_times[2] > snapshot_times[0]


def test_primary_measurement_uses_atomic_start_timestamp_for_deadline_and_elapsed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if engine.start_all_calls == 2 and event == "start_all":
                context.metrics.processed_frames = 100
                context.queue.received = 100
                context.queue.dropped = 40
            if engine.start_all_calls == 2 and seconds > 0:
                _advance_primary_context(context)

    def delay_start_snapshot(_engine, clock, _contexts, snapshot_call: int) -> None:
        if snapshot_call == 1:
            clock.now += 0.2

    result, _clock, _engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        warmup_seconds=0.1,
        measured_seconds=0.1,
        atomic_snapshot_hook=delay_start_snapshot,
    )

    assert result["status"] == "ok"
    assert result["processed_frames_per_stream"] == {"stream-0": 2, "stream-1": 2}
    assert result["aggregate_fps"] == pytest.approx(40.0)
    assert result["dropped_frame_rate"] == 0.0


def test_insufficient_known_primary_video_duration_fails_clearly(monkeypatch, tmp_path) -> None:
    from scripts import benchmark

    monkeypatch.setattr(benchmark, "_known_video_frame_info", lambda _path: (30, 10.0))
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
    assert "shorter than required warmup+measured frame count" in result["reason"]


def test_direct_scenario_times_only_infer_batch(monkeypatch, tmp_path: Path) -> None:
    from scripts import benchmark

    clock = FakePrimaryClock()

    class FakeStream:
        def read(self):
            clock.now += 5.0
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            pass

    class FakeBackend:
        def load(self):
            pass

        def infer_batch(self, frames):
            clock.now += 0.2
            return [
                InferenceResult(Detections.empty(), 7.0, "pytorch", "cpu")
                for _frame in frames
            ]

    monkeypatch.setattr(benchmark, "LoopingVideo", lambda *_args: FakeStream())
    monkeypatch.setattr(benchmark, "create_backend", lambda *_args, **_kwargs: FakeBackend())
    monkeypatch.setattr(
        benchmark,
        "select_device",
        lambda force=None: DeviceInfo("cpu", "cpu", "CPU", "Fake CPU"),
    )
    monkeypatch.setattr(benchmark.time, "perf_counter", clock.perf_counter)

    result = run_scenario(
        name="direct",
        backend_name="pytorch",
        model_path=tmp_path / "best.pt",
        device_name="cpu",
        videos=[tmp_path / "a.avi", tmp_path / "b.avi"],
        image_size=640,
        resolution=(1280, 720),
        warmup_frames=0,
        measured_frames=2,
    )

    assert round(result["batch_inference_latency_ms"], 6) == 200.0
    assert result["backend_reported_latency_ms"] == 7.0
    assert round(result["aggregate_fps"], 6) == 10.0
    assert round(result["fps_per_stream"], 6) == 5.0
    assert "end_to_end_latency_ms" not in result
    assert result["measurement_scope"] == (
        "detector inference only; decoded and resized frames supplied before timing"
    )


def test_primary_readiness_waits_for_all_streams(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        first = engine.contexts_by_id.get("stream-0")
        second = engine.contexts_by_id.get("stream-1")
        if first and now >= 0.05:
            _mark_primary_ready(first)
        if second and now >= 0.15:
            _mark_primary_ready(second)
        if now > 0.15:
            for context in engine.contexts_by_id.values():
                _advance_primary_context(context)

    result, clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "ok"
    assert clock.now >= 0.35


def test_primary_rewinds_after_readiness_before_timed_warmup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot_times: list[float] = []

    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if seconds > 0:
                _advance_primary_context(context)
    result, clock, engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        warmup_seconds=0.1,
        measured_seconds=0.1,
        snapshot_times=snapshot_times,
    )

    assert result["status"] == "ok"
    assert engine.stop_all_calls == 1
    assert engine.start_all_calls == 2
    assert engine.all_stopped_before_restart == [True]
    assert snapshot_times[0] == engine.start_times[1] + 0.1


def test_primary_model_loading_time_is_excluded_from_warmup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot_times: list[float] = []

    def updater(engine, now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        if now >= 0.2:
            for context in engine.contexts_by_id.values():
                _mark_primary_ready(context)
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        warmup_seconds=0.2,
        measured_seconds=0.1,
        snapshot_times=snapshot_times,
    )

    assert result["status"] == "ok"
    assert min(snapshot_times[:2]) >= 0.4


def test_primary_phase_a_runtime_does_not_pollute_measured_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            if engine.start_all_calls == 1:
                _mark_primary_ready(context, backend="phase-a", device="phase-a-device")
                context.phase_a_tracker_token = context.tracker_token
                context.phase_a_counter_token = context.counter_token
                context.metrics.processed_frames = 100
                context.queue.received = 100
                context.queue.dropped = 90
                context.latest_frame = object()
            else:
                _mark_primary_ready(context)
                if seconds > 0:
                    _advance_primary_context(context)

    result, _clock, engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "ok"
    assert result["actual_backend"] == "pytorch"
    assert result["actual_device"] == "cpu"
    assert all(context.reset_count >= 2 for context in engine.contexts_by_id.values())
    assert all(
        context.tracker_token is not context.phase_a_tracker_token
        for context in engine.contexts_by_id.values()
    )
    assert all(
        context.counter_token is not context.phase_a_counter_token
        for context in engine.contexts_by_id.values()
    )
    assert all(value < 100 for value in result["processed_frames_per_stream"].values())
    assert result["dropped_frame_rate"] == 0.0


def test_primary_zero_frame_stream_fails_after_measured_window(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
        if seconds > 0:
            _advance_primary_context(engine.contexts_by_id["stream-0"])

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "measurement_validation"
    assert result["failed_streams"][0]["stream_id"] == "stream-1"
    assert result["failed_streams"][0]["processed_frames_delta"] == 0


def test_primary_eof_during_startup_readiness_fails(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        first = engine.contexts_by_id.get("stream-0")
        second = engine.contexts_by_id.get("stream-1")
        if first:
            _mark_primary_ready(first)
        if second:
            second.state = StreamState.EOF

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "startup_readiness"
    assert "EOF before startup readiness" in result["reason"]


def test_primary_failed_stream_fails_before_readiness(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        first = engine.contexts_by_id.get("stream-0")
        second = engine.contexts_by_id.get("stream-1")
        if first:
            _mark_primary_ready(first)
        if second:
            second.state = StreamState.FAILED
            second.error = "decode failure"

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "startup_readiness"
    assert "failed before startup readiness" in result["reason"]
    assert result["failed_streams"][0]["stream_id"] == "stream-1"


def test_primary_rewind_requires_all_streams_stopped(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            engine.contexts_by_id["stream-1"].state = StreamState.ACTIVE
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "rewind"
    assert result["failed_streams"][0]["stream_id"] == "stream-1"


@pytest.mark.parametrize("state", [StreamState.EOF, StreamState.FAILED])
def test_primary_warmup_fails_on_terminal_stream_state(
    monkeypatch,
    tmp_path: Path,
    state: StreamState,
) -> None:
    def updater(engine, now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
        if engine.start_all_calls == 2 and event == "sleep" and now >= 0.05:
            engine.contexts_by_id["stream-1"].state = state

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "warmup"
    assert result["failed_streams"][0]["stream_id"] == "stream-1"


def test_primary_warmup_deadline_requires_ready_active_streams(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def updater(engine, _now: float, _seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        _mark_primary_ready(engine.contexts_by_id["stream-0"])
        second = engine.contexts_by_id["stream-1"]
        if engine.start_all_calls == 1:
            _mark_primary_ready(second)
        else:
            second.state = StreamState.CONNECTING

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "warmup"
    assert result["failed_streams"][0]["state"] == StreamState.CONNECTING.value


@pytest.mark.parametrize(
    "state",
    [StreamState.EOF, StreamState.FAILED, StreamState.STOPPED],
)
def test_primary_measurement_fails_on_terminal_stream_state(
    monkeypatch,
    tmp_path: Path,
    state: StreamState,
) -> None:
    def updater(engine, now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if seconds > 0:
                _advance_primary_context(context)
        if engine.start_all_calls == 2 and event == "sleep" and now > 0.1:
            engine.contexts_by_id["stream-1"].state = state

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "measurement"
    assert result["failed_streams"][0]["stream_id"] == "stream-1"


@pytest.mark.parametrize(
    "state",
    [StreamState.EOF, StreamState.FAILED, StreamState.STOPPED],
)
def test_primary_terminal_state_visible_at_end_snapshot_fails_validation(
    monkeypatch,
    tmp_path: Path,
    state: StreamState,
) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if seconds > 0:
                _advance_primary_context(context)

    def transition_at_end_snapshot(engine, _clock, _contexts, snapshot_call: int) -> None:
        if snapshot_call == 2:
            engine.contexts_by_id["stream-1"].state = state

    result, _clock, _engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        atomic_snapshot_hook=transition_at_end_snapshot,
    )

    assert result["status"] == "failed"
    assert result["phase"] == "measurement_validation"
    assert result["failed_streams"][0]["stream_id"] == "stream-1"
    assert result["failed_streams"][0]["state"] == state.value


def test_primary_backend_device_disagreement_fails(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        backends = ["pytorch", "onnxruntime"]
        for index, context in enumerate(engine.contexts_by_id.values()):
            _mark_primary_ready(context, backend=backends[index])
            if seconds > 0:
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "measurement_validation"
    assert "different actual backend/device/provider" in result["reason"]


def test_primary_device_disagreement_fails(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        devices = ["cpu", "cuda:0"]
        for index, context in enumerate(engine.contexts_by_id.values()):
            _mark_primary_ready(context, device=devices[index])
            if seconds > 0:
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "measurement_validation"
    assert "different actual backend/device/provider" in result["reason"]


def test_primary_provider_disagreement_fails(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        providers = ["CPUExecutionProvider", None]
        for index, context in enumerate(engine.contexts_by_id.values()):
            _mark_primary_ready(context, provider=providers[index])
            if seconds > 0:
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "failed"
    assert result["phase"] == "measurement_validation"
    assert "different actual backend/device/provider" in result["reason"]


def test_primary_two_valid_streams_produce_ok(monkeypatch, tmp_path: Path) -> None:
    def updater(engine, _now: float, seconds: float, event: str) -> None:
        if event == "stop_all":
            return
        for context in engine.contexts_by_id.values():
            _mark_primary_ready(context)
            if seconds > 0:
                _advance_primary_context(context)

    result, _clock, _engine = _run_fake_primary(monkeypatch, tmp_path, updater)

    assert result["status"] == "ok"
    assert result["per_stream"]["stream-0"]["actual_backend"] == "pytorch"
    assert result["per_stream"]["stream-1"]["actual_device"] == "cpu"


@pytest.mark.parametrize(
    ("required_seconds", "frame_count", "should_fail"),
    [
        (3.0, 31, True),
        (3.0, 32, False),
        (3.05, 31, True),
        (3.05, 32, False),
    ],
)
def test_primary_duration_precheck_enforces_strictly_greater_duration(
    monkeypatch,
    tmp_path: Path,
    required_seconds: float,
    frame_count: int,
    should_fail: bool,
) -> None:
    from scripts import benchmark

    monkeypatch.setattr(
        benchmark,
        "_known_video_frame_info",
        lambda _path: (frame_count, 10.0),
    )

    result = benchmark._primary_duration_failure(
        [tmp_path / "video.avi"],
        required_seconds,
    )

    if should_fail:
        assert result is not None
        assert "frame count 32" in result
    else:
        assert result is None


def test_primary_duration_precheck_allows_unknown_duration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import benchmark

    monkeypatch.setattr(benchmark, "_known_video_frame_info", lambda _path: None)

    assert benchmark._primary_duration_failure([tmp_path / "video.avi"], 3.0) is None


def test_primary_readiness_timeout_fails_clearly(monkeypatch, tmp_path: Path) -> None:
    def updater(_engine, _now: float, _seconds: float, _event: str) -> None:
        pass

    result, _clock, _engine = _run_fake_primary(
        monkeypatch,
        tmp_path,
        updater,
        startup_timeout_seconds=0.1,
    )

    assert result["status"] == "failed"
    assert result["phase"] == "startup_readiness"
    assert "Startup readiness timeout" in result["reason"]


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
    assert payload["protocol"]["latency_scope"] == (
        "backend.infer wall clock from decoded image to postprocessed detections"
    )
    assert payload["protocol"]["throughput_scope"] == (
        "measured images divided by total backend.infer wall-clock time"
    )


def test_artifact_timing_uses_external_backend_infer_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from scripts import compare_artifacts

    clock = FakePrimaryClock()
    events: list[str] = []

    class FakeBackend:
        name = "pytorch"

        def load(self):
            pass

        def infer(self, _image):
            events.append("infer")
            clock.now += 0.25
            return InferenceResult(Detections.empty(), 3.0, "pytorch", "cpu")

    def read_image(_path):
        events.append("decode")
        clock.now += 5.0
        return np.zeros((10, 10, 3), dtype=np.uint8)

    def parse_label(_path, _width, _height):
        events.append("label")
        clock.now += 7.0
        return np.empty((0, 4), dtype=np.float32)

    def match_scores(_predictions, _ground_truth, *, iou_threshold):
        events.append("match")
        clock.now += 11.0
        return types.SimpleNamespace(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            true_positives=0,
            false_positives=0,
            false_negatives=0,
        )

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    image_path = tmp_path / "image.jpg"
    monkeypatch.setattr(compare_artifacts, "create_backend", lambda *_, **__: FakeBackend())
    monkeypatch.setattr(compare_artifacts, "_read_image", read_image)
    monkeypatch.setattr(compare_artifacts, "yolo_labels_to_xyxy", parse_label)
    monkeypatch.setattr(compare_artifacts, "smoke_match_scores", match_scores)
    monkeypatch.setattr(compare_artifacts.time, "perf_counter", clock.perf_counter)

    result = evaluate_artifact(
        name="fine_tuned",
        model_path=model_path,
        backend_name="pytorch",
        device=DeviceInfo("cpu", "cpu", "CPU", "PyTorch CPU"),
        image_paths=[image_path],
        label_dir=tmp_path,
        image_size=640,
        confidence=0.35,
        nms_iou=0.5,
        gt_iou=0.5,
        warmup_count=0,
    )

    assert events == ["decode", "label", "infer", "match"]
    assert result["pipeline_latency_ms"] == 250.0
    assert result["throughput_fps"] == 4.0
    assert result["backend_reported_latency_ms"] == 3.0
    assert "inference_latency_ms" not in result


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
