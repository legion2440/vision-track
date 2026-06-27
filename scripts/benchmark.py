from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.detector import create_backend
from vision_track.device import select_device
from vision_track.engine import ProcessingEngine
from vision_track.lifecycle import StreamState
from vision_track.metrics import software_versions, write_performance_report


PRIMARY_SCENARIO_NAME = "primary_two_local_720p_pytorch_best"


class LoopingVideo:
    def __init__(self, path: Path, resolution: tuple[int, int]) -> None:
        self.path = path
        self.resolution = resolution
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise OSError(f"Unable to open benchmark video: {path}")

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok or frame is None:
            raise OSError(f"Unable to decode benchmark video: {self.path}")
        return cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        self.capture.release()


def run_scenario(
    *,
    name: str,
    backend_name: str,
    model_path: Path,
    device_name: str,
    videos: list[Path],
    image_size: int,
    resolution: tuple[int, int],
    warmup_frames: int,
    measured_frames: int,
) -> dict:
    device = select_device(force=device_name)
    streams = [LoopingVideo(path, resolution) for path in videos]
    backend = create_backend(
        backend_name,
        model_path,
        device,
        image_size=image_size,
        confidence=0.35,
        iou=0.5,
        person_class_id=0,
    )
    try:
        backend.load()
        for _ in range(warmup_frames):
            backend.infer_batch([stream.read() for stream in streams])
        inference_latencies: list[float] = []
        end_to_end_latencies: list[float] = []
        started = time.perf_counter()
        for _ in range(measured_frames):
            frame_started = time.perf_counter()
            results = backend.infer_batch([stream.read() for stream in streams])
            end_to_end_latencies.append(
                (time.perf_counter() - frame_started) * 1000.0 / len(streams)
            )
            inference_latencies.extend(result.latency_ms for result in results)
        elapsed = time.perf_counter() - started
        aggregate_fps = measured_frames * len(streams) / elapsed
        return {
            "name": name,
            "status": "ok",
            "model": str(model_path),
            "backend": backend_name,
            "actual_backend": results[0].backend,
            "provider": results[0].provider,
            "requested_device": device_name,
            "actual_device": results[0].device,
            "streams": len(streams),
            "resolution": list(resolution),
            "image_size": image_size,
            "warmup_frames": warmup_frames,
            "measured_frames": measured_frames,
            "inference_latency_ms": statistics.fmean(inference_latencies),
            "end_to_end_latency_ms": statistics.fmean(end_to_end_latencies),
            "aggregate_fps": aggregate_fps,
            "fps_per_stream": aggregate_fps / len(streams),
        }
    finally:
        for stream in streams:
            stream.close()


def skipped(name: str, reason: str) -> dict:
    return {"name": name, "status": "skipped", "reason": reason}


def _capture_fps(capture: cv2.VideoCapture) -> float:
    value = float(capture.get(cv2.CAP_PROP_FPS))
    return value if np.isfinite(value) and 0 < value <= 240 else 30.0


def prepare_resized_video(
    source: Path,
    destination: Path,
    resolution: tuple[int, int],
) -> Path:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"Unable to open benchmark video: {source}")
    try:
        fps = _capture_fps(capture)
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            resolution,
        )
        if not writer.isOpened():
            raise OSError(f"Unable to create resized benchmark video: {destination}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                writer.write(cv2.resize(frame, resolution, interpolation=cv2.INTER_LINEAR))
        finally:
            writer.release()
    finally:
        capture.release()
    return destination


def run_processing_engine_scenario(
    *,
    name: str,
    config,
    backend_name: str,
    model_path: Path,
    device_name: str | None,
    videos: list[Path],
    image_size: int,
    resolution: tuple[int, int],
    warmup_seconds: float,
    measured_seconds: float,
) -> dict:
    device = select_device(force=device_name) if device_name else select_device()
    detector = create_backend(
        backend_name,
        model_path,
        device,
        image_size=image_size,
        confidence=config.model.confidence,
        iou=config.model.iou,
        person_class_id=config.model.person_class_id,
    )
    engine = ProcessingEngine(config, device=device, detector=detector)
    stream_ids = [engine.add_stream(str(path)) for path in videos]
    try:
        engine.start_all()
        warmup_deadline = time.perf_counter() + warmup_seconds
        while time.perf_counter() < warmup_deadline:
            time.sleep(0.05)
        start_frames = {
            stream_id: engine.get(stream_id).metrics.processed_frames
            for stream_id in stream_ids
        }
        started = time.perf_counter()
        deadline = started + measured_seconds
        while time.perf_counter() < deadline:
            time.sleep(0.05)
        elapsed = time.perf_counter() - started
        contexts = [engine.get(stream_id) for stream_id in stream_ids]
        processed = {
            context.stream_id: max(
                0,
                context.metrics.processed_frames - start_frames[context.stream_id],
            )
            for context in contexts
        }
        total_processed = sum(processed.values())
        if total_processed == 0:
            return {
                "name": name,
                "status": "failed",
                "reason": "No frames were processed during the measured window",
                "model": str(model_path),
                "backend": backend_name,
                "streams": len(stream_ids),
                "resolution": list(resolution),
            }
        inference_latencies = [
            context.metrics.inference_latency_ms
            for context in contexts
            if context.metrics.processed_frames > 0
        ]
        end_to_end_latencies = [
            context.metrics.end_to_end_latency_ms
            for context in contexts
            if context.metrics.processed_frames > 0
        ]
        dropped_rates = [context.queue.dropped_rate for context in contexts]
        first_context = contexts[0]
        aggregate_fps = total_processed / elapsed if elapsed > 0 else 0.0
        return {
            "name": name,
            "status": "ok",
            "model": str(model_path),
            "backend": backend_name,
            "actual_backend": first_context.actual_backend or detector.name,
            "provider": first_context.actual_provider,
            "requested_device": device.kind,
            "actual_device": first_context.actual_device or device.torch_device,
            "gpu_name": device.name if device.kind == "cuda" else None,
            "streams": len(stream_ids),
            "resolution": list(resolution),
            "image_size": image_size,
            "warmup_seconds": warmup_seconds,
            "measured_seconds": measured_seconds,
            "processed_frames_per_stream": processed,
            "tracking_enabled": True,
            "counting_enabled": True,
            "rendering_enabled": True,
            "inference_latency_ms": statistics.fmean(inference_latencies),
            "end_to_end_latency_ms": statistics.fmean(end_to_end_latencies),
            "dropped_frame_rate": statistics.fmean(dropped_rates),
            "aggregate_fps": aggregate_fps,
            "fps_per_stream": aggregate_fps / len(stream_ids),
        }
    finally:
        engine.shutdown()


def run_broken_source_isolation(
    *,
    config,
    model_path: Path,
    backend_name: str,
    working_video: Path,
) -> dict:
    device = select_device(force="cpu")
    detector = create_backend(
        backend_name,
        model_path,
        device,
        image_size=config.model.image_size,
        confidence=config.model.confidence,
        iou=config.model.iou,
        person_class_id=0,
    )
    engine = ProcessingEngine(config, device=device, detector=detector)
    working_id = engine.add_stream(str(working_video))
    broken_id = engine.add_stream(
        str(ROOT / "data" / "demo" / "missing-video.mp4")
    )
    try:
        engine.start_all()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            working = engine.get(working_id)
            broken = engine.get(broken_id)
            if (
                working.metrics.processed_frames > 0
                and broken.state is StreamState.FAILED
            ):
                return {
                    "name": "working_source_plus_broken_source",
                    "status": "ok",
                    "working_frames": working.metrics.processed_frames,
                    "working_state": working.state.value,
                    "broken_state": broken.state.value,
                    "dropped_frame_rate": working.queue.dropped_rate,
                }
            time.sleep(0.05)
        return {
            "name": "working_source_plus_broken_source",
            "status": "failed",
            "reason": "Isolation condition was not reached within 15 seconds",
        }
    finally:
        engine.shutdown()


def pytorch_model_statistics(model_path: Path) -> tuple[int | None, float | None]:
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path), task="detect")
        _, parameters, _, flops = model.info(verbose=False)
        return int(parameters), float(flops)
    except Exception:
        return None, None


def build_performance_payload(
    *,
    primary: dict,
    scenarios: list[dict],
    detection: dict,
    config,
    resolution: tuple[int, int],
    warmup_seconds: float,
    measured_seconds: float,
    parameter_count: int | None,
    flops: float | None,
) -> dict:
    primary_ok = primary.get("status") == "ok"
    return {
        "status": "measured" if primary_ok else "incomplete",
        "detection_precision": detection.get("detection_precision"),
        "detection_recall": detection.get("detection_recall"),
        "f1_score": detection.get("f1_score"),
        "average_fps_per_stream": primary.get("fps_per_stream") if primary_ok else None,
        "average_latency_ms": primary.get("end_to_end_latency_ms") if primary_ok else None,
        "mAP50": detection.get("mAP50"),
        "mAP50_95": detection.get("mAP50_95"),
        "model_name": primary.get("model") if primary_ok else None,
        "backend": primary.get("actual_backend") if primary_ok else None,
        "provider": primary.get("provider") if primary_ok else None,
        "device": primary.get("actual_device") if primary_ok else None,
        "gpu_name": primary.get("gpu_name") if primary_ok else None,
        "input_video_resolution": list(resolution),
        "inference_image_size": config.model.image_size,
        "number_of_streams": primary.get("streams") if primary_ok else None,
        "model_size_mb": Path(primary["model"]).stat().st_size / 1_000_000
        if primary_ok and primary.get("model") and Path(primary["model"]).exists()
        else None,
        "parameter_count": parameter_count,
        "flops": flops,
        "inference_latency_ms": primary.get("inference_latency_ms") if primary_ok else None,
        "end_to_end_latency_ms": primary.get("end_to_end_latency_ms") if primary_ok else None,
        "fps_per_stream": primary.get("fps_per_stream") if primary_ok else None,
        "aggregate_fps": primary.get("aggregate_fps") if primary_ok else None,
        "dropped_frame_rate": primary.get("dropped_frame_rate") if primary_ok else None,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "seed": config.seed,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
        },
        "software_versions": software_versions(
            [
                "torch",
                "ultralytics",
                "supervision",
                "opencv-python",
                "onnxruntime",
                "numpy",
            ]
        ),
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible VisionTrack benchmark")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--warmup-duration", type=float)
    parser.add_argument(
        "--detection-metrics",
        type=Path,
        default=ROOT / "reports" / "best_test_metrics.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = config.raw["benchmark"]
    resolution = tuple(int(value) for value in benchmark["resolution"])
    warmup = args.warmup or int(benchmark["warmup_frames"])
    frames = args.frames or int(benchmark["measured_frames"])
    warmup_seconds = args.warmup_duration or float(
        benchmark.get("warmup_duration_seconds", warmup / 30.0)
    )
    measured_seconds = args.duration or float(
        benchmark.get("measured_duration_seconds", frames / 30.0)
    )
    videos = [path.resolve() for path in args.videos]
    for path in videos:
        if not path.is_file():
            raise FileNotFoundError(path)

    best = resolve_project_path(config.model.checkpoint)
    pruned = resolve_project_path(config.model.pruned_checkpoint)
    quantized = resolve_project_path(config.model.quantized_checkpoint)
    scenarios: list[dict] = []
    primary: dict
    if best.exists():
        two_videos = [videos[0], videos[1] if len(videos) > 1 else videos[0]]
        with tempfile.TemporaryDirectory(prefix="visiontrack-benchmark-") as tmp:
            tmp_dir = Path(tmp)
            prepared = [
                prepare_resized_video(
                    path,
                    tmp_dir / f"stream-{index}-{resolution[0]}x{resolution[1]}.avi",
                    resolution,
                )
                for index, path in enumerate(two_videos)
            ]
            primary = run_processing_engine_scenario(
                name=PRIMARY_SCENARIO_NAME,
                config=config,
                backend_name="pytorch",
                model_path=best,
                device_name=None,
                videos=prepared,
                image_size=config.model.image_size,
                resolution=resolution,
                warmup_seconds=warmup_seconds,
                measured_seconds=measured_seconds,
            )
        scenarios.append(primary)
        scenarios.append(
            run_scenario(
                name="one_local_720p_cpu_best",
                backend_name="pytorch",
                model_path=best,
                device_name="cpu",
                videos=[videos[0]],
                image_size=config.model.image_size,
                resolution=resolution,
                warmup_frames=warmup,
                measured_frames=frames,
            )
        )
        scenarios.append(
            run_scenario(
                name="two_local_720p_cpu_best",
                backend_name="pytorch",
                model_path=best,
                device_name="cpu",
                videos=two_videos,
                image_size=config.model.image_size,
                resolution=resolution,
                warmup_frames=warmup,
                measured_frames=frames,
            )
        )
        auto_device = select_device()
        if auto_device.kind != "cpu":
            scenarios.append(
                run_scenario(
                    name=f"one_local_720p_{auto_device.kind}_best",
                    backend_name="pytorch",
                    model_path=best,
                    device_name=auto_device.kind,
                    videos=[videos[0]],
                    image_size=config.model.image_size,
                    resolution=resolution,
                    warmup_frames=warmup,
                    measured_frames=frames,
                )
            )
        else:
            scenarios.extend(
                [
                    skipped("one_local_720p_cuda_best", "CUDA unavailable"),
                    skipped("one_local_720p_mps_best", "MPS unavailable"),
                ]
            )
    else:
        primary = skipped(PRIMARY_SCENARIO_NAME, f"Missing artifact: {best}")
        scenarios.append(primary)
        scenarios.append(skipped("best.pt scenarios", f"Missing artifact: {best}"))

    if pruned.exists():
        scenarios.append(
            run_scenario(
                name="one_local_720p_cpu_pruned",
                backend_name="pytorch",
                model_path=pruned,
                device_name="cpu",
                videos=[videos[0]],
                image_size=config.model.image_size,
                resolution=resolution,
                warmup_frames=warmup,
                measured_frames=frames,
            )
        )
    else:
        scenarios.append(skipped("best_pruned.pt", f"Missing artifact: {pruned}"))
    if quantized.exists():
        scenarios.append(
            run_scenario(
                name="one_local_720p_cpu_int8_onnx",
                backend_name="onnxruntime",
                model_path=quantized,
                device_name="cpu",
                videos=[videos[0]],
                image_size=config.model.image_size,
                resolution=resolution,
                warmup_frames=warmup,
                measured_frames=frames,
            )
        )
    else:
        scenarios.append(skipped("best_quantized.onnx", f"Missing artifact: {quantized}"))

    isolation_model = best if best.exists() else quantized
    if isolation_model.exists():
        scenarios.append(
            run_broken_source_isolation(
                config=config,
                model_path=isolation_model,
                backend_name="pytorch" if isolation_model.suffix == ".pt" else "onnxruntime",
                working_video=videos[0],
            )
        )
    else:
        scenarios.append(
            skipped(
                "working_source_plus_broken_source",
                "No trained PyTorch or ONNX artifact is available",
            )
        )

    parameter_count, flops = (
        pytorch_model_statistics(Path(primary["model"]))
        if primary.get("status") == "ok"
        and primary.get("model")
        and Path(primary["model"]).suffix == ".pt"
        else (None, None)
    )
    detection = {}
    if args.detection_metrics.exists():
        detection = json.loads(args.detection_metrics.read_text(encoding="utf-8"))
    payload = build_performance_payload(
        primary=primary,
        scenarios=scenarios,
        detection=detection,
        config=config,
        resolution=resolution,
        warmup_seconds=warmup_seconds,
        measured_seconds=measured_seconds,
        parameter_count=parameter_count,
        flops=flops,
    )
    output = ROOT / "reports" / "performance_metrics.json"
    write_performance_report(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
