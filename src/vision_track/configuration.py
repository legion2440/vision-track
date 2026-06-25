from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
_ULTRALYTICS_CONFIG_ROOT = ROOT / ".ultralytics"
_ULTRALYTICS_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_ROOT))


@dataclass(frozen=True)
class ModelConfig:
    pretrained: str = "yolo26n.pt"
    checkpoint: str = "models/checkpoints/best.pt"
    pruned_checkpoint: str = "models/checkpoints/best_pruned.pt"
    quantized_checkpoint: str = "models/checkpoints/best_quantized.onnx"
    image_size: int = 640
    confidence: float = 0.35
    iou: float = 0.50
    person_class_id: int = 0


@dataclass(frozen=True)
class TrackingConfig:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 2
    frame_rate: float = 30.0
    trajectory_length: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    reconnect_attempts: int = 5
    reconnect_backoff_seconds: float = 1.0
    scheduler_idle_seconds: float = 0.005
    max_batch_size: int = 4
    max_batch_wait_ms: int = 10


@dataclass(frozen=True)
class CountingConfig:
    line_start: tuple[float, float] = (0.15, 0.55)
    line_end: tuple[float, float] = (0.85, 0.55)
    polygon: tuple[tuple[float, float], ...] = (
        (0.15, 0.20),
        (0.85, 0.20),
        (0.85, 0.90),
        (0.15, 0.90),
    )


@dataclass(frozen=True)
class AppConfig:
    name: str = "VisionTrack"
    seed: int = 42
    log_file: str = "logs/app_errors.log"
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _as_tuple_points(values: list[list[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in values)


def load_config(path: str | Path = ROOT / "configs" / "app.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    project = raw.get("project", {})
    model = ModelConfig(**raw.get("model", {}))
    tracking = TrackingConfig(**raw.get("tracking", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    counting_raw = raw.get("counting", {})
    counting = CountingConfig(
        line_start=tuple(counting_raw.get("line_start", (0.15, 0.55))),
        line_end=tuple(counting_raw.get("line_end", (0.85, 0.55))),
        polygon=_as_tuple_points(
            counting_raw.get(
                "polygon",
                [[0.15, 0.20], [0.85, 0.20], [0.85, 0.90], [0.15, 0.90]],
            )
        ),
    )
    return AppConfig(
        name=project.get("name", "VisionTrack"),
        seed=int(project.get("seed", 42)),
        log_file=project.get("log_file", "logs/app_errors.log"),
        model=model,
        tracking=tracking,
        runtime=runtime,
        counting=counting,
        raw=raw,
    )


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
