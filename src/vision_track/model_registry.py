from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .configuration import AppConfig, resolve_project_path


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    display_name: str
    path: Path
    runtime_path: str
    backend: str
    model_type: str
    recommended_device: str
    sha256: str | None
    expected_sha256: str | None
    available: bool
    selectable: bool
    downloadable: bool = False
    default: bool = False
    recommended_runtime: bool = False
    notes: str = ""


def _sha256_if_available(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(
    *,
    model_id: str,
    display_name: str,
    path: str | Path,
    backend: str,
    model_type: str,
    recommended_device: str,
    expected_sha256: str | None = None,
    downloadable: bool = False,
    default: bool = False,
    recommended_runtime: bool = False,
    selectable_when_available: bool = True,
    notes: str = "",
) -> ModelRegistryEntry:
    resolved = resolve_project_path(path)
    available = resolved.is_file()
    runtime_path = str(resolved) if available or not downloadable else str(path)
    return ModelRegistryEntry(
        model_id=model_id,
        display_name=display_name,
        path=resolved,
        runtime_path=runtime_path,
        backend=backend,
        model_type=model_type,
        recommended_device=recommended_device,
        sha256=_sha256_if_available(resolved),
        expected_sha256=expected_sha256,
        available=available,
        selectable=(available or downloadable) and selectable_when_available,
        downloadable=downloadable,
        default=default,
        recommended_runtime=recommended_runtime,
        notes=notes,
    )


def build_model_registry(config: AppConfig) -> list[ModelRegistryEntry]:
    return [
        _entry(
            model_id="fine_tuned_n",
            display_name="Bundled fine-tuned N - portable fallback",
            path=config.model.checkpoint,
            backend="pytorch",
            model_type="fine-tuned",
            recommended_device="GPU or CPU",
            expected_sha256=(
                "ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa"
            ),
            default=True,
            notes=(
                "Bundled transfer-learning deliverable stored as "
                "models/checkpoints/best.pt."
            ),
        ),
        _entry(
            model_id="pretrained_m",
            display_name="Pretrained M - balanced",
            path="yolo26m.pt",
            backend="pytorch",
            model_type="pretrained",
            recommended_device="GPU",
            expected_sha256=(
                "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
            ),
            downloadable=True,
            notes="Downloaded on demand by Ultralytics when not present locally.",
        ),
        _entry(
            model_id="pretrained_l",
            display_name="Pretrained L - recommended GPU",
            path="yolo26l.pt",
            backend="pytorch",
            model_type="pretrained",
            recommended_device="GPU",
            recommended_runtime=True,
            expected_sha256=(
                "9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26"
            ),
            downloadable=True,
            notes="Recommended GPU option; downloaded on demand when missing.",
        ),
        _entry(
            model_id="pretrained_x",
            display_name="Pretrained X - maximum quality",
            path="yolo26x.pt",
            backend="pytorch",
            model_type="pretrained",
            recommended_device="High-end GPU",
            expected_sha256=(
                "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
            ),
            downloadable=True,
            notes="Maximum-quality option; downloaded on demand when missing.",
        ),
        _entry(
            model_id="quantized_n_int8",
            display_name="Fine-tuned N INT8 ONNX - CPU optimized",
            path=config.model.quantized_checkpoint,
            backend="onnxruntime",
            model_type="quantized",
            recommended_device="CPU",
            selectable_when_available=True,
            notes=(
                "Available only after nano INT8 quantization passes validation "
                "and writes models/checkpoints/best_quantized.onnx."
            ),
        ),
    ]


def default_model_entry(entries: list[ModelRegistryEntry]) -> ModelRegistryEntry:
    for entry in entries:
        if entry.default and entry.selectable:
            return entry
    for entry in entries:
        if entry.selectable:
            return entry
    raise RuntimeError("No selectable detector model is available")
