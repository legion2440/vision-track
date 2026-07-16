from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.device import DeviceInfo
from vision_track.detector import OnnxRuntimeBackend
from vision_track.preprocessing import to_onnx_tensor


class ImageCalibrationReader:
    def __init__(self, input_name: str, image_paths: list[Path], image_size: int) -> None:
        self.input_name = input_name
        self.image_paths = image_paths
        self.image_size = image_size
        self._iterator = iter(self.image_paths)

    def get_next(self):
        import cv2

        for path in self._iterator:
            image = cv2.imread(str(path))
            if image is None:
                continue
            tensor, _ = to_onnx_tensor(image, self.image_size)
            return {self.input_name: tensor}
        return None

    def rewind(self) -> None:
        self._iterator = iter(self.image_paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and statically quantize YOLO to ONNX INT8")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "app.yaml")
    parser.add_argument(
        "--model",
        type=Path,
        help="Defaults to the pruned checkpoint when present, otherwise best.pt",
    )
    parser.add_argument("--calibration-images", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fp32-output", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "quantization_report.json")
    args = parser.parse_args()

    import cv2
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from ultralytics import YOLO

    config = load_config(args.config)
    quantization = config.raw["quantization"]
    pruned = resolve_project_path(config.model.pruned_checkpoint)
    source = args.model or (
        pruned if pruned.exists() else resolve_project_path(config.model.checkpoint)
    )
    if not Path(source).exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    train_images = sorted(
        (
            resolve_project_path(config.raw["training"]["dataset_yaml"]).parent
            / "images"
            / "train"
        ).glob("*")
    )
    limit = args.calibration_images or int(quantization["calibration_images"])
    train_images = train_images[:limit]
    if not train_images:
        raise FileNotFoundError("No train split images found for INT8 calibration")

    model = YOLO(str(source), task="detect")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=config.model.image_size,
            opset=int(quantization["opset"]),
            simplify=True,
            dynamic=False,
            nms=False,
            device="cpu",
        )
    )
    fp32_path = args.fp32_output or (
        ROOT / "models" / "checkpoints" / "quantization_source_fp32.onnx"
    )
    fp32_path.parent.mkdir(parents=True, exist_ok=True)
    exported.replace(fp32_path)
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    reader = ImageCalibrationReader(
        session.get_inputs()[0].name,
        train_images,
        config.model.image_size,
    )
    destination = args.output or resolve_project_path(config.model.quantized_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(fp32_path),
        str(destination),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(str(destination)))
    verification_session = ort.InferenceSession(
        str(destination), providers=["CPUExecutionProvider"]
    )
    if not verification_session.get_inputs() or not verification_session.get_outputs():
        raise RuntimeError("Quantized ONNX model has no usable inputs or outputs")
    sample = cv2.imread(str(train_images[0]))
    backend = OnnxRuntimeBackend(
        destination,
        DeviceInfo("cpu", "cpu", "CPU", "ONNX Runtime CPU"),
        image_size=config.model.image_size,
        confidence=config.model.confidence,
        iou=config.model.iou,
    )
    backend.load()
    result = backend.infer(sample)
    report = {
        "status": "quantized_and_verified",
        "source_model": str(source),
        "source_model_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "fp32_onnx": str(fp32_path),
        "fp32_onnx_sha256": hashlib.sha256(fp32_path.read_bytes()).hexdigest(),
        "quantized_model": str(destination),
        "quantized_model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "calibration_split": "train",
        "calibration_images": len(train_images),
        "fp32_size_mb": fp32_path.stat().st_size / 1_000_000,
        "int8_size_mb": destination.stat().st_size / 1_000_000,
        "verification_provider": backend.actual_provider,
        "verification_latency_ms": result.latency_ms,
        "verification_detections": len(result.detections),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
