from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .detections import Detections
from .device import DeviceInfo
from .preprocessing import LetterboxInfo, restore_boxes, to_onnx_tensor


@dataclass
class InferenceResult:
    detections: Detections
    latency_ms: float
    backend: str
    device: str


class DetectorBackend(ABC):
    name: str

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceInfo,
        image_size: int = 640,
        confidence: float = 0.35,
        iou: float = 0.5,
        person_class_id: int = 0,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.person_class_id = int(person_class_id)

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.infer(frame)

    def infer(self, frame: np.ndarray) -> InferenceResult:
        return self.infer_batch([frame])[0]

    @abstractmethod
    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[InferenceResult]:
        raise NotImplementedError


class UltralyticsBackend(DetectorBackend):
    name = "pytorch"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None

    def load(self) -> None:
        from ultralytics import YOLO

        self.model = YOLO(self.model_path, task="detect")

    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[InferenceResult]:
        if self.model is None:
            self.load()
        if not frames:
            return []
        started = time.perf_counter()
        predictions = self.model.predict(
            source=list(frames),
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            device=self.device.torch_device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        per_frame_ms = elapsed_ms / len(frames)
        output: list[InferenceResult] = []
        for result in predictions:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections = Detections.empty()
            else:
                detections = Detections(
                    boxes.xyxy.detach().cpu().numpy(),
                    boxes.conf.detach().cpu().numpy(),
                    boxes.cls.detach().cpu().numpy().astype(np.int32),
                ).filter(
                    class_id=self.person_class_id,
                    confidence=self.confidence,
                )
            output.append(
                InferenceResult(
                    detections=detections,
                    latency_ms=per_frame_ms,
                    backend=self.name,
                    device=self.device.kind,
                )
            )
        return output


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = boxes.copy()
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        inter_x1 = np.maximum(x1[current], x1[remaining])
        inter_y1 = np.maximum(y1[current], y1[remaining])
        inter_x2 = np.minimum(x2[current], x2[remaining])
        inter_y2 = np.minimum(y2[current], y2[remaining])
        intersection = np.maximum(0, inter_x2 - inter_x1) * np.maximum(
            0, inter_y2 - inter_y1
        )
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int32)


def nms_detections(detections: Detections, iou_threshold: float) -> Detections:
    keep = _nms(detections.xyxy, detections.confidence, iou_threshold)
    tracker_ids = (
        detections.tracker_id[keep] if detections.tracker_id is not None else None
    )
    return Detections(
        detections.xyxy[keep],
        detections.confidence[keep],
        detections.class_id[keep],
        tracker_ids,
    )


def _postprocess_output(
    raw: np.ndarray,
    info: LetterboxInfo,
    confidence: float,
    iou: float,
    person_class_id: int,
) -> Detections:
    prediction = np.asarray(raw)
    prediction = np.squeeze(prediction)
    if prediction.size == 0:
        return Detections.empty()

    if prediction.ndim == 1:
        prediction = prediction[None, :]
    if prediction.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {np.asarray(raw).shape}")

    # End-to-end exports commonly return rows [x1, y1, x2, y2, score, class].
    if prediction.shape[1] == 6:
        boxes = prediction[:, :4]
        scores = prediction[:, 4]
        class_ids = prediction[:, 5].astype(np.int32)
        mask = (scores >= confidence) & (class_ids == person_class_id)
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    else:
        # Classic Ultralytics output is [channels, anchors], transpose when needed.
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            raise ValueError(f"Unsupported ONNX output shape: {np.asarray(raw).shape}")
        boxes = _xywh_to_xyxy(prediction[:, :4])
        class_scores = prediction[:, 4:]
        if class_scores.shape[1] == 1:
            class_ids = np.zeros(len(prediction), dtype=np.int32)
            scores = class_scores[:, 0]
        else:
            class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
            scores = class_scores[np.arange(len(prediction)), class_ids]
        mask = (scores >= confidence) & (class_ids == person_class_id)
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    keep = _nms(boxes, scores, iou)
    boxes = restore_boxes(boxes[keep], info)
    return Detections(boxes, scores[keep], class_ids[keep])


class OnnxRuntimeBackend(DetectorBackend):
    name = "onnxruntime"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.session = None
        self.input_name = ""
        self.actual_provider = ""

    def load(self) -> None:
        import onnxruntime as ort

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        available = ort.get_available_providers()
        providers: list[str] = []
        if self.device.kind == "cuda" and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.actual_provider = self.session.get_providers()[0]

    def infer_batch(self, frames: Sequence[np.ndarray]) -> list[InferenceResult]:
        if self.session is None:
            self.load()
        output: list[InferenceResult] = []
        for frame in frames:
            tensor, info = to_onnx_tensor(frame, self.image_size)
            started = time.perf_counter()
            raw_outputs = self.session.run(None, {self.input_name: tensor})
            latency_ms = (time.perf_counter() - started) * 1000.0
            detections = _postprocess_output(
                raw_outputs[0],
                info,
                self.confidence,
                self.iou,
                self.person_class_id,
            )
            output.append(
                InferenceResult(
                    detections=detections,
                    latency_ms=latency_ms,
                    backend=f"{self.name}:{self.actual_provider}",
                    device="cuda" if self.actual_provider == "CUDAExecutionProvider" else "cpu",
                )
            )
        return output


def available_backends(
    pytorch_model: str | Path,
    onnx_model: str | Path,
) -> list[str]:
    choices: list[str] = []
    if str(pytorch_model).startswith(("yolo", "http://", "https://")) or Path(pytorch_model).exists():
        choices.append("pytorch")
    if Path(onnx_model).exists():
        choices.append("onnxruntime")
    return choices


def create_backend(
    backend: str,
    model_path: str | Path,
    device: DeviceInfo,
    **kwargs,
) -> DetectorBackend:
    if backend == "pytorch":
        return UltralyticsBackend(model_path, device, **kwargs)
    if backend == "onnxruntime":
        return OnnxRuntimeBackend(model_path, device, **kwargs)
    raise ValueError(f"Unsupported detector backend: {backend}")
