from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    original_shape: tuple[int, int]
    input_shape: tuple[int, int]
    scale: float
    pad_x: float
    pad_y: float


def letterbox(
    image: np.ndarray,
    size: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxInfo]:
    target_h, target_w = (size, size) if isinstance(size, int) else size
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image is empty")
    scale = min(target_w / width, target_h / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = target_w - resized_w, target_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, LetterboxInfo(
        original_shape=(height, width),
        input_shape=(target_h, target_w),
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
    )


def to_onnx_tensor(image: np.ndarray, image_size: int) -> tuple[np.ndarray, LetterboxInfo]:
    padded, info = letterbox(image, image_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), info


def restore_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    restored = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
    if len(restored) == 0:
        return restored
    restored[:, [0, 2]] -= info.pad_x
    restored[:, [1, 3]] -= info.pad_y
    restored /= info.scale
    height, width = info.original_shape
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, height)
    return restored

