from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_id = np.asarray(self.class_id, dtype=np.int32).reshape(-1)
        if not (len(self.xyxy) == len(self.confidence) == len(self.class_id)):
            raise ValueError("Detection arrays must have equal length")
        if self.tracker_id is not None:
            self.tracker_id = np.asarray(self.tracker_id, dtype=np.int32).reshape(-1)
            if len(self.tracker_id) != len(self.xyxy):
                raise ValueError("tracker_id must match detection count")

    @classmethod
    def empty(cls) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )

    def __len__(self) -> int:
        return len(self.xyxy)

    def filter(
        self,
        *,
        class_id: int | None = None,
        confidence: float | None = None,
    ) -> "Detections":
        mask = np.ones(len(self), dtype=bool)
        if class_id is not None:
            mask &= self.class_id == class_id
        if confidence is not None:
            mask &= self.confidence >= confidence
        tracker_ids = self.tracker_id[mask] if self.tracker_id is not None else None
        return Detections(
            self.xyxy[mask],
            self.confidence[mask],
            self.class_id[mask],
            tracker_ids,
        )

    def with_tracker_ids(self, tracker_ids: np.ndarray) -> "Detections":
        return Detections(self.xyxy, self.confidence, self.class_id, tracker_ids)

    def to_supervision(self):
        import supervision as sv

        return sv.Detections(
            xyxy=self.xyxy.copy(),
            confidence=self.confidence.copy(),
            class_id=self.class_id.copy(),
            tracker_id=None if self.tracker_id is None else self.tracker_id.copy(),
        )

    @classmethod
    def from_supervision(cls, detections) -> "Detections":
        class_id = detections.class_id
        if class_id is None:
            class_id = np.zeros(len(detections), dtype=np.int32)
        confidence = detections.confidence
        if confidence is None:
            confidence = np.ones(len(detections), dtype=np.float32)
        return cls(
            detections.xyxy,
            confidence,
            class_id,
            detections.tracker_id,
        )


def composite_track_id(stream_id: str, tracker_id: int) -> tuple[str, int]:
    return stream_id, int(tracker_id)

