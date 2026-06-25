from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detections import Detections


@dataclass(frozen=True)
class ZoneGeometry:
    line_start: tuple[float, float] = (0.15, 0.55)
    line_end: tuple[float, float] = (0.85, 0.55)
    polygon: tuple[tuple[float, float], ...] = (
        (0.15, 0.20),
        (0.85, 0.20),
        (0.85, 0.90),
        (0.15, 0.90),
    )

    def to_pixels(
        self, width: int, height: int
    ) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        start = (int(self.line_start[0] * width), int(self.line_start[1] * height))
        end = (int(self.line_end[0] * width), int(self.line_end[1] * height))
        polygon = np.asarray(
            [(int(x * width), int(y * height)) for x, y in self.polygon],
            dtype=np.int32,
        )
        if len(polygon) < 3:
            raise ValueError("Polygon ROI requires at least three points")
        if start == end:
            raise ValueError("Counting line cannot have zero length")
        return start, end, polygon


class ZoneCounter:
    def __init__(self, geometry: ZoneGeometry) -> None:
        self.geometry = geometry
        self.frame_shape: tuple[int, int] | None = None
        self.line_zone = None
        self.polygon_zone = None
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0
        self._counted_events: set[tuple[int, str]] = set()

    def _ensure_zones(self, frame_shape: tuple[int, int]) -> None:
        if self.frame_shape == frame_shape and self.line_zone is not None:
            return
        import supervision as sv

        height, width = frame_shape
        start, end, polygon = self.geometry.to_pixels(width, height)
        self.line_zone = sv.LineZone(
            start=sv.Point(*start),
            end=sv.Point(*end),
            minimum_crossing_threshold=2,
        )
        self.polygon_zone = sv.PolygonZone(polygon=polygon)
        self.frame_shape = frame_shape
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0
        self._counted_events.clear()

    def update(
        self,
        detections: Detections,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_zones(frame_shape)
        if detections.tracker_id is None:
            self.occupancy = 0
            return np.zeros(len(detections), dtype=bool), np.zeros(len(detections), dtype=bool)
        sv_detections = detections.to_supervision()
        crossed_in, crossed_out = self.line_zone.trigger(sv_detections)
        self.polygon_zone.trigger(sv_detections)
        self.occupancy = int(self.polygon_zone.current_count)
        for index, tracker_id in enumerate(detections.tracker_id):
            tracker_id = int(tracker_id)
            if crossed_in[index] and (tracker_id, "in") not in self._counted_events:
                self._counted_events.add((tracker_id, "in"))
                self.in_count += 1
            if crossed_out[index] and (tracker_id, "out") not in self._counted_events:
                self._counted_events.add((tracker_id, "out"))
                self.out_count += 1
        return crossed_in, crossed_out

    def reset(self) -> None:
        self.frame_shape = None
        self.line_zone = None
        self.polygon_zone = None
        self.in_count = 0
        self.out_count = 0
        self.occupancy = 0
        self._counted_events.clear()

    def reset_tracking_state(self) -> None:
        """Reset tracker-dependent zone history while preserving totals."""
        totals = self.in_count, self.out_count
        self.reset()
        self.in_count, self.out_count = totals


class CrossingGuard:
    """Small testable duplicate-event guard used by ZoneCounter."""

    def __init__(self) -> None:
        self.events: set[tuple[int, str]] = set()

    def record(self, tracker_id: int, direction: str) -> bool:
        event = (int(tracker_id), direction)
        if event in self.events:
            return False
        self.events.add(event)
        return True
