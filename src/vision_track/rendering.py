from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .counting import ZoneGeometry
from .detections import Detections


def _draw_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(y, height + baseline + 2)
    cv2.rectangle(frame, (x, y - height - baseline - 4), (x + width + 4, y), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline - 2), font, scale, (255, 255, 255), thickness)


def render_frame(
    frame: np.ndarray,
    detections: Detections,
    *,
    trajectories: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    geometry: ZoneGeometry | None = None,
    in_count: int = 0,
    out_count: int = 0,
    occupancy: int = 0,
    show_detections: bool = True,
    show_tracking: bool = True,
    show_counting: bool = True,
) -> np.ndarray:
    output = frame.copy()
    if show_counting and geometry is not None:
        height, width = output.shape[:2]
        start, end, polygon = geometry.to_pixels(width, height)
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (255, 120, 0))
        cv2.addWeighted(overlay, 0.12, output, 0.88, 0, output)
        cv2.polylines(output, [polygon], True, (255, 160, 0), 2)
        cv2.line(output, start, end, (0, 255, 255), 2)

    if show_detections:
        for index, (box, score) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = map(int, box)
            color = (70, 210, 70)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            tracker_text = ""
            if show_tracking and detections.tracker_id is not None:
                tracker_text = f" ID {int(detections.tracker_id[index])}"
            _draw_label(output, f"person {score:.2f}{tracker_text}", (x1, y1), color)

    if show_tracking and trajectories:
        for points in trajectories.values():
            if len(points) > 1:
                cv2.polylines(
                    output,
                    [np.asarray(points, dtype=np.int32)],
                    False,
                    (255, 0, 255),
                    2,
                )

    if show_counting:
        summary = f"IN {in_count}  OUT {out_count}  OCC {occupancy}"
        _draw_label(output, summary, (12, 30), (20, 20, 20))
    return output

