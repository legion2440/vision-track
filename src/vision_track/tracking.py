from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .detections import Detections


@dataclass(frozen=True)
class ByteTrackSettings:
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 2
    frame_rate: float = 30.0
    trajectory_length: int = 30


class StreamTracker:
    def __init__(self, settings: ByteTrackSettings) -> None:
        self.settings = settings
        self._tracker = self._create_tracker()
        self.trajectories: dict[int, deque[tuple[int, int]]] = {}

    def _create_tracker(self):
        import supervision as sv

        return sv.ByteTrack(
            track_activation_threshold=self.settings.track_activation_threshold,
            lost_track_buffer=self.settings.lost_track_buffer,
            minimum_matching_threshold=self.settings.minimum_matching_threshold,
            frame_rate=self.settings.frame_rate,
            minimum_consecutive_frames=self.settings.minimum_consecutive_frames,
        )

    def update(self, detections: Detections) -> Detections:
        tracked = self._tracker.update_with_detections(detections.to_supervision())
        result = Detections.from_supervision(tracked)
        self._update_trajectories(result)
        return result

    def _update_trajectories(self, detections: Detections) -> None:
        if detections.tracker_id is None:
            return
        active_ids: set[int] = set()
        for box, tracker_id in zip(detections.xyxy, detections.tracker_id):
            tracker_id = int(tracker_id)
            active_ids.add(tracker_id)
            center = (int((box[0] + box[2]) / 2), int(box[3]))
            trajectory = self.trajectories.setdefault(
                tracker_id, deque(maxlen=self.settings.trajectory_length)
            )
            trajectory.append(center)
        if len(self.trajectories) > 1000:
            self.trajectories = {
                key: value for key, value in self.trajectories.items() if key in active_ids
            }

    def reset(self) -> None:
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
        else:
            self._tracker = self._create_tracker()
        self.trajectories.clear()


def independent_trackers(
    settings: ByteTrackSettings,
) -> tuple[StreamTracker, StreamTracker]:
    return StreamTracker(settings), StreamTracker(settings)

