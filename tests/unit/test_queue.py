from __future__ import annotations

import numpy as np

from vision_track.queues import FramePacket, LatestFrameQueue


def test_latest_frame_queue_drops_stale_frame() -> None:
    queue = LatestFrameQueue()
    queue.put(FramePacket.create(np.zeros((2, 2, 3)), 1))
    queue.put(FramePacket.create(np.ones((2, 2, 3)), 2))
    packet = queue.get_nowait()
    assert packet.frame_index == 2
    assert queue.dropped == 1
    assert queue.dropped_rate == 0.5

