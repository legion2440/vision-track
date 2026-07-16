from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.configuration import load_config, resolve_project_path
from vision_track.counting import ZoneCounter, ZoneGeometry
from vision_track.detector import create_backend
from vision_track.device import select_device
from vision_track.rendering import render_frame
from vision_track.tracking import ByteTrackSettings, StreamTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real ROI and multi-stream demo artifacts")
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--backend", choices=["pytorch", "onnxruntime"], default="pytorch")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()

    config = load_config()
    model_path = (
        resolve_project_path(config.model.quantized_checkpoint)
        if args.backend == "onnxruntime"
        else resolve_project_path(config.model.checkpoint)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Required model artifact not found: {model_path}")
    if len(args.videos) < 2:
        raise ValueError("Provide at least two videos for a genuine multi-stream demo")
    captures = [cv2.VideoCapture(str(path)) for path in args.videos]
    if not all(capture.isOpened() for capture in captures):
        raise OSError("One or more demo videos could not be opened")

    device = select_device(force=None if args.device == "auto" else args.device)
    detector = create_backend(
        args.backend,
        model_path,
        device,
        image_size=config.model.image_size,
        confidence=config.model.confidence,
        iou=config.model.iou,
        person_class_id=0,
    )
    detector.load()
    detector.warmup()
    tracking_cfg = config.tracking
    trackers = [
        StreamTracker(
            ByteTrackSettings(
                tracking_cfg.track_activation_threshold,
                tracking_cfg.lost_track_buffer,
                tracking_cfg.minimum_matching_threshold,
                tracking_cfg.minimum_consecutive_frames,
                tracking_cfg.frame_rate,
                tracking_cfg.trajectory_length,
            )
        )
        for _ in captures
    ]
    geometry = ZoneGeometry(
        config.counting.line_start,
        config.counting.line_end,
        config.counting.polygon,
    )
    counters = [ZoneCounter(geometry) for _ in captures]
    output_dir = ROOT / "reports" / "demo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "multi_stream_demo.mp4"
    image_path = output_dir / "roi_counting_example.png"
    tile_size = (640, 360)
    grid_size = (tile_size[0] * 2, tile_size[1])
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        grid_size,
    )
    if not writer.isOpened():
        raise OSError("Unable to create demo MP4 with the available OpenCV codecs")
    saved_example = False
    try:
        for _ in range(args.frames):
            frames = []
            for capture in captures:
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok or frame is None:
                    raise OSError("Demo video decoder failed")
                frames.append(frame)
            results = detector.infer_batch(frames)
            rendered = []
            for frame, result, tracker, counter in zip(frames, results, trackers, counters):
                detections = tracker.update(result.detections)
                counter.update(detections, frame.shape[:2])
                annotated = render_frame(
                    frame,
                    detections,
                    trajectories=tracker.trajectories,
                    geometry=geometry,
                    in_count=counter.in_count,
                    out_count=counter.out_count,
                    occupancy=counter.occupancy,
                )
                rendered.append(cv2.resize(annotated, tile_size))
                if not saved_example:
                    cv2.imwrite(str(image_path), annotated)
                    saved_example = True
            grid = np.hstack(rendered[:2])
            writer.write(grid)
    finally:
        writer.release()
        for capture in captures:
            capture.release()
    if not image_path.exists() or not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("Demo artifacts were not generated successfully")
    print(f"Created {image_path}")
    print(f"Created {video_path}")


if __name__ == "__main__":
    main()
