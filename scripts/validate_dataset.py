from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.dataset_validation import validate_yolo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a person-only YOLO dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "processed" / "coco_person",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "dataset_validation.json",
    )
    parser.add_argument("--minimum-area", type=float, default=0.0001)
    args = parser.parse_args()

    report = validate_yolo_dataset(
        args.dataset,
        allowed_class_ids={0},
        minimum_area=args.minimum_area,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

