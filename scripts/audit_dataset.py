from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.dataset_audit import audit_dataset, render_audit_markdown


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit raw COCO person annotations and a prepared YOLO dataset"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "coco2017",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "processed" / "coco_person",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "dataset_audit.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=ROOT / "reports" / "dataset_audit.md",
    )
    parser.add_argument(
        "--contact-sheet-dir",
        type=Path,
        default=ROOT / "reports" / "dataset_audit_contact_sheets",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--perceptual-distance", type=int, default=6)
    args = parser.parse_args()

    report = audit_dataset(
        args.raw_dir,
        args.dataset,
        contact_sheet_dir=args.contact_sheet_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        perceptual_distance=args.perceptual_distance,
    )
    report["raw_dir"] = _display_path(args.raw_dir)
    report["prepared_dir"] = _display_path(args.dataset)
    report["missing_inputs"] = [
        _display_path(Path(path)) for path in report["missing_inputs"]
    ]
    if "contact_sheets" in report:
        report["contact_sheets"] = [
            _display_path(Path(path)) for path in report["contact_sheets"]
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown_report.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_audit_markdown(report))
    print(json.dumps({"status": report["status"], "report": _display_path(args.report)}, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
