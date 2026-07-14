from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_track.dataset_audit import (
    audit_dataset,
    build_manual_review_evidence,
    render_audit_markdown,
)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def audit_exit_code(report: dict) -> int:
    complete = (
        report.get("status") == "complete"
        and report.get("manual_review", {}).get("status") == "complete"
    )
    return 0 if complete else 2


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
    parser.add_argument(
        "--exact-resolution-manifest",
        type=Path,
        default=ROOT / "reports" / "dataset_audit_exact_resolution.json",
    )
    parser.add_argument(
        "--phash-edges-csv",
        type=Path,
        default=ROOT / "reports" / "dataset_audit_phash_edges.csv",
    )
    parser.add_argument(
        "--phash-clusters",
        type=Path,
        default=ROOT / "reports" / "dataset_audit_phash_clusters.json",
    )
    parser.add_argument(
        "--manual-review",
        type=Path,
        default=ROOT / "reports" / "dataset_audit_manual_review.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--perceptual-distance", type=int, default=6)
    parser.add_argument("--box-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    report = audit_dataset(
        args.raw_dir,
        args.dataset,
        contact_sheet_dir=args.contact_sheet_dir,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        perceptual_distance=args.perceptual_distance,
        box_tolerance=args.box_tolerance,
    )
    duplicates = report.get("prepared_dataset", {}).get("duplicates", {})
    phash = duplicates.get("perceptual_hash", {})
    existing_review = None
    if args.manual_review.is_file():
        existing_review = json.loads(args.manual_review.read_text(encoding="utf-8"))
    report["manual_review"] = build_manual_review_evidence(
        report.get("contact_sheets", ()),
        phash.get("clusters", ()),
        existing_review,
    )
    report["raw_dir"] = _display_path(args.raw_dir)
    report["prepared_dir"] = _display_path(args.dataset)
    report["missing_inputs"] = [
        _display_path(Path(path)) for path in report["missing_inputs"]
    ]
    extracted_images = report.get("input_validation", {}).get(
        "extracted_images", {}
    )
    for item in extracted_images.values():
        if "directory" in item:
            item["directory"] = _display_path(Path(item["directory"]))
    if "contact_sheets" in report:
        report["contact_sheets"] = [
            {**item, "path": _display_path(Path(item["path"]))}
            for item in report["contact_sheets"]
        ]
    if "perceptual_cluster_sheets" in report:
        report["perceptual_cluster_sheets"] = [
            {
                **item,
                "sheets": [_display_path(Path(path)) for path in item["sheets"]],
            }
            for item in report["perceptual_cluster_sheets"]
        ]
    for item in report["manual_review"]["category_contact_sheets"]:
        item["path"] = _display_path(Path(item["path"]))

    reviewed_phash_leakage = sum(
        item["decision"] in {"duplicate", "near_duplicate_same_scene"}
        for item in report["manual_review"]["phash_clusters"]
    )
    report["dataset_v2_recommendations"] = {
        "control_coco_person": "preserve unchanged",
        "exact_leakage": {
            "cross_split_group_count": duplicates.get(
                "exact_cross_split_group_count", 0
            ),
            "action": "apply the resolution manifest only to dataset_v2",
        },
        "perceptual_leakage": {
            "review_status": report["manual_review"]["status"],
            "reviewed_duplicate_or_same_scene_clusters": reviewed_phash_leakage,
            "action": (
                "group each reviewed duplicate/same-scene cluster into one split "
                "when dataset_v2 is materialized"
            ),
        },
        "crowd": (
            "choose exclusion, ignore-region handling, or manually verified positive "
            "labels before training dataset_v2"
        ),
        "negatives": (
            "consider COCO images without person only in dataset_v2, alongside domain "
            "positives and hard negatives"
        ),
        "split": (
            "rebuild dataset_v2 with source/scene/time-block grouping and a frozen test"
        ),
        "training_started": False,
    }

    args.exact_resolution_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.exact_resolution_manifest.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(
            report.get("exact_duplicate_resolution_manifest", {}),
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    args.phash_clusters.parent.mkdir(parents=True, exist_ok=True)
    with args.phash_clusters.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            {
                "algorithm": phash.get("algorithm"),
                "maximum_hamming_distance": phash.get("maximum_hamming_distance"),
                "cluster_count": phash.get("cluster_count", 0),
                "clusters": phash.get("clusters", []),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    args.phash_edges_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.phash_edges_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("left", "right", "hamming_distance"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(phash.get("edges", ()))
    args.manual_review.parent.mkdir(parents=True, exist_ok=True)
    with args.manual_review.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report["manual_review"], handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    report["evidence_files"] = {
        "exact_resolution_manifest": _display_path(args.exact_resolution_manifest),
        "phash_edges_csv": _display_path(args.phash_edges_csv),
        "phash_clusters": _display_path(args.phash_clusters),
        "manual_review": _display_path(args.manual_review),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
    with args.markdown_report.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(render_audit_markdown(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "manual_review_status": report["manual_review"]["status"],
                "report": _display_path(args.report),
            },
            indent=2,
        )
    )
    return audit_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
