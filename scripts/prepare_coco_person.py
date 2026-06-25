from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)


def extract(archive: Path, destination: Path) -> None:
    marker = destination / f".{archive.stem}.extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def load_coco(annotation_path: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["name"]: item["id"] for item in payload["categories"]}
    person_id = categories["person"]
    images = {item["id"]: item for item in payload["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for item in payload["annotations"]:
        if item["category_id"] == person_id and not item.get("iscrowd", 0):
            annotations[item["image_id"]].append(item)
    return images, annotations


def yolo_line(annotation: dict, image: dict) -> str | None:
    x, y, width, height = map(float, annotation["bbox"])
    if width <= 0 or height <= 0:
        return None
    image_width, image_height = float(image["width"]), float(image["height"])
    x_center = (x + width / 2) / image_width
    y_center = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    values = [x_center, y_center, normalized_width, normalized_height]
    if any(value < 0 or value > 1 for value in values):
        return None
    return "0 " + " ".join(f"{value:.8f}" for value in values)


def convert_split(
    image_source: Path,
    images: dict[int, dict],
    annotations: dict[int, list[dict]],
    image_ids: list[int],
    output: Path,
    split: str,
    limit: int | None,
) -> int:
    output_images = output / "images" / split
    output_labels = output / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    written = 0
    for image_id in image_ids:
        objects = annotations.get(image_id, [])
        if not objects:
            continue
        image = images[image_id]
        lines = [line for item in objects if (line := yolo_line(item, image))]
        if not lines:
            continue
        source = image_source / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_images / source.name)
        (output_labels / f"{source.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        written += 1
        if limit and written >= limit:
            break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 and convert it to a person-only YOLO dataset"
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "coco2017")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "processed" / "coco_person"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.5,
        help="Fraction of annotated val2017 images kept for validation; the rest is isolated test data",
    )
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-val-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use archives or extracted COCO files already present in --raw-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing converted output directory",
    )
    args = parser.parse_args()

    archives = args.raw_dir / "archives"
    if not args.skip_download:
        for filename, url in COCO_URLS.items():
            download(url, archives / filename)
    for filename in COCO_URLS:
        archive = archives / filename
        if archive.exists():
            extract(archive, args.raw_dir)

    train_images, train_annotations = load_coco(
        args.raw_dir / "annotations" / "instances_train2017.json"
    )
    val_images, val_annotations = load_coco(
        args.raw_dir / "annotations" / "instances_val2017.json"
    )
    train_ids = sorted(image_id for image_id in train_images if train_annotations.get(image_id))
    holdout_ids = sorted(image_id for image_id in val_images if val_annotations.get(image_id))
    random.Random(args.seed).shuffle(holdout_ids)
    cut = int(len(holdout_ids) * args.validation_fraction)
    val_ids, test_ids = holdout_ids[:cut], holdout_ids[cut:]

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output}. Pass --overwrite to replace it."
            )
        resolved_output = args.output.resolve()
        protected = {ROOT.resolve(), (ROOT / "data").resolve(), (ROOT / "data" / "processed").resolve()}
        if resolved_output in protected:
            raise ValueError(f"Refusing to remove protected directory: {resolved_output}")
        shutil.rmtree(resolved_output)
    counts = {
        "train": convert_split(
            args.raw_dir / "train2017",
            train_images,
            train_annotations,
            train_ids,
            args.output,
            "train",
            args.max_train_images,
        ),
        "val": convert_split(
            args.raw_dir / "val2017",
            val_images,
            val_annotations,
            val_ids,
            args.output,
            "val",
            args.max_val_images,
        ),
        "test": convert_split(
            args.raw_dir / "val2017",
            val_images,
            val_annotations,
            test_ids,
            args.output,
            "test",
            args.max_test_images,
        ),
    }
    dataset_yaml = (
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: person\n"
    )
    (args.output / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    metadata = {
        "dataset": "COCO 2017 person-only",
        "source": "https://cocodataset.org/",
        "annotations_license": "Creative Commons Attribution 4.0",
        "image_licenses": "Per-image Flickr licenses recorded in COCO metadata",
        "seed": args.seed,
        "validation_fraction_of_val2017": args.validation_fraction,
        "split_counts": counts,
        "test_isolation": "test is a deterministic holdout from val2017 and must be evaluated once after all choices are frozen",
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
