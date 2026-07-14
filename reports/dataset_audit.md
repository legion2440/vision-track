# Dataset audit

Status: **complete**

## Input integrity

Overall extracted-input status: **confirmed**

| Split | Expected | Actual | Decode failures | Dimension mismatches |
|---|---:|---:|---:|---:|
| train2017 | 118287 | 118287 | 0 | 0 |
| val2017 | 5000 | 5000 | 0 | 0 |

Limitation: Image ZIP integrity cannot be checked because train2017.zip and/or val2017.zip is unavailable; extracted file names, decode results, and dimensions are checked against COCO metadata instead.

## Raw COCO person inventory

| Source split | Images | Usable positives | No person | Only excluded person | Usable boxes | Crowd boxes excluded | Invalid boxes excluded |
|---|---:|---:|---:|---:|---:|---:|---:|
| train2017 | 118287 | 64115 | 54172 | 0 | 257252 | 5212 | 1 |
| val2017 | 5000 | 2693 | 2307 | 0 | 10777 | 227 | 0 |

| Source split | Images with crowd | Retained normal+crowd images | Unlabeled retained crowd annotations |
|---|---:|---:|---:|
| train2017 | 5212 | 5212 | 5212 |
| val2017 | 227 | 227 | 227 |

| Source split | People/image positive p50 / p95 | Median W×H | Median image AR | Bbox area p05 / p50 / p95 | Bboxes <1% | Edge-touching boxes |
|---|---:|---:|---:|---:|---:|---:|
| train2017 | 2.00 / 13.00 | 640×480 | 1.33 | 0.03% / 1.30% / 39.86% | 45.87% | 20.53% |
| val2017 | 2.00 / 13.00 | 640×480 | 1.33 | 0.03% / 1.41% / 39.14% | 44.59% | 21.71% |

## Current preparer selection

The current converter selects only images with at least one usable, non-crowd person box. Images without such boxes are omitted rather than preserved as negative samples.

| Split | Selected images | Objects | People/image p50 / p95 | Bbox area p50 | Images / annotations with unlabeled crowd |
|---|---:|---:|---:|---:|---:|
| train | 64115 | 257252 | 2.00 / 13.00 | 1.30% | 5212 / 5212 |
| val | 1346 | 5427 | 2.00 / 13.00 | 1.44% | 117 / 117 |
| test | 1347 | 5350 | 2.00 / 13.00 | 1.39% | 110 / 110 |

## Prepared dataset

| Split | Images | Empty/negative images | Objects |
|---|---:|---:|---:|
| train | 64115 | 0 | 257252 |
| val | 1346 | 0 | 5427 |
| test | 1347 | 0 | 5350 |

## Cross-split leakage evidence

- Exact SHA-256 groups: 0
- pHash candidate edges (distance ≤ 6): 7
- pHash connected clusters: 6
- Machine JSON/CSV evidence is complete; only prose previews may be capped.

The deterministic exact-duplicate resolution manifest contains 0 groups and has not been applied to the control `coco_person` materialization.

## Raw vs prepared integrity

- Image selection matches current preparer: **True**
- Labels match current preparer: **True**
- Missing expected boxes: 0
- Extra actual boxes: 0
- Coordinate-mismatched box pairs: 0
- Coordinate tolerance: 1e-06

## Warnings

- **unlabeled_crowd_positive_risk**: The current preparer retains images containing normal person boxes and iscrowd regions, but does not write the crowd regions to YOLO labels. Real crowded people therefore become unlabeled background and may suppress recall. Do not choose exclusion, ignore-region handling, or manual review until the dataset policy decision is made.

## Manual visual review

Review status: **complete**

- `random`: reviewed — Varied scenes and scales; sampled person boxes align visually and no corrupt image or label rendering is apparent.
- `tiny`: reviewed — Very small people are frequently only a few pixels but boxes land on plausible person locations; these examples remain inherently hard to verify precisely.
- `edge_partial`: reviewed — Partial and out-of-frame people are common. Some extreme boxes show only limited visible body evidence and should be considered explicitly in dataset_v2 policy; the control labels remain unchanged.
- `high_density`: reviewed — Dense scenes contain many plausible person boxes, while additional crowd/background people are not represented by ordinary YOLO boxes; this supports the documented crowd-risk review.
- `crowd`: reviewed — All 16 displayed crowd regions decoded from COCO RLE segmentation masks (orange), with zero bbox fallbacks. Masks expose substantial real-person content omitted from ordinary green YOLO boxes.
- `phash-00001`: near_duplicate_same_scene — Same child and composition in three color/processing variants across train and test.
- `phash-00002`: similar_not_duplicate — False pHash candidate: a woman in a dark night scene versus people on a daylight tennis court.
- `phash-00003`: near_duplicate_same_scene — Same kitchen, person, pose, and framing across train and test.
- `phash-00004`: near_duplicate_same_scene — Same four road workers and composition across train and validation.
- `phash-00005`: near_duplicate_same_scene — Same storefront scene with a small framing shift across train and validation.
- `phash-00006`: near_duplicate_same_scene — Same indoor group and framing across train and validation; validation includes one additional visible-person box.

## Dataset v2 recommendations

- Preserve this `coco_person` output unchanged as the seed-42 control.
- Exact leakage groups: 0; apply any exact-resolution manifest only to `dataset_v2`.
- Group each of the 5 reviewed duplicate/same-scene pHash clusters into one split only in `dataset_v2` or a separately named deduplicated materialization.
- Decide whether retained iscrowd regions become exclusions, ignore regions, or manually verified positives before training.
- Add COCO images without person as ordinary negatives only in the new dataset version, together with domain positives and hard negatives.
- Assign webcam/CCTV frames by source, scene, and time block; keep COCO, domain, and hard-negative metrics separate and freeze test before threshold selection.

Training was not started by this audit.
