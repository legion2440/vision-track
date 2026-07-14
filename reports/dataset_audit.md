# Dataset audit

Status: **partial**

## Missing inputs

- `data/processed/coco_person/images/{train,val,test}`

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

## Warnings

- **unlabeled_crowd_positive_risk**: The current preparer retains images containing normal person boxes and iscrowd regions, but does not write the crowd regions to YOLO labels. Real crowded people therefore become unlabeled background and may suppress recall. Do not choose exclusion, ignore-region handling, or manual review until the dataset policy decision is made.

## Manual review still required

Lighting, indoor/outdoor context, body-part distractors, screens/posters, reflections, occlusion quality, and label correctness require review of the generated contact sheets. They are not inferred from COCO metadata.

## Split recommendation

Keep COCO general evaluation separate from domain webcam/CCTV and hard-negative results. Assign domain samples by source/scene/time-block group, never by individual frame. Freeze the final test groups before threshold selection and use only train data for augmentation/calibration.

## Checks not executed

- **prepared YOLO statistics and raw/prepared comparison**: prepared images/{train,val,test} is missing
- **exact and perceptual duplicate leakage**: image files for the materialized splits are missing
- **annotation contact sheets and manual visual review**: image files for the materialized splits are missing
