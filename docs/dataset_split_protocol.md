# Dataset split protocol

## Current COCO-person baseline

The existing converter uses all usable non-crowd person images from COCO
`train2017` for train. Usable person images from `val2017` are shuffled with
seed 42 and divided equally into validation and an isolated test holdout. Images
without usable person boxes are currently omitted; the audit reports this
explicitly instead of treating the prepared dataset as if it contained
negatives.

Keep the current split unchanged for the controlled A/B/C comparison unless a
new dataset version is deliberately created. Do not use its test half for
threshold selection.

## Domain and hard-negative data

Every sample must have a manifest record conforming to
[`configs/domain_manifest.schema.json`](../configs/domain_manifest.schema.json).
The atomic split unit is `group_id`, representing one source, scene, and
contiguous time block. All frames with the same `group_id` must have the same
split.

Recommended first version:

- train: approximately 70% of groups;
- validation: approximately 15% of groups;
- isolated test: approximately 15% of groups;
- hold out complete sources or scenes for at least part of domain test;
- stratify at group level for indoor/outdoor, camera, lighting, crowd density,
  positives, mixed samples, and hard-negative types;
- deduplicate before assignment, then re-run exact and perceptual leakage checks
  after materializing the split.

Percentages are secondary to source independence. When data is limited, fewer
independent test groups are preferable to leaking adjacent frames across train
and test.

## Evaluation strata

Report these independently before producing any aggregate:

1. COCO/general person validation and isolated test;
2. domain webcam/CCTV positives and mixed frames;
3. hard negatives, including false person detections per image and per minute;
4. short counting videos after detector decisions are frozen.

Confidence is selected only on validation/domain-validation. The isolated test
is run after model, data version, threshold, and post-processing decisions are
frozen.

## Controlled model sequence

- Z0: zero-shot pretrained `yolo26n.pt`;
- A: COCO person-only fine-tune;
- B: COCO plus domain positives;
- C: COCO plus domain positives and hard negatives.

Use the same seed, split manifest, image size, and training budget for A/B/C.
This protocol does not authorize training during the dataset-audit stage.
