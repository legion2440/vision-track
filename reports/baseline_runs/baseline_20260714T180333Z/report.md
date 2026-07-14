# Detector baseline Z0/A readiness

Run ID: `baseline_20260714T180333Z`

Status: **complete**

## Scope

- Z0: pretrained `yolo26n.pt`, full control validation split.
- A smoke: one epoch on deterministic 256 train / 64 val linked samples.
- Full A training: **not started**.
- Test split: **not used**.
- Control dataset: unchanged by before/after fingerprint.

## Z0 standard evaluator

- Evaluator confidence: 0.001
- Precision: 0.810378
- Recall: 0.641007
- mAP50: 0.753210
- mAP50-95: 0.518085
- Inference: 2.874 ms/image
- Images / objects: 1346 / 5427

## Z0 project runtime threshold

- Confidence / NMS IoU / match IoU: 0.35 / 0.5 / 0.5
- Precision: 0.884961
- Recall: 0.581168
- Detections: 3564
- False positives/image: 0.304606
- Wall time: 6.775 ms/image

## A smoke

- Training wall time: 40.89 s
- Train / val objects: 995 / 278
- best.pt reload: passed
- last.pt reload: passed
- Both checkpoints remain in the ignored local training-run directory.

## Full A plan (not executed)

- Configured epochs / batch / patience: 80 / 16 / 12
- Provisional time range: 17.30–86.48 hours.
- Provisional disk range: 0.01–0.50 GiB.

## Known control limitation

Exact SHA leakage is zero. Five manually reviewed cross-split same-scene pHash clusters remain in the unchanged control split. Z0/A results should be interpreted with this small contamination; remediation belongs only to dataset_v2 or a separate deduplicated materialization.
