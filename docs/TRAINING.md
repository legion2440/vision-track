# VisionTrack: Training and Evaluation

This document contains the reproducible data, training, evaluation, optimization, and benchmarking workflow. It is not required for normal application startup.

[Русский README](../README.md) · [English README](../README_EN.md)

## 📋 TOC

- [🧰 Environment](#-environment)
- [📦 Dataset](#-dataset)
- [🔍 Validation and audit](#-validation-and-audit)
- [📓 Analysis notebook](#-analysis-notebook)
- [0️⃣ Pretrained baseline](#0️⃣-pretrained-baseline)
- [🏋️ Full transfer learning](#️-full-transfer-learning)
- [🧪 Evaluation policy](#-evaluation-policy)
- [📤 Checkpoint promotion](#-checkpoint-promotion)
- [📈 Model scaling](#-model-scaling)
- [✂️ Structured pruning](#️-structured-pruning)
- [📉 INT8 quantization](#-int8-quantization)
- [⚡ Performance benchmark](#-performance-benchmark)
- [🎬 Demo generation](#-demo-generation)
- [📁 Reports and artifacts](#-reports-and-artifacts)
- [⚠️ Known data limitations](#️-known-data-limitations)

## 🧰 Environment

Python is fixed to:

```text
3.13.*
```

Main recorded environment:

```text
Python             3.13.5
PyTorch            2.12.1+cu130
Torchvision        0.27.1+cu130
Ultralytics        8.4.78
Supervision        0.29.1
OpenCV             4.13.0.92
NumPy              2.4.6
ONNX Runtime       1.27.0
OS                 Windows 11
GPU                NVIDIA RTX 4080 Laptop GPU
CUDA build         13.0
```

Dependency files:

```text
requirements.txt
requirements-cuda.txt
requirements-mps.txt
```

Verify the environment:

```bash
python --version
python -c "import torch, supervision, cv2, streamlit"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 📦 Dataset

### Source

The project uses COCO 2017 person annotations.

COCO annotations are distributed under CC BY 4.0. Image licenses remain the per-image Flickr licenses stored in COCO metadata.

### Conversion policy

Only class `0: person` is exported.

`iscrowd` annotations are excluded.

Images without at least one usable non-crowd person box are omitted from the prepared dataset.

Ultralytics performs resize, letterbox conversion, tensor conversion, and normalization. The project does not apply a second manual normalization path before PyTorch inference.

### Split policy

Prepared split:

| Split | Images | Person objects |
|---|---:|---:|
| Train | 64,115 | 257,252 |
| Validation | 1,346 | 5,427 |
| Test | 1,347 | 5,350 |

Rules:

- all usable `train2017` person images are used for training;
- usable `val2017` person images are shuffled with seed `42`;
- one half is used for validation;
- the other half is isolated as the project test split;
- official `test2017` is not used because it has no public annotations;
- INT8 calibration uses only the training split.

The project test set is therefore an annotated subset of official `val2017`, not official COCO `test2017`.

### Prepare the dataset

```bash
python scripts/prepare_coco_person.py
```

Use `--overwrite` only when intentionally rebuilding the prepared dataset.

Limited pipeline smoke:

```bash
python scripts/prepare_coco_person.py \
  --max-train-images 500 \
  --max-val-images 100 \
  --max-test-images 100
```

Generated dataset configuration is compatible with Ultralytics YOLO.

## 🔍 Validation and audit

### Prepared-data validation

```bash
python scripts/validate_dataset.py
```

The validator checks:

- corrupt or unreadable images;
- image/label pairing;
- missing and empty label files;
- unknown classes;
- five-field YOLO syntax;
- normalized coordinate ranges;
- non-positive boxes;
- duplicate annotations;
- tiny objects.

Primary report:

```text
reports/dataset_validation.json
```

### Read-only dataset audit

```bash
python scripts/audit_dataset.py
```

The audit records:

- split counts;
- person density;
- image resolution distributions;
- relative box-size distributions;
- raw-to-prepared label integrity;
- proof of the empty-image filter;
- cross-split SHA-256 duplicates;
- cross-split perceptual-hash similarity;
- deterministic annotation contact sheets.

Recorded leakage result:

```text
exact SHA-256 cross-split leakage: 0
```

Perceptual-hash review found seven edges forming six clusters. Five clusters were judged same-scene near-duplicates and one was a false visual match. These clusters remain documented rather than silently removed after the control dataset was frozen.

Related policies:

```text
docs/annotation_policy.md
docs/dataset_split_protocol.md
```

## 📓 Analysis notebook

Open:

```text
notebooks/VisionTrack_Analysis.ipynb
```

Run:

```bash
jupyter notebook
```

The notebook covers:

- source and license;
- split distributions;
- objects per image;
- image resolutions;
- box sizes;
- validator and audit outputs;
- annotated samples;
- ONNX preprocessing;
- pretrained baseline predictions and errors;
- transfer-learning reports;
- pretrained, fine-tuned, pruned, and quantized comparisons.

Runtime production logic remains in `src/vision_track`.

## 0️⃣ Pretrained baseline

Run:

```bash
python scripts/run_baseline_stage.py --device cuda
```

The baseline stage:

1. validates the frozen control dataset;
2. evaluates pretrained `yolo26n.pt` on the full validation split;
3. records standard Ultralytics metrics;
4. records project operational-threshold metrics;
5. runs a deterministic one-epoch smoke on 256 train and 64 validation images;
6. reloads and verifies both smoke `best.pt` and `last.pt`;
7. does not use the isolated test split.

Latest readiness report:

```text
reports/baseline_runs/baseline_20260714T180333Z/
```

### Z0 standard validation

```text
Precision        0.8104
Recall           0.6410
mAP50            0.7532
mAP50-95         0.5181
Inference        2.874 ms/image
```

### Z0 operational point

Configuration:

```text
confidence        0.35
NMS IoU           0.50
matching IoU      0.50
```

Result:

```text
Precision         0.8850
Recall            0.5812
TP                 3,154
FP                   410
FN                 2,273
FP/image           0.3046
```

The one-epoch smoke is a pipeline check, not a quality result.

## 🏋️ Full transfer learning

Run:

```bash
python scripts/train.py --device cuda --confirm-full-run
```

Configuration is stored in:

```text
configs/app.yaml
models/checkpoints/config.yaml
```

Completed full-A setup:

```text
Pretrained model       yolo26n.pt
Task                   detect
Class                  0: person
Image size             640
Epochs                 80
Batch size             16
Optimizer              AdamW
Learning rate          0.001
Weight decay           0.0005
Patience               12
Seed                   42
Validation             every epoch
```

Training completed epoch `80/80`; early stopping did not trigger.

Training writes its run under the ignored directory:

```text
models/training_runs/
```

It does not overwrite the runtime checkpoint.

### Selected full-A checkpoint

```text
File:
models/training_runs/full_a/a_full_20260714T193616Z/ultralytics/a_full/weights/best.pt

SHA-256:
ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa

Size:
5,365,509 bytes
```

### Full-A best validation

```text
Precision        0.8147
Recall           0.6877
F1               0.7458
mAP50            0.7902
mAP50-95         0.5452
```

Delta against pretrained nano baseline:

```text
Precision       +0.0043
Recall          +0.0467
mAP50           +0.0370
mAP50-95        +0.0271
```

### Confidence sweep

Validation sweep:

| Confidence | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.4438 | 0.8406 | 0.5809 |
| 0.10 | 0.5943 | 0.7953 | 0.6803 |
| 0.25 | 0.7943 | 0.6987 | **0.7435** |
| 0.35 | 0.8616 | 0.6449 | 0.7377 |

Confidence `0.25` was selected by maximum operational validation F1.

A detector confidence score is not treated as a calibrated probability.

## 🧪 Evaluation policy

The isolated test split is not used for:

- hyperparameter selection;
- confidence selection;
- architecture scaling;
- pruning decisions;
- quantization decisions.

After model and threshold freeze, the selected full-A checkpoint was evaluated on test exactly once.

### Frozen full-A test result

```text
Precision        0.8096
Recall           0.6721
F1               0.7345
mAP50            0.7766
mAP50-95         0.5290
```

Pretrained M/L/X and both INT8 experiments were evaluated only on validation data.

The test split was not rerun during closeout.

## 📤 Checkpoint promotion

Training and runtime publication are separate operations.

Promotion command:

```bash
python scripts/promote_model.py \
  --source models/training_runs/full_a/<run-id>/ultralytics/a_full/weights/best.pt \
  --destination models/checkpoints/best.pt \
  --expected-sha256 <sha256>
```

Promotion performs:

- mandatory source and destination validation;
- expected SHA-256 verification;
- same-directory staged copy;
- file flush and `fsync`;
- checkpoint structure verification;
- requirement that class `0` is `person`;
- deterministic synthetic-frame inference smoke;
- actual runtime device selection;
- finite output and tensor-shape checks;
- atomic `os.replace`;
- destination SHA-256 verification.

A failed staged verification leaves the previous runtime checkpoint unchanged.

Promotion reports are written to:

```text
reports/model_promotions/
```

Bundled runtime checkpoint:

```text
models/checkpoints/best.pt
```

## 📈 Model scaling

Fine-tuning larger models was not repeated because the assignment transfer-learning requirement was already demonstrated by the full-A nano run.

Official pretrained M/L/X were evaluated to measure deployment quality and speed.

### Reproducibility hashes

| Model | Ultralytics name | SHA-256 |
|---|---|---|
| Fine-tuned N | `models/checkpoints/best.pt` | `ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa` |
| Pretrained M | `yolo26m.pt` | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| Pretrained L | `yolo26l.pt` | `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26` |
| Pretrained X | `yolo26x.pt` | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |

Pretrained weights are not stored in Git. They are downloaded by Ultralytics on demand.

### Standard validation

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Pretrained M | 0.8529 | 0.7516 | 0.7990 | 0.8555 | 0.6406 |
| Pretrained L | 0.8533 | 0.7565 | 0.8020 | 0.8606 | 0.6499 |
| Pretrained X | 0.8590 | 0.7735 | 0.8140 | 0.8740 | 0.6681 |

X produced the highest validation quality.

L was retained as the recommended GPU option because its quality was close to X while its runtime cost was lower.

Fine-tuned N remains the bundled transfer-learning artifact and portable fallback.

No tested model met the assignment recall and F1 targets simultaneously on the full COCO-person protocol.

## ✂️ Structured pruning

The pruning pipeline is implemented in:

```text
scripts/prune.py
```

Method:

```text
Torch-Pruning dependency graph
magnitude importance
target channel sparsity: 0.20
recovery fine-tuning: 10 epochs
```

The script requires a real structural reduction in parameters and operations. A mask that only writes zeros is not accepted as pruning.

Expected outputs for a successful accepted run:

```text
models/checkpoints/best_pruned.pt
reports/pruning_report.json
```

No pruned model is bundled in the final repository. Runtime selection does not depend on a pruned artifact.

## 📉 INT8 quantization

The quantization pipeline is implemented in:

```text
scripts/quantize.py
```

Configuration:

```text
Export format          ONNX
Precision              static INT8
Quantization format    QDQ
Activation type        QUInt8
Weight type            QInt8
Per-channel            enabled
Calibration split      train
Calibration images     256
ONNX opset             17
Simplify               enabled
Dynamic shapes         disabled
Embedded NMS            disabled
```

Artifacts are written to an ignored experiments directory first:

```text
models/checkpoints/experiments/
```

An INT8 artifact may be promoted to:

```text
models/checkpoints/best_quantized.onnx
```

only if it:

- loads successfully;
- executes real inference;
- produces non-zero valid evaluation output;
- loses at most one absolute percentage point of F1;
- loses at most one absolute percentage point of mAP50-95;
- improves CPU throughput by at least 15%.

### Pretrained L INT8

```text
FP32 ONNX size         99.6 MB
INT8 ONNX size         26.2 MB
Validation P/R/mAP     0 / 0 / 0
Status                 rejected
```

### Fine-tuned N INT8

```text
FP32 ONNX size         9.8 MB
INT8 ONNX size         3.1 MB
Load/inference smoke   passed
Validation P/R/F1/mAP  0 / 0 / 0 / 0
Status                 rejected
```

Both artifacts failed the quality gate.

Therefore:

```text
models/checkpoints/best_quantized.onnx
```

is intentionally absent.

The zero results most likely indicate an unresolved output/postprocessing or class-mapping incompatibility in the ONNX evaluation path. This was not proven, so the artifacts are recorded as failed rather than presented as valid models.

## ⚡ Performance benchmark

Main benchmark command:

```bash
python scripts/benchmark.py data/demo/video-a.mp4 data/demo/video-b.mp4
```

Inputs are normalized to:

```text
1280 × 720
```

Recorded fields include:

- hardware and OS;
- Python and dependency versions;
- model and SHA-256;
- backend and provider;
- requested and actual device;
- resolution and image size;
- stream count;
- warmup and measured batches;
- inference latency;
- end-to-end latency;
- per-stream and aggregate FPS;
- stage timings;
- artifact size;
- scenario status.

Unavailable devices and missing artifacts are recorded as `skipped`; values are not fabricated.

### Fine-tuned N detector-only measurements

| Device | Streams | Aggregate FPS | FPS per stream |
|---|---:|---:|---:|
| CPU | 1 | 35.43 | 35.43 |
| CPU | 2 | 44.69 | 22.35 |
| CUDA | 1 | 76.74 | 76.74 |
| CUDA | 2 | 128.22 | 64.11 |

These measurements use decoded and resized frames supplied before timing.

### Full-pipeline pretrained measurements

| Model | Device | Streams | Aggregate FPS | FPS per stream |
|---|---|---:|---:|---:|
| M | CPU | 1 | 7.11 | 7.11 |
| M | CPU | 2 | 7.00 | 3.50 |
| M | CUDA | 1 | 42.71 | 42.71 |
| M | CUDA | 2 | 57.51 | 28.75 |
| L | CPU | 1 | 5.83 | 5.83 |
| L | CPU | 2 | 5.99 | 2.99 |
| L | CUDA | 1 | 36.29 | 36.29 |
| L | CUDA | 2 | 47.38 | 23.69 |
| X | CPU | 1 | 2.81 | 2.81 |
| X | CPU | 2 | 2.16 | 1.08 |
| X | CUDA | 1 | 22.87 | 22.87 |
| X | CUDA | 2 | 36.83 | 18.42 |

GPU scenarios pass the assignment FPS target. CPU M/L/X scenarios do not.

Fine-tuned N detector-only FPS and pretrained full-pipeline FPS use different measurement scopes and must not be compared directly.

## 🎬 Demo generation

Generate demo artifacts from two real videos:

```bash
python scripts/generate_demo.py data/demo/video-a.mp4 data/demo/video-b.mp4
```

Outputs:

```text
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
reports/demo_results/demo_metadata.json
```

The metadata records:

- model ID and display name;
- checkpoint path and SHA-256;
- backend and device;
- confidence and IoU;
- hardware;
- source videos;
- output parameters.

The repository does not use placeholder demo files or fabricated benchmark values.

## 📁 Reports and artifacts

Primary artifacts:

```text
models/checkpoints/best.pt
models/checkpoints/config.yaml
reports/performance_metrics.json
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
reports/demo_results/demo_metadata.json
```

Training and evaluation reports include:

```text
reports/dataset_validation.json
reports/dataset_audit.json
reports/dataset_audit.md
reports/baseline_runs/
reports/best_val_metrics.json
reports/best_test_metrics.json
reports/confidence_sweep_val.csv
reports/confidence_sweep_val.json
reports/final_post_training_summary.json
reports/frozen_selection_before_test.json
reports/model_promotions/
reports/quantization_report.json
```

Some generated reports exist only after the corresponding workflow has been run.

Large pretrained weights, experimental ONNX files, training runs, and caches are ignored and are not stored in Git.

## ⚠️ Known data limitations

### Positive-only evaluation set

The converter omits images without usable person boxes.

Validation and test therefore contain positive images only. False-positive performance on person-free backgrounds is not measured completely.

### Crowd annotations

`iscrowd` boxes are excluded, but an image may still contain ordinary labeled persons together with unlabeled crowd persons.

Recorded affected images:

```text
Train          5,212
Validation       117
Test             110
Total          5,439
```

Predictions on unlabeled crowd persons can be counted as false positives.

### Near-duplicate scenes

Exact cross-split SHA leakage is zero.

Five manually reviewed perceptual-hash clusters contain same-scene frames across splits. They remain documented because the control dataset and completed training run were already frozen.

### Tracking metrics

COCO detection annotations do not contain trajectories.

The project therefore does not report:

- IDF1;
- HOTA;
- MOTA;
- ID switches.

Tracking and counting behavior are covered by deterministic unit and integration tests plus real demo artifacts.

### Domain scope

COCO-person is used as a general benchmark.

No separate surveillance-domain training set was added during closeout. Domain-specific quality would require a grouped split by source video or camera to prevent adjacent-frame leakage.
