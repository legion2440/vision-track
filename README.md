# VisionTrack

VisionTrack is a multi-stream person detection, tracking, line-crossing, and ROI-occupancy application. It uses an Ultralytics YOLO26 nano detector, one shared inference scheduler, a separate video reader and ByteTrack state for every stream, OpenCV rendering, and a Streamlit dashboard.

The project supports local videos, server-side local webcams, HTTP video URLs, and RTSP URLs. CUDA is selected first when available, then Apple MPS, then CPU. OpenCV CUDA is not required; acceleration applies to PyTorch inference. ONNX Runtime INT8 inference uses CUDA only when a CUDA execution provider is installed and available, otherwise it reports and uses CPU.

## Features

- Multiple simultaneous local-file, local-webcam, HTTP, and RTSP sources.
- A single detector instance shared across streams.
- One reader thread and one latest-frame queue of size 1 per stream.
- Stale frame dropping instead of latency-producing queue growth.
- Per-stream ByteTrack, trajectories, line counts, polygon occupancy, settings, errors, and lifecycle.
- States: `CREATED`, `CONNECTING`, `ACTIVE`, `EOF`, `RECONNECTING`, `FAILED`, and `STOPPED`.
- Bounded reconnect with exponential backoff for webcams and HTTP/RTSP; local-file EOF never reconnects.
- Per-stream start, stop, restart, removal, source replacement, tracker reset, and counter reset.
- PyTorch and direct ONNX Runtime detector backends with a common detection format.
- Credential and query-token masking in logs and displayed errors.
- Dataset conversion/validation, baseline evaluation, transfer learning, structured pruning, INT8 quantization, benchmarking, and demo generation.

## Architecture

Each source has an independent `StreamContext`. A reader writes only the newest frame to a queue with capacity one. The shared scheduler gathers available frames, performs one detector batch, and routes results back to the correct per-stream tracker/counter/rendering state. Inference never runs inside a Streamlit rerun.

```text
reader A -> latest queue A --\
reader B -> latest queue B ----> shared inference scheduler -> shared detector
reader N -> latest queue N --/                |
                                               +-> tracker/counter/render A
                                               +-> tracker/counter/render B
                                               +-> tracker/counter/render N
```

Each rendered publication follows a separate preview path:

```text
rendered frame -> latest-only loopback WebSocket -> binary JPEG -> persistent browser canvas
```

Streamlit owns layout, controls, settings, metrics, and lifecycle/error text; live frames do not travel through Streamlit image elements or per-frame reruns. Each stream canvas has an independent WebSocket connection and the application retains only the latest encoded JPEG for that session and stream. Preview transport is capped at 15 FPS, but a processed preview cannot update faster than the inference scheduler publishes rendered frames. Detector inference FPS and WebSocket preview FPS are separate measurements.

Tracked identities are represented externally as `(stream_id, tracker_id)`, so the same ByteTrack integer in two streams cannot collide.

## Repository layout

```text
app.py
configs/app.yaml
src/vision_track/
  configuration.py       device.py            detections.py
  preprocessing.py       detector.py          sources.py
  readers.py             queues.py            lifecycle.py
  context.py             scheduler.py         tracking.py
  counting.py            rendering.py         metrics.py
  engine.py              logging_utils.py     streamlit_state.py
  preview.py             webcams.py
scripts/
notebooks/VisionTrack_Analysis.ipynb
data/{raw,processed,demo}/
models/checkpoints/
reports/demo_results/
logs/app_errors.log
tests/{unit,integration}/
```

## Supported platforms

- Windows 11 with Git Bash.
- macOS on Apple Silicon with MPS fallback logic.
- Linux with CUDA or CPU.
- Python is intentionally fixed to `3.13.*`.

All commands below are run from the repository root. On Windows Git Bash:

```bash
cd /d/TSchool/vision-track
```

## Installation

### Windows 11

Install 64-bit Python 3.13, then in Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA CUDA, install a compatible NVIDIA driver and use the CUDA dependency file:

```bash
python -m pip install -r requirements-cuda.txt
```

### macOS Apple Silicon

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mps.txt
python -c "import torch; print(torch.backends.mps.is_available())"
```

The normal macOS PyTorch wheel includes MPS support. Unsupported operations may fall back to CPU at framework level.

### Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA, install from `requirements-cuda.txt` after installing a compatible NVIDIA driver. CPU remains the default and tested fallback.

Verify the audit imports:

```bash
python -c "import torch, supervision, cv2, streamlit"
```

## Device and backend selection

`vision_track.device.select_device()` selects:

1. CUDA when `torch.cuda.is_available()` is true.
2. MPS when `torch.backends.mps.is_available()` is true.
3. CPU otherwise.

The UI and benchmark display the actual device and backend/provider. The
sidebar selects an explicit model from the model registry. The bundled default
is the fine-tuned nano transfer-learning checkpoint at
`models/checkpoints/best.pt`, which is small enough to keep in ordinary Git.
Pretrained M/L/X are offered by official Ultralytics model names and are
downloaded on demand when selected. If a pretrained model is unavailable or the
download fails, the app keeps the existing bundled nano runtime instead of
crashing. INT8 ONNX models are only selectable after their validation gates pass.

## Dataset

The reproducible data pipeline uses COCO 2017 person annotations. COCO annotations are CC BY 4.0; image licenses are the per-image Flickr licenses in COCO metadata.

- `train2017` becomes the training split.
- Annotated person images from `val2017` are shuffled with seed 42.
- Half become validation and half become an isolated test holdout.
- Crowd annotations and images without usable person boxes are excluded.
- INT8 calibration uses only `images/train`.

Download and convert:

```bash
python scripts/prepare_coco_person.py
```

Use `--overwrite` when intentionally rebuilding an existing converted directory.

For a quick pipeline check without processing the full dataset:

```bash
python scripts/prepare_coco_person.py \
  --max-train-images 500 \
  --max-val-images 100 \
  --max-test-images 100
```

Validate:

```bash
python scripts/validate_dataset.py
```

The validator checks corrupt images, image/label pairing, missing and empty annotations, unknown classes, five-field YOLO syntax, normalized coordinate ranges, non-positive boxes, duplicates, and tiny objects. The report is written to `reports/dataset_validation.json`.

Run the read-only raw/prepared audit before training:

```bash
python scripts/audit_dataset.py
```

The audit writes JSON and Markdown reports, split-level person density,
resolution and relative box distributions, proof of the current empty-image
filter, per-image expected/actual label integrity, cross-split SHA-256 and
perceptual-hash leakage, and deterministic annotation contact sheets. It
reports missing raw/prepared inputs explicitly instead of fabricating
statistics. Domain collection follows
[`docs/annotation_policy.md`](docs/annotation_policy.md) and the grouped split
protocol in [`docs/dataset_split_protocol.md`](docs/dataset_split_protocol.md).

## Analysis notebook

Start Jupyter and open `notebooks/VisionTrack_Analysis.ipynb`:

```bash
jupyter notebook
```

The notebook covers source/license, split distribution, objects per image, resolutions, box sizes, validator results, annotated samples, ONNX preprocessing, baseline predictions/errors, transfer-learning reports, and pretrained/fine-tuned/pruned/quantized comparisons. Production runtime logic stays in `src/vision_track`.

## Baseline and transfer learning

Run the reproducible baseline-readiness stage:

```bash
python scripts/run_baseline_stage.py --device cuda
```

This command validates the unchanged control dataset, evaluates pretrained
`yolo26n.pt` on the full validation split, records separate standard-evaluator
and project-runtime-threshold metrics, and runs one epoch on deterministic
256-train/64-val linked smoke lists. The smoke reloads and validates both
`best.pt` and `last.pt`. Reports are written under `reports/baseline_runs/`;
weights and Ultralytics run data stay under the Git-ignored
`models/training_runs/` tree. The test split is not used by this stage.
The latest completed readiness report is
[`baseline_20260714T180333Z`](reports/baseline_runs/baseline_20260714T180333Z/report.md).

Full A training is gated behind explicit confirmation. After separate
approval, train with validation after every epoch, early stopping, seed 42,
and the unchanged parameters in `configs/app.yaml`:

```bash
python scripts/train.py --device cuda --confirm-full-run
```

The full-training command still does not use test. After thresholds,
hyperparameters, pruning, and quantization choices are frozen, explicitly run
the isolated final evaluation:

```bash
python scripts/evaluate_model.py \
  --model models/training_runs/full_a/<run-id>/ultralytics/a_full/weights/best.pt \
  --split test \
  --output reports/best_test_metrics.json
```

Training never replaces the runtime checkpoint. After evaluation and model
selection, promotion is a separate explicit step with mandatory source,
destination, and expected SHA-256. The command verifies a detect checkpoint,
requires class `0=person` even for multiclass COCO checkpoints, runs a
deterministic inference smoke with person filtering on the staged checkpoint
using the project model settings and the runtime-selected device (CUDA when
available), and only then publishes it atomically:

```bash
python scripts/promote_model.py \
  --source models/training_runs/full_a/<run-id>/ultralytics/a_full/weights/best.pt \
  --destination models/checkpoints/best.pt \
  --expected-sha256 <sha256>
```

Promotion evidence is written separately under `reports/model_promotions/`.

Current model roles are intentionally separated:

- `models/checkpoints/best.pt`: bundled full-A fine-tuned nano checkpoint.
- `yolo26m.pt`: official Ultralytics pretrained balanced option, downloaded on demand.
- `yolo26l.pt`: official Ultralytics pretrained recommended GPU option, downloaded on demand.
- `yolo26x.pt`: official Ultralytics pretrained maximum-quality option, downloaded on demand.
- `models/checkpoints/experiments/yolo26l_int8.onnx`: ignored failed L INT8 experiment.
- `models/checkpoints/experiments/fine_tuned_n_int8.onnx`: ignored failed nano INT8 experiment.

The recorded SHA-256 values for reproducibility are:

| Model | Source name/path | SHA-256 |
|---|---|---|
| bundled fine-tuned N | `models/checkpoints/best.pt` | `ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa` |
| pretrained M | `yolo26m.pt` | `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7` |
| pretrained L | `yolo26l.pt` | `9fe3c544f2b19bebad7ea41e76d7ad3d88b7c2f10d11d24430c5311f6b32db26` |
| pretrained X | `yolo26x.pt` | `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92` |

The isolated test split was used once for the frozen full-A nano decision.
Pretrained M/L/X and INT8 comparisons are validation-only.

### Current validation and deployment summary

`reports/performance_metrics.json` is the canonical machine-readable summary.
COCO-person is used as the general benchmark. Fine-tuned nano demonstrates the
transfer-learning workflow, while pretrained scaling shows the deployment
quality/speed trade-off.

| Model | Split | P | R | F1 | mAP50 | mAP50-95 | GPU 1-stream FPS | GPU 2-stream FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full-A nano bundled | test once after freeze | 0.8096 | 0.6721 | 0.7345 | 0.7766 | 0.5290 | 76.74 | 64.11 |
| pretrained M | val | 0.8529 | 0.7516 | 0.7990 | 0.8555 | 0.6406 | 42.71 | 57.51 |
| pretrained L | val | 0.8533 | 0.7565 | 0.8020 | 0.8606 | 0.6499 | 36.29 | 47.38 |
| pretrained X | val | 0.8590 | 0.7735 | 0.8140 | 0.8740 | 0.6681 | 22.87 | 36.83 |
| L INT8 ONNX | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |
| fine-tuned N INT8 ONNX | val | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a |

Scaling to X improves validation quality but still does not satisfy the quality
targets simultaneously. L is selected as the recommended GPU option because it
is close to X quality while being materially faster and more realistic on the
checked hardware. X remains the maximum-quality option. The bundled nano is the
portable fallback and the transfer-learning deliverable. FPS depends on the
user's device, so the application exposes explicit model selection instead of
hidden automatic hardware-based switching.

Ultralytics performs its own resize, letterbox, tensor conversion, and normalization. The project does not apply a second manual normalization path to PyTorch inference.

## Structured pruning

`scripts/prune.py` uses Torch-Pruning dependency graphs and magnitude importance to physically remove coupled channels. It rejects a run if parameter count and operations do not decrease. It then performs short recovery fine-tuning.

```bash
python scripts/prune.py
```

Output:

- `models/checkpoints/best_pruned.pt`
- `reports/pruning_report.json`

This is structural pruning, not a PyTorch mask that merely writes zeros.

## ONNX INT8 quantization

The quantization workflow exports FP32 ONNX and calibrates static QDQ INT8 using
only train-split images. INT8 is written as an experimental artifact first:

```bash
models/checkpoints/experiments/<model>_int8.onnx
```

Only after the artifact loads, runs inference, loses at most one absolute
percentage point of F1 and mAP50-95 on validation, and improves CPU throughput
by at least 15% may it be promoted to
`models/checkpoints/best_quantized.onnx`.

Current INT8 status:

- Pretrained L INT8 is rejected: FP32 export was 99.6 MB, INT8 was 26.2 MB,
  but validation precision/recall/mAP were all zero.
- Fine-tuned nano INT8 is rejected: FP32 export was 9.8 MB, INT8 was 3.1 MB,
  load/inference smoke passed, but validation precision/recall/F1/mAP were all
  zero.
- `models/checkpoints/best_quantized.onnx` is therefore not produced.

Tracking scores such as IDF1, HOTA, MOTA, and ID switches are not reported because COCO detection labels do not contain ground-truth trajectories.

## Benchmark

Provide one or two local videos; input frames are normalized to 1280×720 for measurement:

```bash
python scripts/benchmark.py data/demo/video-a.mp4 data/demo/video-b.mp4
```

The script records hardware, OS, Python/dependency versions, model/backend/device, resolution, image size, stream count, warmup/measured frames, inference latency, end-to-end latency, per-stream and aggregate FPS, file size, seed, and scenario status.

Scenarios include:

- one 720p local stream;
- two simultaneous 720p local streams;
- CPU fallback;
- CUDA or MPS when actually available;
- `best.pt`;
- `best_pruned.pt`;
- `best_quantized.onnx`;
- a missing source alongside a successful working-source benchmark.

Unavailable devices or missing trained artifacts are recorded as `skipped`; no values are invented. Detection precision/recall/F1 are read from `reports/best_test_metrics.json` when present.

## Demo artifacts

After trained artifacts and two real videos are available:

```bash
python scripts/generate_demo.py data/demo/video-a.mp4 data/demo/video-b.mp4
```

This produces real model output:

- `reports/demo_results/roi_counting_example.png`
- `reports/demo_results/multi_stream_demo.mp4`

The repository does not contain placeholder images, videos, checkpoints, or fabricated benchmark values.

## Streamlit application

Launch:

```bash
streamlit run app.py
```

Sidebar controls:

- add multiple uploaded videos;
- refresh local camera devices and add a camera;
- add HTTP/RTSP URLs;
- select/remove a stream;
- confidence and IoU;
- ByteTrack activation, lost-buffer, and matching thresholds;
- detection/tracking/counting toggles;
- detector model from the registry;
- start, stop, restart, reset counters, start all, stop all.

The main area shows a stream grid and selected-stream detail: rendered frame, source/lifecycle/error state, actual device/backend, FPS, model and end-to-end latency, dropped-frame rate, in/out counts, and occupancy.

Local videos, `webcam://N` devices, HTTP URLs, and RTSP URLs are backend inputs. A local camera is opened by OpenCV on the machine running Streamlit; Refresh probes indices 0 through 9 without reopening a camera that is already active in the session. On Windows, camera opening validates the first frame with MSMF, then falls back to DSHOW and CAP_ANY. Browser `getUserMedia`, browser-to-server frame transfer, and WebRTC are out of scope. Rendered previews use one persistent HTML canvas per stream and a latest-only WebSocket bound to `127.0.0.1`.

The engine and preview session token are stored in `st.session_state`; normal Streamlit reruns reuse detector, scheduler, reader, tracker, counter, and preview bindings. A preview socket reconnect or iframe remount within that session reuses a compatible cached JPEG. A hard browser reload (`Ctrl+R`) resets Streamlit Session State and isn't expected to preserve streams or engine state.

The loopback preview assumes Streamlit and the browser run on the same machine. A remote deployment requires a proxied WebSocket endpoint with TLS (`wss://`) rather than the loopback URL.

## Artifacts

`models/checkpoints/config.yaml`, `reports/performance_metrics.json`, and `logs/app_errors.log` are present from setup. The metrics file starts with `status: not_measured` and null measurements. The following are generated only by successful real workflows:

```text
models/checkpoints/best.pt
models/checkpoints/best_pruned.pt
models/checkpoints/best_quantized.onnx  # only after an INT8 gate passes
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
reports/demo_results/demo_metadata.json
```

## Tests and audit checks

Run:

```bash
python -c "import torch, supervision, cv2, streamlit"
pytest -q
```

Tests cover device priority/fallback, annotations, detection filtering/format, frame dropping, independent tracking/counting state, composite IDs, line crossing and re-entry, metrics/schema, lifecycle, credential masking, local video and fake-webcam integration, webcam reconnect/release, two streams, broken plus working sources, EOF, stop/replay/remove/replace, latest-only WebSocket preview semantics, Streamlit state reuse, ONNX inference, audit imports, and Streamlit startup.

The real pretrained CPU inference test is opt-in because it may download model weights:

```bash
VISIONTRACK_RUN_MODEL_TESTS=1 pytest -m slow
```

Manual audit:

```bash
python -c "import torch, supervision, cv2, streamlit"
pytest -q
streamlit run app.py
```

Real webcam, driver fallback, unplug/reconnect, device ownership, and visual-drift
checks are tracked separately in
[`docs/webcam_hardware_smoke.md`](docs/webcam_hardware_smoke.md). Its entries are
deliberately `NOT RUN` until executed on the target hardware with recorded evidence.

Audit performance thresholds are precision ≥ 0.85, recall ≥ 0.80, F1 ≥ 0.85, and average per-stream FPS ≥ 15 on 720p. These are targets, not hard-coded claims; `reports/performance_metrics.json` must contain measured values from the target hardware and isolated detection test set.

## Logging and failures

Errors are written to `logs/app_errors.log` with timestamp, stream ID, source type, lifecycle state, exception type/message, and traceback for unexpected failures. RTSP user information and common token/password query parameters are masked.

Handled failures include missing/corrupt files, unavailable cameras or URLs, webcam/RTSP disconnects, decoder errors, local EOF, model/backend load errors, worker exceptions, invalid ROIs, and invalid configuration. A failed stream does not stop readers or tracking state for other streams.

## Limitations

- Browser webcam capture is not required or implemented; webcam sources refer to devices attached to the Streamlit host.
- Hard browser reload does not preserve streams or engine state; reconnect support is scoped to the same Streamlit session.
- Preview registry entries from unexpectedly disappeared Streamlit sessions may remain until process shutdown. Patch 1 intentionally has no TTL cleanup.
- OpenCV wheels are CPU-only; GPU acceleration is detector inference through PyTorch CUDA/MPS or an installed ONNX Runtime provider.
- RTSP behavior depends on the OpenCV/FFmpeg build and server codec.
- Supervision 0.29.1 still provides `sv.ByteTrack`, but marks it deprecated for a future release; the exact pin prevents an unreviewed removal.
- Performance thresholds require real trained artifacts, representative videos, and target hardware.
- Full COCO conversion, training, pruning, and calibration require substantial disk, compute, and time.

## Troubleshooting

- `python --version` must report Python 3.13.x.
- If `torch.cuda.is_available()` is false, verify the NVIDIA driver and reinstall the PyTorch build selected for the machine.
- If MPS is unavailable, verify Apple Silicon, supported macOS, and an MPS-enabled PyTorch wheel.
- If RTSP fails, test the URL and codec with another client; credentials will be masked in logs.
- If a local camera is missing, close other applications that may hold it, click Refresh cameras, and retry; Windows tries MSMF, then DSHOW, then CAP_ANY.
- If MP4 writing fails, install an OpenCV/FFmpeg build with an MP4 encoder or use compatible input/output codecs.
- If ONNX is not offered in the UI, run `scripts/quantize.py` successfully first.
- If the app appears delayed, inspect dropped-frame rate and lower stream count or inference image size; the queue intentionally drops stale frames to preserve freshness.
