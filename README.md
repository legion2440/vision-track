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

The UI and benchmark display the actual device and backend/provider. Available UI backends are derived from real artifacts: PyTorch is available through `best.pt` or the pretrained `yolo26n.pt`; ONNX Runtime appears only when `best_quantized.onnx` exists.

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

## Analysis notebook

Start Jupyter and open `notebooks/VisionTrack_Analysis.ipynb`:

```bash
jupyter notebook
```

The notebook covers source/license, split distribution, objects per image, resolutions, box sizes, validator results, annotated samples, ONNX preprocessing, baseline predictions/errors, transfer-learning reports, and pretrained/fine-tuned/pruned/quantized comparisons. Production runtime logic stays in `src/vision_track`.

## Baseline and transfer learning

Evaluate the pretrained YOLO26 nano model on validation data:

```bash
python scripts/evaluate_baseline.py \
  --model yolo26n.pt \
  --split val \
  --output reports/baseline_val_metrics.json
```

Train with validation after every epoch, early stopping, deterministic seed 42, and the parameters in `configs/app.yaml`:

```bash
python scripts/train.py --skip-final-test
```

During model development, use `--skip-final-test`. After thresholds, hyperparameters, pruning, and quantization choices are frozen, run the training/final evaluation workflow without that flag, or explicitly evaluate:

```bash
python scripts/evaluate_model.py \
  --model models/checkpoints/best.pt \
  --split test \
  --output reports/best_test_metrics.json
```

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

The quantization script chooses `best_pruned.pt` when available, otherwise `best.pt`. It exports FP32 ONNX, calibrates static QDQ INT8 using only train-split images, checks both ONNX models, loads the INT8 model in ONNX Runtime, and executes real inference before declaring success.

```bash
python scripts/quantize.py
```

Output:

- `models/checkpoints/best_quantized.onnx`
- `reports/quantization_report.json`

Compare validation artifacts:

```bash
python scripts/compare_artifacts.py --split val
```

Run the isolated final comparison only after all choices are frozen:

```bash
python scripts/compare_artifacts.py \
  --split test \
  --acknowledge-test-isolation
```

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
- available backend;
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
models/checkpoints/best_quantized.onnx
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
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
