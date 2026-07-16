# VisionTrack

A multi-stream application for real-time person detection, tracking, line-crossing counts, and ROI occupancy. It accepts local files, webcams, HTTP, and RTSP sources, processes them through one shared YOLO detector, and keeps independent ByteTrack and counting state for each stream.

[Русская версия](README.md) · [Training and evaluation](docs/TRAINING.md)

## 📋 TOC

- [🚀 Run](#-run)
- [📝 About](#-about)
- [🔗 Links](#-links)
- [✨ Features](#-features)
- [🎥 Video sources](#-video-sources)
- [🧠 Models](#-models)
- [⚙️ Architecture](#️-architecture)
- [📊 Results](#-results)
- [🎬 Demo](#-demo)
- [🧪 Verification](#-verification)
- [📁 Structure](#-structure)
- [⚠️ Limitations](#️-limitations)
- [🧑‍💻 Author](#-author)

## 🚀 Run

Python `3.13.*` is required.

### Windows / NVIDIA CUDA

```bash
git clone https://01.tomorrow-school.ai/git/nyestaye/vision-track.git
cd vision-track

python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt

streamlit run app.py
```

### CPU

```bash
python -m venv .venv
source .venv/Scripts/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

streamlit run app.py
```

### macOS Apple Silicon

```bash
python3.13 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-mps.txt

streamlit run app.py
```

Open the local URL printed by Streamlit.

Device selection order:

```text
CUDA → Apple MPS → CPU
```

## 📝 About

VisionTrack combines four related tasks:

- person detection;
- persistent track IDs;
- line-crossing counts;
- current ROI occupancy.

The main use case is simultaneous processing of several video sources through one detector while keeping separate tracking, counting, settings, and lifecycle state for every stream.

Streamlit is used for interface controls and metrics. Frame reading, inference, tracking, and preview transport run outside the Streamlit rerun loop.

## 🔗 Links

- Primary repository: `https://01.tomorrow-school.ai/git/nyestaye/vision-track`
- GitHub mirror: `https://github.com/legion2440/vision-track`
- Training and evaluation: [`docs/TRAINING.md`](docs/TRAINING.md)
- Analysis notebook: [`notebooks/VisionTrack_Analysis.ipynb`](notebooks/VisionTrack_Analysis.ipynb)
- Final metrics: [`reports/performance_metrics.json`](reports/performance_metrics.json)

## ✨ Features

### Detection and tracking

- YOLO26 person detection;
- `supervision.ByteTrack`;
- independent tracker IDs for every stream;
- trajectory history;
- confidence, IoU, and ByteTrack settings;
- independent detection, tracking, and counting toggles.

### Counting

- line-crossing `IN` and `OUT` counters;
- polygon ROI;
- current occupancy;
- counter reset;
- tracker-state reset without losing accumulated totals.

### Multi-stream

- one detector shared by all active streams;
- one reader per source;
- latest-frame queue with capacity `1`;
- stale frames are dropped instead of accumulating latency;
- one failed source does not stop the remaining streams.

### Interface

- detector model selection;
- source addition and removal;
- selected-stream details;
- Start, Stop, Restart, and Reset;
- Start all and Stop all;
- FPS, inference latency, and end-to-end latency;
- dropped-frame rate;
- `IN`, `OUT`, and occupancy;
- actual device, backend, lifecycle, and error state.

## 🎥 Video sources

Supported inputs:

- local `MP4`, `AVI`, `MOV`, `MKV`, and `WebM` files;
- webcams attached to the machine running Streamlit;
- HTTP video URLs;
- RTSP streams.

A local webcam is opened through OpenCV on the Streamlit host.

Browser `getUserMedia`, WebRTC, and browser-to-backend frame transfer are not used.

On Windows, camera opening is tried through MSMF, then DSHOW and `CAP_ANY`.

## 🧠 Models

### Bundled model

The repository contains one fine-tuned nano model:

```text
models/checkpoints/best.pt
```

```text
SHA-256:
ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa
```

It is the default model and portable fallback.

### Download on demand

The selector also provides official pretrained YOLO26 models:

| Model | Role                   |
|-------|------------------------|
| M     | balanced               |
| L     | recommended GPU option |
| X     | maximum quality        |

M, L, and X are not stored in Git. Ultralytics downloads them when selected for the first time.

If loading or downloading fails, the application keeps the actually active model and returns the selector to that model.

### INT8

Static INT8 ONNX was tested for Pretrained L and Fine-tuned N.

Both artifacts loaded and executed inference, but produced zero validation metrics through the current ONNX evaluation path. They failed the acceptance gate.

The production artifact:

```text
models/checkpoints/best_quantized.onnx
```

is therefore not created and is not exposed in the selector.

## ⚙️ Architecture

```text
reader A -> latest queue A --\
reader B -> latest queue B ----> shared scheduler -> shared detector
reader N -> latest queue N --/                         |
                                                       +-> tracker/counter/render A
                                                       +-> tracker/counter/render B
                                                       +-> tracker/counter/render N
```

Every stream has an independent:

- reader;
- latest-frame queue;
- lifecycle;
- ByteTrack instance;
- trajectory history;
- line counter;
- ROI occupancy;
- settings and error state.

The shared scheduler collects the newest available frames, runs one detector batch, and routes each result back to the corresponding stream context.

Preview uses a separate latest-only path:

```text
rendered frame
    -> loopback WebSocket
    -> JPEG
    -> persistent browser canvas
```

Streamlit owns layout, controls, metrics, and lifecycle text. Inference does not run inside Streamlit reruns.

## 📊 Results

COCO 2017 person is used as the general evaluation protocol.

| Model        | Split                            | Precision | Recall | F1     | mAP50  | mAP50-95 | GPU 1-stream FPS |
|--------------|----------------------------------|----------:|-------:|-------:|-------:|---------:|-----------------:|
| Fine-tuned N | isolated test, once after freeze | 0.8096    | 0.6721 | 0.7345 | 0.7766 | 0.5290   | 76.74*           |
| Pretrained M | validation                       | 0.8529    | 0.7516 | 0.7990 | 0.8555 | 0.6406   | 42.71            |
| Pretrained L | validation                       | 0.8533    | 0.7565 | 0.8020 | 0.8606 | 0.6499   | 36.29            |
| Pretrained X | validation                       | 0.8590    | 0.7735 | 0.8140 | 0.8740 | 0.6681   | 22.87            |

`*` Fine-tuned N used a different speed measurement scope and should not be compared directly with the full-pipeline M/L/X figures.

GPU benchmarks were recorded on an NVIDIA RTX 4080 Laptop GPU under Windows 11.

Scaling to X improved quality, but no tested model met all assignment quality thresholds on the full COCO-person protocol. Recorded values are kept unchanged.

The complete training and evaluation workflow is documented in [`docs/TRAINING.md`](docs/TRAINING.md).

## 🎬 Demo

Real model outputs are stored in:

```text
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
reports/demo_results/demo_metadata.json
```

Metadata includes model ID, SHA-256, backend, device, thresholds, hardware, and generation parameters.

## 🧪 Verification

Import check:

```bash
python -c "import torch, supervision, cv2, streamlit"
```

Full test suite:

```bash
pytest -q
```

Latest recorded result:

```text
317 passed, 1 skipped
```

Startup smoke:

```bash
streamlit run app.py
```

The optional real-model test may download pretrained weights:

```bash
VISIONTRACK_RUN_MODEL_TESTS=1 pytest -m slow
```

Application errors are written to:

```text
logs/app_errors.log
```

Credentials and common token/password query parameters are masked.

## 📁 Structure

```text
vision-track/
├── app.py
├── configs/
│   └── app.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── demo/
├── docs/
│   ├── TRAINING.md
│   ├── annotation_policy.md
│   ├── dataset_split_protocol.md
│   └── webcam_hardware_smoke.md
├── models/
│   └── checkpoints/
│       ├── best.pt
│       └── config.yaml
├── notebooks/
│   └── VisionTrack_Analysis.ipynb
├── reports/
│   ├── performance_metrics.json
│   └── demo_results/
├── scripts/
├── src/
│   └── vision_track/
├── tests/
├── logs/
│   └── app_errors.log
├── README.md
├── README_EN.md
├── requirements.txt
├── requirements-cuda.txt
└── requirements-mps.txt
```

## ⚠️ Limitations

- The loopback WebSocket uses `127.0.0.1`; the default preview assumes that browser and Streamlit run on the same machine.
- Remote deployment requires a proxied `wss://` endpoint.
- A hard browser reload resets Streamlit Session State and active streams.
- The OpenCV wheel does not use CUDA; GPU acceleration applies to PyTorch inference.
- RTSP behavior depends on the OpenCV/FFmpeg build and server codec.
- FPS depends on the selected model, device, decoder, stream count, and video content.
- COCO detection labels do not contain ground-truth trajectories, so IDF1, HOTA, MOTA, and ID switches are not reported.
- INT8 ONNX is disabled because its validation gate failed.
- Webcam hardware smoke requires a manual run on the target machine.

## 🧑‍💻 Author

**Nazar Yestayev**
- Nazar Yestayev (@nyestaye)
