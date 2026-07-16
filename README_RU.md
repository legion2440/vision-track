# VisionTrack

Многопоточное приложение для детекции, трекинга и подсчёта людей на видео. Система принимает локальные файлы, камеры, HTTP- и RTSP-потоки, обрабатывает их через общий YOLO detector и хранит независимое состояние ByteTrack и счётчиков для каждого источника.

[English version](README_EN.md) · [Обучение и оценка](docs/TRAINING.md)

## 📋 TOC

- [🚀 Как запустить](#-как-запустить)
- [📝 О проекте](#-о-проекте)
- [🔗 Ссылки](#-ссылки)
- [✨ Возможности](#-возможности)
- [🎥 Источники видео](#-источники-видео)
- [🧠 Модели](#-модели)
- [⚙️ Архитектура](#️-архитектура)
- [📊 Результаты](#-результаты)
- [🎬 Demo](#-demo)
- [🧪 Проверка](#-проверка)
- [📁 Структура](#-структура)
- [⚠️ Ограничения](#️-ограничения)
- [🧑‍💻 Автор](#-автор)

## 🚀 Как запустить

Требуется Python `3.13.*`.

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

После запуска открыть локальный адрес, который выведет Streamlit.

Приложение выбирает устройство в порядке:

```text
CUDA → Apple MPS → CPU
```

## 📝 О проекте

VisionTrack решает четыре связанные задачи:

- детекция людей;
- присвоение и сохранение track ID;
- подсчёт пересечений линии;
- расчёт текущей заполненности ROI.

Основной сценарий — несколько одновременных видеопотоков с общей моделью и отдельным состоянием каждого источника.

Streamlit используется для интерфейса, настроек и метрик. Чтение кадров, inference, tracking и preview работают вне цикла перерисовки Streamlit.

## 🔗 Ссылки

- Основной репозиторий: `https://01.tomorrow-school.ai/git/nyestaye/vision-track`
- GitHub mirror: `https://github.com/legion2440/vision-track`
- Обучение и оценка: [`docs/TRAINING.md`](docs/TRAINING.md)
- Analysis notebook: [`notebooks/VisionTrack_Analysis.ipynb`](notebooks/VisionTrack_Analysis.ipynb)
- Итоговые метрики: [`reports/performance_metrics.json`](reports/performance_metrics.json)

## ✨ Возможности

### Detection и tracking

- YOLO26 для person detection;
- ByteTrack из библиотеки `supervision`;
- независимые tracker ID для каждого потока;
- история траекторий;
- отдельные настройки confidence, IoU и ByteTrack;
- включение и отключение detection, tracking и counting для каждого источника.

### Counting

- счётчики пересечения линии `IN` и `OUT`;
- polygon ROI;
- текущая occupancy;
- отдельный сброс счётчиков;
- сохранение totals при сбросе tracker-dependent состояния.

### Multi-stream

- одна модель на все активные потоки;
- отдельный reader для каждого источника;
- latest-frame queue размером `1`;
- устаревшие кадры отбрасываются вместо накопления задержки;
- ошибка одного потока не останавливает остальные.

### Интерфейс

- выбор detector model;
- добавление и удаление источников;
- выбор потока для детального просмотра;
- Start, Stop, Restart и Reset;
- Start all и Stop all;
- FPS, inference latency и end-to-end latency;
- dropped-frame rate;
- `IN`, `OUT` и occupancy;
- фактические device, backend и lifecycle state.

## 🎥 Источники видео

Поддерживаются:

- локальные `MP4`, `AVI`, `MOV`, `MKV` и `WebM`;
- камеры, подключённые к машине, где запущен Streamlit;
- HTTP video URL;
- RTSP.

Локальная камера открывается через OpenCV на стороне Streamlit host.

Browser `getUserMedia`, WebRTC и передача кадров из браузера на backend не используются.

На Windows камера проверяется через MSMF, затем DSHOW и `CAP_ANY`.

## 🧠 Модели

### Bundled model

В репозитории хранится fine-tuned nano:

```text
models/checkpoints/best.pt
```

```text
SHA-256:
ab09e99711a9057442691bde03802c86d6b0e63a61f1957b1c013f78134073aa
```

Это модель по умолчанию и переносной fallback.

### Download-on-demand

В selector также доступны официальные pretrained YOLO26:

| Модель | Роль                          |
|--------|-------------------------------|
| M      | сбалансированный вариант      |
| L      | рекомендуемый вариант для GPU |
| X      | максимальное качество         |

M, L и X не хранятся в Git. Ultralytics скачивает их при первом выборе.

Если загрузка или скачивание не удались, приложение сохраняет реально работающую модель и возвращает selector к ней.

### INT8

Static INT8 ONNX был проверен для pretrained L и fine-tuned N.

Оба artifact загружались и выполняли inference, но получили нулевые validation-метрики через текущий ONNX evaluation path. Они не прошли acceptance gate.

Поэтому production artifact:

```text
models/checkpoints/best_quantized.onnx
```

не создан и не доступен в selector.

## ⚙️ Архитектура

```text
reader A -> latest queue A --\
reader B -> latest queue B ----> shared scheduler -> shared detector
reader N -> latest queue N --/                         |
                                                       +-> tracker/counter/render A
                                                       +-> tracker/counter/render B
                                                       +-> tracker/counter/render N
```

У каждого потока отдельные:

- reader;
- latest-frame queue;
- lifecycle;
- ByteTrack;
- trajectories;
- line counter;
- ROI occupancy;
- настройки и ошибки.

Общий scheduler собирает последние доступные кадры, выполняет один batch inference и возвращает результаты в нужные stream contexts.

Preview работает по отдельному каналу:

```text
rendered frame
    -> latest-only loopback WebSocket
    -> JPEG
    -> persistent browser canvas
```

Streamlit отвечает за layout, controls, metrics и lifecycle text. Inference не запускается внутри Streamlit rerun.

## 📊 Результаты

COCO 2017 person использован как общий протокол оценки.

| Модель       | Split                                | Precision | Recall | F1     | mAP50  | mAP50-95 | GPU 1-stream FPS |
|--------------|--------------------------------------|----------:|-------:|-------:|-------:|---------:|-----------------:|
| Fine-tuned N | isolated test, один раз после freeze | 0.8096    | 0.6721 | 0.7345 | 0.7766 | 0.5290   | 76.74*           |
| Pretrained M | validation                           | 0.8529    | 0.7516 | 0.7990 | 0.8555 | 0.6406   | 42.71            |
| Pretrained L | validation                           | 0.8533    | 0.7565 | 0.8020 | 0.8606 | 0.6499   | 36.29            |
| Pretrained X | validation                           | 0.8590    | 0.7735 | 0.8140 | 0.8740 | 0.6681   | 22.87            |

`*` Для Fine-tuned N использовался другой scope замера скорости. Его FPS нельзя напрямую сравнивать с full-pipeline результатами M/L/X.

GPU-бенчмарки выполнены на NVIDIA RTX 4080 Laptop GPU под Windows 11.

Увеличение модели до X повысило качество, но ни одна проверенная модель не выполнила одновременно все пороги задания на полном COCO-person протоколе. Метрики сохранены без подгонки.

Полный training и evaluation pipeline описан в [`docs/TRAINING.md`](docs/TRAINING.md).

## 🎬 Demo

В репозитории сохранены реальные результаты работы модели:

```text
reports/demo_results/roi_counting_example.png
reports/demo_results/multi_stream_demo.mp4
reports/demo_results/demo_metadata.json
```

Metadata содержит model ID, SHA-256, backend, device, thresholds, hardware и параметры генерации.

## 🧪 Проверка

Проверка импортов:

```bash
python -c "import torch, supervision, cv2, streamlit"
```

Полный набор тестов:

```bash
pytest -q
```

Последний зафиксированный результат:

```text
317 passed, 1 skipped
```

Startup smoke:

```bash
streamlit run app.py
```

Опциональный real-model тест может скачать pretrained weights:

```bash
VISIONTRACK_RUN_MODEL_TESTS=1 pytest -m slow
```

Ошибки приложения записываются в:

```text
logs/app_errors.log
```

Логины, пароли и распространённые token-параметры в URL маскируются.

## 📁 Структура

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

## ⚠️ Ограничения

- Loopback WebSocket использует `127.0.0.1`; стандартный preview рассчитан на браузер и Streamlit на одной машине.
- Для remote deployment нужен проксированный `wss://` endpoint.
- Hard reload браузера сбрасывает Streamlit Session State и активные потоки.
- OpenCV wheel не использует CUDA; GPU acceleration относится к PyTorch inference.
- RTSP зависит от OpenCV/FFmpeg build и codec сервера.
- FPS зависит от модели, device, decoder, количества потоков и содержимого видео.
- В COCO detection labels нет ground-truth trajectories, поэтому IDF1, HOTA, MOTA и ID switches не рассчитываются.
- INT8 ONNX не включён, потому что validation gate не пройден.
- Hardware smoke для камер требует ручного запуска на целевой машине.

## 🧑‍💻 Автор
- Nazar Yestayev (@nyestaye)
