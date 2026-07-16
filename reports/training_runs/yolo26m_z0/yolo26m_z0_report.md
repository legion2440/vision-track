# yolo26m.pt zero-shot val + 720p pipeline report

Generated: 2026-07-16T06:06:16.644391+00:00

## Standard val evaluation

- P: 0.852864
- R: 0.751612
- F1: 0.799043
- mAP50: 0.855514
- mAP50-95: 0.640554

## Confidence sweep argmax F1 on val

- selected confidence: 0.35
- P: 0.877723
- R: 0.727474
- F1: 0.795567
- detections: 4498
- FP/image: 0.408618

## 720p pipeline benchmark

| scenario | FPS aggregate | FPS/stream | infer ms/batch | render ms/batch | jpeg ms/batch | total ms/batch |
|---|---:|---:|---:|---:|---:|---:|
| yolo26m_720p_cpu_1_stream | 7.108 | 7.108 | 130.039 | 0.830 | 8.048 | 140.693 |
| yolo26m_720p_cpu_2_stream | 7.004 | 3.502 | 263.124 | 1.352 | 15.308 | 285.570 |
| yolo26m_720p_cuda_1_stream | 42.710 | 42.710 | 15.295 | 0.633 | 6.075 | 23.414 |
| yolo26m_720p_cuda_2_stream | 57.507 | 28.753 | 16.691 | 1.023 | 12.026 | 34.779 |

## Requirement check

- Standard P>=0.85: True
- Standard R>=0.80: False
- Standard F1>=0.85: False
- Operational sweep P>=0.85: True
- Operational sweep R>=0.80: False
- Operational sweep F1>=0.85: False
- Any scenario meets P/R/F1/FPS together using standard eval quality: False
- Any scenario meets P/R/F1/FPS together using operational sweep quality: False

Notes: test split was not used; model was not promoted; dataset and main config were not intentionally changed. Ultralytics cache generated during eval was removed.
