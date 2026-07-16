# yolo26l.pt zero-shot val + 720p pipeline report

Generated: 2026-07-16T06:17:04.786808+00:00

## Standard val evaluation

- P: 0.853276
- R: 0.756544
- F1: 0.802004
- mAP50: 0.860579
- mAP50-95: 0.649853

## Confidence sweep argmax F1 on val

- selected confidence: 0.30
- P: 0.841431
- R: 0.749954
- F1: 0.793063
- detections: 4837
- FP/image: 0.569837

## 720p pipeline benchmark

| scenario | FPS aggregate | FPS/stream | infer ms/batch | render ms/batch | jpeg ms/batch | total ms/batch |
|---|---:|---:|---:|---:|---:|---:|
| yolo26l_720p_cpu_1_stream | 5.825 | 5.825 | 161.398 | 0.861 | 7.847 | 171.666 |
| yolo26l_720p_cpu_2_stream | 5.990 | 2.995 | 311.950 | 1.529 | 14.989 | 333.896 |
| yolo26l_720p_cuda_1_stream | 36.292 | 36.292 | 19.792 | 0.646 | 5.870 | 27.554 |
| yolo26l_720p_cuda_2_stream | 47.379 | 23.689 | 24.193 | 1.113 | 12.074 | 42.213 |

## Requirement check

- Standard P>=0.85: True
- Standard R>=0.80: False
- Standard F1>=0.85: False
- Operational sweep P>=0.85: False
- Operational sweep R>=0.80: False
- Operational sweep F1>=0.85: False
- Any scenario meets P/R/F1/FPS together using standard eval quality: False
- Any scenario meets P/R/F1/FPS together using operational sweep quality: False

Notes: test split was not used; model was not promoted; dataset and main config were not intentionally changed. Ultralytics cache generated during eval was removed.
