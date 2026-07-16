# yolo26x.pt zero-shot val + 720p pipeline report

Generated: 2026-07-16T06:41:24.830251+00:00

## Standard val evaluation

- P: 0.858983
- R: 0.773540
- F1: 0.814025
- mAP50: 0.874005
- mAP50-95: 0.668052

## Confidence sweep argmax F1 on val

- selected confidence: 0.30
- P: 0.845982
- R: 0.768196
- F1: 0.805215
- detections: 4928
- FP/image: 0.563893

## 720p pipeline benchmark

| scenario | FPS aggregate | FPS/stream | infer ms/batch | render ms/batch | jpeg ms/batch | total ms/batch |
|---|---:|---:|---:|---:|---:|---:|
| yolo26x_720p_cpu_1_stream | 2.811 | 2.811 | 345.123 | 0.935 | 8.031 | 355.749 |
| yolo26x_720p_cpu_2_stream | 2.158 | 1.079 | 900.806 | 1.521 | 16.655 | 926.594 |
| yolo26x_720p_cuda_1_stream | 22.865 | 22.865 | 32.384 | 0.876 | 8.496 | 43.734 |
| yolo26x_720p_cuda_2_stream | 36.833 | 18.417 | 30.242 | 1.387 | 15.671 | 54.299 |

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
