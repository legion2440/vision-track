# Live-source smoke — 2026-07-14

## Environment

- Base commit: `6e565588b15deb37ca9a19a64f46bebea8606986` plus the
  credential-mask correction documented below.
- Host: Microsoft Windows 11 Pro `10.0.26200` (build `26200`).
- Python: `3.13.5`.
- OpenCV: `4.13.0.92` with FFmpeg capture support.
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU, PyTorch CUDA backend.
- Camera: `USB2.0 HD UVC WebCam`, index `0`; validated backend MSMF
  (`1400`), first frame `640x480`.
- Controlled RTSP: MediaMTX `1.19.2` with an FFmpeg-generated H.264
  `640x360@15 FPS` publisher and TCP-only RTSP transport.

## Endpoint preflight and application results

| Source | Probe/application result |
|---|---|
| `https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8` | PASS. `ffprobe` found H.264 variants; VisionTrack reached `ACTIVE`, displayed frames, and passed stop/restart. |
| Apple advanced example HLS | BLOCKED. The 35-second `ffprobe` preflight timed out from this network, so it was not added to the app. |
| Google Big Buck Bunny MP4 from the requested bucket | BLOCKED. The endpoint returned HTTP 403. |
| `https://media.w3.org/2010/05/sintel/trailer.mp4` | PASS for progressive HTTP compatibility. H.264 `854x480@24 FPS`; VisionTrack reached `ACTIVE` and displayed frames. |
| Three supplied public RTSP candidates | BLOCKED externally. Two refused TCP port 554 and one returned network unreachable. This was not treated as an application failure. |
| `rtsp://vision_user:vision_password@127.0.0.1:8554/test` | PASS through the controlled relay. VisionTrack reached `ACTIVE` over TCP and displayed frames. |
| `.smoke/trailer.mp4` during the run | PASS through the local-file reader; it reached `ACTIVE`, played with local pacing, and ended at `EOF`. |

The progressive HTTP MP4 is classified as reconnectable because source type is
derived from the HTTP scheme. After a complete fast decode it reconnects and
replays instead of producing local-file `EOF`. This existing protocol-based
semantic was recorded as a limitation and was not changed by this smoke patch.

## Lifecycle and fault evidence

- HLS: `CREATED -> ACTIVE -> STOPPED -> ACTIVE` through Start, Stop, and
  Restart.
- Controlled RTSP: stopping the publisher caused
  `ACTIVE -> RECONNECTING -> FAILED`. The first failure was logged at
  `00:24:58.231`; the bounded budget reached `FAILED` at `00:25:30.990`.
  Restarting the publisher and pressing Restart returned the same stream to
  `ACTIVE`.
- Credentials: searches for the literal username and password returned no
  matches in `logs/app_errors.log` or either Streamlit stdout/stderr capture.
  Expected error lines contained
  `rtsp://***:***@127.0.0.1:8554/test`.
- Camera: manual Refresh found only Camera 0. Add was disabled after the camera
  was added. Start, Stop, Restart, and active Remove all completed while HLS and
  RTSP continued running.
- Camera ownership: after active Remove, and again after forcibly terminating
  Streamlit with Camera 0 active, a separate Python process immediately opened
  Camera 0 through MSMF and received a `640x480` frame.
- Mixed pipeline: webcam, local MP4, HLS, and RTSP were simultaneously
  `ACTIVE`. The local file later reached `EOF` without stopping the three live
  sources.

Windows Camera itself and a physical unplug/replug were not used. Results that
specifically require those actions remain `BLOCKED` or `NOT RUN` in the hardware
checklist.

## 30-minute live soak

Order in every cell is `FPS / end-to-end latency / dropped rate`.

| Minute | Mux HLS | Camera 0 | Local RTSP |
|---:|---:|---:|---:|
| 0 | 11.1 / 167.4 ms / 96.0% | 11.5 / 187.9 ms / 71.0% | 11.3 / 177.8 ms / 65.8% |
| 5 | 11.8 / 124.6 ms / 95.7% | 10.5 / 131.5 ms / 71.9% | 11.2 / 127.2 ms / 58.4% |
| 10 | 12.1 / 131.4 ms / 95.9% | 9.5 / 158.3 ms / 71.9% | 10.5 / 146.0 ms / 58.5% |
| 15 | 11.0 / 141.6 ms / 95.9% | 10.6 / 159.3 ms / 71.9% | 10.6 / 164.0 ms / 58.4% |
| 20 | 9.0 / 154.9 ms / 95.9% | 10.8 / 164.7 ms / 71.8% | 10.8 / 164.7 ms / 58.2% |
| 25 | 12.5 / 135.1 ms / 95.9% | 12.9 / 127.4 ms / 71.9% | 11.9 / 151.2 ms / 58.2% |
| 30 | 12.8 / 132.4 ms / 95.9% | 13.0 / 157.3 ms / 71.9% | 12.9 / 131.6 ms / 58.2% |

All three streams remained `ACTIVE`. No latency series increased monotonically,
and every final end-to-end value was below its baseline. The high dropped rates
are expected from the latest-only queues under shared inference load; they
stabilized instead of producing delayed playback or catch-up.

## Defect found and corrected

The credential masker correctly produced `***`, but Streamlit Markdown treated
those asterisks as formatting in stream titles, detail headings, and errors. The
visible result was `rtsp://:@host/path`, although the secret itself was never
shown. UI Markdown sinks now escape the mask asterisks. Unit coverage and the
post-fix browser run both show `rtsp://***:***@127.0.0.1:8554/test`.

## Evidence

- [HLS active](evidence/live-source-2026-07-14/hls-active.png)
- [Four-source mixed run](evidence/live-source-2026-07-14/mixed-four-active.png)
- [30-minute final state](evidence/live-source-2026-07-14/soak-30m-end.png)
- [Controlled RTSP reconnect/FAILED log excerpt](evidence/live-source-2026-07-14/controlled-rtsp-errors.log)
