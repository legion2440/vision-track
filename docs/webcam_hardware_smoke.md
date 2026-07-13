# Webcam hardware smoke checklist

This checklist covers behavior that fake captures and automated tests cannot prove:
real device enumeration, driver behavior, physical disconnect/reconnect, visual
freshness, and operating-system camera ownership.

Rows that were not exercised remain explicitly `NOT RUN` or `BLOCKED`. Do not
mark an item passed without recording the machine, camera, backend, input
sources, date, and observed evidence. The 2026-07-14 execution details and
measurement table are in [the live-source smoke report](live_source_smoke_2026-07-14.md).

## Preconditions

- Windows 11 host running Streamlit locally.
- At least one physical webcam attached to the Streamlit host.
- One playable local video, one reachable HTTP video URL, and one reachable RTSP
  stream for the coexistence scenario.
- A second camera application, such as Windows Camera, for ownership checks.
- A new application process and a new Streamlit session.

Start the application:

```bash
streamlit run app.py
```

Record the environment before testing:

```text
Date/time: 2026-07-14, Asia/Qyzylorda
Commit: 6e565588b15deb37ca9a19a64f46bebea8606986 plus the tested mask patch
Windows version: Windows 11 Pro 10.0.26200, build 26200
Python version: 3.13.5
OpenCV version: 4.13.0.92
Camera model/index: USB2.0 HD UVC WebCam / 0
GPU/device: NVIDIA GeForce RTX 4080 Laptop GPU / cuda:0
Local video: .smoke/trailer.mp4 (W3C Sintel trailer during execution)
HTTP source: Mux x36xhzz HLS; W3C Sintel progressive MP4
RTSP source: local MediaMTX H.264 640x360@15 FPS over TCP
```

## Checklist

Use `PASS`, `FAIL`, or `BLOCKED` and add concrete evidence for every row.

| Result | Scenario | Required observation | Evidence |
|---|---|---|---|
| PASS | Manual discovery | No camera probe occurs before clicking **Refresh cameras**. Clicking it probes indices 0–9 and lists available devices. | Initial list was empty. Manual Refresh found only Camera 0; OpenCV warnings for unavailable indices appeared only after the click. |
| BLOCKED | Probe release | After Refresh completes, Windows Camera can immediately open every discovered device that is not running in VisionTrack. | Camera 0 was successfully opened by VisionTrack after discovery, proving the probe released it, but Windows Camera itself was not used. |
| PASS | Duplicate protection | After adding Camera N, **Add camera** is disabled for N and a second stream for the same index cannot be created in the session. | Camera 0 changed to `added`/`in use`; **Add camera** remained disabled. |
| NOT RUN | Cold start | Start Camera N with detection enabled. `PREPARING` is non-blocking, then the stream reaches `CONNECTING`/`ACTIVE`; webcam capture begins after detector readiness without local-file first-frame waiting. | |
| BLOCKED | Live freshness | Run Camera N for at least five minutes with visible motion and a wall clock in view. End-to-end latency does not trend upward and preview motion does not drift progressively behind reality. | Camera ran for 30 minutes with stable 127–188 ms measured latency, but no physical wall clock was framed, so visual drift was not independently proven. |
| BLOCKED | Stop release | Stop Camera N, then immediately open it in Windows Camera. The other application obtains the device and VisionTrack publishes no new frames. | Stop reached `STOPPED` and preview showed `Stopped`; Windows Camera was not used before Restart. |
| PASS | Restart | Close the other application and restart Camera N. A new capture opens and frames resume without reloading an already prepared detector. | Stop followed by Restart returned Camera 0 to `ACTIVE`; other live streams stayed responsive. |
| BLOCKED | Remove release | Remove an active Camera N stream, then open the device in Windows Camera. No reader or reconnect activity remains. | Active Remove succeeded; an independent Python process immediately reopened MSMF and read `640x480`. Windows Camera itself was not used. |
| BLOCKED | Shutdown release | Start Camera N, terminate Streamlit normally with `Ctrl+C`, and open the device in Windows Camera. The device is released. | A forced process termination with Camera 0 active was followed by an immediate successful MSMF `640x480` open in another process. Normal Ctrl+C and Windows Camera were not used. |
| NOT RUN | Successful reconnect | While active, disconnect or disable the camera long enough to enter `RECONNECTING`, then restore it before the configured budget expires. The same stream returns to `ACTIVE`. | |
| NOT RUN | Bounded failure | Keep the device unavailable. Reconnect attempts stop after the configured limit and the stream reaches `FAILED`; it does not loop forever. | |
| NOT RUN | Stable budget reset | After one reconnect, keep the camera healthy for at least 30 consecutive frames and at least 3 seconds, then disconnect it again. A fresh reconnect budget is available. | |
| NOT RUN | Unstable budget retained | Reconnect for fewer than 30 frames or fewer than 3 seconds, or inject an intervening read failure, then disconnect again. The previous failures still count toward the limit. | |
| BLOCKED | Backend fallback | On hardware where MSMF or DSHOW opens but cannot deliver a frame, the next backend is tried in MSMF → DSHOW → CAP_ANY order and succeeds only after its first frame is read. Record the selected backend. | Camera 0 succeeded through MSMF (`1400`); no real backend failure occurred, so DSHOW/CAP_ANY fallback was not exercised on hardware. |
| PASS | Mixed sources | Run webcam, local file, HTTP, and RTSP streams together. All four continue publishing independently through the shared scheduler; one failure does not stop the others. | Camera 0, local Sintel MP4, Mux HLS, and local MediaMTX RTSP were simultaneously `ACTIVE`; local later reached `EOF` independently. |
| PASS | Queue freshness | Under detector load, dropped-frame metrics may rise but end-to-end latency remains bounded; there is no growing backlog or delayed playback catch-up. | 30-minute table: Camera final 157.3 ms vs 187.9 ms baseline; HLS 132.4 vs 167.4; RTSP 131.6 vs 177.8. Dropped rates stabilized. |
| PASS | Stop during load | Stop/restart the camera while inference or rendering is busy. Controls and metrics remain responsive. The cached last preview may remain during restart, but an old-generation render never overwrites a newer frame afterward. | With HLS and RTSP active, Camera 0 reached `STOPPED` with preview status `Stopped`, then Restart returned it to `ACTIVE`. No stale overwrite was observed. |
| BLOCKED | Two physical cameras | If two devices exist, run Camera 0 and Camera 1 together, stop/remove them independently, and verify both are released. | Only one working camera was discovered. |

To print the backend selected by the same validated open path:

```bash
PYTHONPATH=src python -c "import cv2; from vision_track.webcams import open_webcam; opened=open_webcam(0); print(cv2.videoio_registry.getBackendName(opened.backend)); opened.capture.release()"
```

Browser `getUserMedia`, browser camera permissions, frame upload from the browser,
and WebRTC are outside this checklist and outside the application architecture.
