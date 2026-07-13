# Webcam hardware smoke checklist

This checklist covers behavior that fake captures and automated tests cannot prove:
real device enumeration, driver behavior, physical disconnect/reconnect, visual
freshness, and operating-system camera ownership.

The checklist is intentionally unexecuted in the repository. Do not mark an item
passed without recording the machine, camera, backend, input sources, date, and
observed evidence.

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
Date/time:
Commit:
Windows version:
Python version:
OpenCV version:
Camera model/index:
GPU/device:
Local video:
HTTP source:
RTSP source:
```

## Checklist

Use `PASS`, `FAIL`, or `BLOCKED` and add concrete evidence for every row.

| Result | Scenario | Required observation | Evidence |
|---|---|---|---|
| NOT RUN | Manual discovery | No camera probe occurs before clicking **Refresh cameras**. Clicking it probes indices 0–9 and lists available devices. | |
| NOT RUN | Probe release | After Refresh completes, Windows Camera can immediately open every discovered device that is not running in VisionTrack. | |
| NOT RUN | Duplicate protection | After adding Camera N, **Add camera** is disabled for N and a second stream for the same index cannot be created in the session. | |
| NOT RUN | Cold start | Start Camera N with detection enabled. `PREPARING` is non-blocking, then the stream reaches `CONNECTING`/`ACTIVE`; webcam capture begins after detector readiness without local-file first-frame waiting. | |
| NOT RUN | Live freshness | Run Camera N for at least five minutes with visible motion and a wall clock in view. End-to-end latency does not trend upward and preview motion does not drift progressively behind reality. | |
| NOT RUN | Stop release | Stop Camera N, then immediately open it in Windows Camera. The other application obtains the device and VisionTrack publishes no new frames. | |
| NOT RUN | Restart | Close the other application and restart Camera N. A new capture opens and frames resume without reloading an already prepared detector. | |
| NOT RUN | Remove release | Remove an active Camera N stream, then open the device in Windows Camera. No reader or reconnect activity remains. | |
| NOT RUN | Shutdown release | Start Camera N, terminate Streamlit normally with `Ctrl+C`, and open the device in Windows Camera. The device is released. | |
| NOT RUN | Successful reconnect | While active, disconnect or disable the camera long enough to enter `RECONNECTING`, then restore it before the configured budget expires. The same stream returns to `ACTIVE`. | |
| NOT RUN | Bounded failure | Keep the device unavailable. Reconnect attempts stop after the configured limit and the stream reaches `FAILED`; it does not loop forever. | |
| NOT RUN | Stable budget reset | After one reconnect, keep the camera healthy for at least 30 consecutive frames and at least 3 seconds, then disconnect it again. A fresh reconnect budget is available. | |
| NOT RUN | Unstable budget retained | Reconnect for fewer than 30 frames or fewer than 3 seconds, or inject an intervening read failure, then disconnect again. The previous failures still count toward the limit. | |
| NOT RUN | Backend fallback | On hardware where MSMF or DSHOW opens but cannot deliver a frame, the next backend is tried in MSMF → DSHOW → CAP_ANY order and succeeds only after its first frame is read. Record the selected backend. | |
| NOT RUN | Mixed sources | Run webcam, local file, HTTP, and RTSP streams together. All four continue publishing independently through the shared scheduler; one failure does not stop the others. | |
| NOT RUN | Queue freshness | Under detector load, dropped-frame metrics may rise but end-to-end latency remains bounded; there is no growing backlog or delayed playback catch-up. | |
| NOT RUN | Stop during load | Stop/restart the camera while inference or rendering is busy. Controls and metrics remain responsive. The cached last preview may remain during restart, but an old-generation render never overwrites a newer frame afterward. | |
| NOT RUN | Two physical cameras | If two devices exist, run Camera 0 and Camera 1 together, stop/remove them independently, and verify both are released. | |

To print the backend selected by the same validated open path:

```bash
PYTHONPATH=src python -c "import cv2; from vision_track.webcams import open_webcam; opened=open_webcam(0); print(cv2.videoio_registry.getBackendName(opened.backend)); opened.capture.release()"
```

Browser `getUserMedia`, browser camera permissions, frame upload from the browser,
and WebRTC are outside this checklist and outside the application architecture.
