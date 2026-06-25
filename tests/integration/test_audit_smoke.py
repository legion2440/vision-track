from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


def test_audit_import_command() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, supervision, cv2, streamlit",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_streamlit_startup_smoke() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            f"--server.port={port}",
            "--server.fileWatcherType=none",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output = ""
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                pytest.fail(f"Streamlit exited early:\n{output}")
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex(("127.0.0.1", port)) == 0:
                    return
            time.sleep(0.2)
        pytest.fail("Streamlit did not open its port within 15 seconds")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.mark.slow
def test_real_cpu_inference_when_enabled() -> None:
    if os.environ.get("VISIONTRACK_RUN_MODEL_TESTS") != "1":
        pytest.skip("Set VISIONTRACK_RUN_MODEL_TESTS=1 to allow model download/inference")
    import numpy as np

    from vision_track.detector import UltralyticsBackend
    from vision_track.device import DeviceInfo

    backend = UltralyticsBackend(
        "yolo26n.pt",
        DeviceInfo("cpu", "cpu", "CPU", "PyTorch CPU"),
        image_size=320,
    )
    result = backend.infer(np.zeros((320, 320, 3), dtype=np.uint8))
    assert result.device == "cpu"

