from __future__ import annotations

from vision_track.device import select_device


class Availability:
    def __init__(self, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available


class Cuda(Availability):
    def get_device_name(self, index: int) -> str:
        return "Test NVIDIA GPU"


class TorchMock:
    def __init__(self, cuda: bool, mps: bool) -> None:
        self.cuda = Cuda(cuda)
        self.backends = type("Backends", (), {"mps": Availability(mps)})()


def test_cuda_has_priority() -> None:
    device = select_device(TorchMock(cuda=True, mps=True))
    assert device.kind == "cuda"
    assert device.name == "Test NVIDIA GPU"


def test_mps_is_second_choice() -> None:
    assert select_device(TorchMock(cuda=False, mps=True)).kind == "mps"


def test_cpu_is_required_fallback() -> None:
    assert select_device(TorchMock(cuda=False, mps=False)).kind == "cpu"

