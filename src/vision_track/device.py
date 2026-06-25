from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    kind: str
    torch_device: str
    name: str
    backend: str


def select_device(torch_module: Any | None = None, force: str | None = None) -> DeviceInfo:
    if torch_module is None:
        import torch as torch_module

    requested = force.lower() if force else None
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {None, "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device: {force}")

    if requested == "cuda" or (requested is None and cuda_available):
        name = torch_module.cuda.get_device_name(0)
        return DeviceInfo("cuda", "0", str(name), "PyTorch CUDA")
    if requested == "mps" or (requested is None and mps_available):
        return DeviceInfo("mps", "mps", "Apple Silicon GPU", "PyTorch MPS")
    return DeviceInfo("cpu", "cpu", "CPU", "PyTorch CPU")

