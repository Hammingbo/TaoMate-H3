# Modified for public FL2VA inference: use the installed PyTorch package only.
from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any


class PPUTorchError(RuntimeError):
    pass


def _module_source(module: ModuleType) -> Path:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str) or not raw:
        raise PPUTorchError("PyTorch has no filesystem source")
    try:
        return Path(raw).resolve(strict=True)
    except OSError as exc:
        raise PPUTorchError(f"cannot resolve PyTorch source: {raw}") from exc


def load_execution_torch() -> ModuleType:
    return importlib.import_module("torch")


def execution_torch_facts(torch: ModuleType) -> dict[str, Any]:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        raise PPUTorchError("PyTorch has no CUDA module")
    available = bool(cuda.is_available())
    count = int(cuda.device_count())
    return {
        "schema": "digitalhuman_h3.runtime.ppu_cuda_torch.v1",
        "source": str(_module_source(torch)),
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": available,
        "cuda_device_count": count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


__all__ = ["PPUTorchError", "execution_torch_facts", "load_execution_torch"]
