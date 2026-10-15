from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator


_NVTX_ENABLED = os.environ.get("H3_PROFILE_NVTX", "0") == "1"


@contextmanager
def h3_nvtx_range(name: str) -> Iterator[None]:
    """Emit a semantic range only for explicit Nsight profiling runs."""

    if not _NVTX_ENABLED:
        yield
        return
    import torch

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def h3_nvtx_enabled() -> bool:
    return _NVTX_ENABLED


__all__ = ["h3_nvtx_enabled", "h3_nvtx_range"]
