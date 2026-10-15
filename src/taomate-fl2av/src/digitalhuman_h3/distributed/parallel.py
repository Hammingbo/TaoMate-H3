from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ParallelContext:
    """Explicit rank topology for native TP/Ulysses execution.

    Process groups are deliberately not inferred here. The runtime owns group
    construction and installs the rank view for the current worker.
    """

    tp_world_size: int = 1
    tp_rank: int = 0
    ulysses_world_size: int = 1
    ulysses_rank: int = 0
    ring_world_size: int = 1
    ring_rank: int = 0
    tp_group: Any | None = None
    ulysses_group: Any | None = None

    def __post_init__(self) -> None:
        for name in ("tp", "ulysses", "ring"):
            size = getattr(self, f"{name}_world_size")
            rank = getattr(self, f"{name}_rank")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(f"invalid {name} topology: world_size={size}, rank={rank}")
        if self.tp_world_size > 1 and self.tp_group is None:
            raise ValueError("tp_group is required when TP world size exceeds one")
        if self.ulysses_world_size > 1 and self.ulysses_group is None:
            raise ValueError(
                "ulysses_group is required when Ulysses world size exceeds one"
            )


_CURRENT = ContextVar("digitalhuman_h3_parallel_context", default=ParallelContext())


def current_parallel_context() -> ParallelContext:
    return _CURRENT.get()


@contextmanager
def use_parallel_context(context: ParallelContext) -> Iterator[ParallelContext]:
    if not isinstance(context, ParallelContext):
        raise TypeError("context must be ParallelContext")
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        _CURRENT.reset(token)


__all__ = ["ParallelContext", "current_parallel_context", "use_parallel_context"]
