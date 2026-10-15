from __future__ import annotations

from typing import Any

import torch

from digitalhuman_h3.inference.profiling import h3_nvtx_range

from .parallel import ParallelContext, current_parallel_context


def _require_distributed(world_size: int, group: Any | None, name: str) -> Any:
    if world_size == 1:
        return None
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(f"{name} collective requires initialized torch.distributed")
    if group is None:
        raise RuntimeError(f"{name} process group is absent")
    return group


def all_reduce_sum(
    value: torch.Tensor,
    *,
    group: Any,
    world_size: int,
) -> torch.Tensor:
    _require_distributed(world_size, group, "all-reduce")
    if world_size == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        from torch.distributed.nn.functional import all_reduce

        return all_reduce(value, group=group)
    with h3_nvtx_range(f"collective.all_reduce.world{world_size}"):
        output = value.clone()
        torch.distributed.all_reduce(output, group=group)
    return output


def all_reduce_sum_inplace_(
    value: torch.Tensor,
    *,
    group: Any,
    world_size: int,
) -> torch.Tensor:
    """Reduce an inference-owned temporary without cloning it first."""

    _require_distributed(world_size, group, "all-reduce")
    if world_size == 1:
        return value
    with h3_nvtx_range(f"collective.all_reduce_inplace.world{world_size}"):
        torch.distributed.all_reduce(value, group=group)
    return value


def all_gather_concat(
    value: torch.Tensor,
    *,
    dim: int,
    group: Any,
    world_size: int,
) -> torch.Tensor:
    _require_distributed(world_size, group, "all-gather")
    if world_size == 1:
        return value
    if torch.is_grad_enabled() and value.requires_grad:
        from torch.distributed.nn.functional import all_gather

        gathered = all_gather(value, group=group)
    else:
        with h3_nvtx_range(f"collective.all_gather.world{world_size}"):
            gathered = [torch.empty_like(value) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, value.contiguous(), group=group)
    return torch.cat(tuple(gathered), dim=dim)


def tp_all_reduce(value: torch.Tensor, context: ParallelContext | None = None) -> torch.Tensor:
    context = context or current_parallel_context()
    return all_reduce_sum(
        value,
        group=context.tp_group,
        world_size=context.tp_world_size,
    )


def tp_all_reduce_inplace_(
    value: torch.Tensor,
    context: ParallelContext | None = None,
) -> torch.Tensor:
    context = context or current_parallel_context()
    return all_reduce_sum_inplace_(
        value,
        group=context.tp_group,
        world_size=context.tp_world_size,
    )


def tp_all_gather(
    value: torch.Tensor,
    *,
    dim: int = -1,
    context: ParallelContext | None = None,
) -> torch.Tensor:
    context = context or current_parallel_context()
    return all_gather_concat(
        value,
        dim=dim,
        group=context.tp_group,
        world_size=context.tp_world_size,
    )


def ulysses_all_gather_rows(
    value: torch.Tensor,
    context: ParallelContext | None = None,
) -> torch.Tensor:
    context = context or current_parallel_context()
    return all_gather_concat(
        value,
        dim=0,
        group=context.ulysses_group,
        world_size=context.ulysses_world_size,
    )


__all__ = [
    "all_gather_concat",
    "all_reduce_sum",
    "all_reduce_sum_inplace_",
    "tp_all_gather",
    "tp_all_reduce",
    "tp_all_reduce_inplace_",
    "ulysses_all_gather_rows",
]
