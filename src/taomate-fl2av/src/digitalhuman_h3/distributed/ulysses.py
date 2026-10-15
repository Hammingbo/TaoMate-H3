from __future__ import annotations

import os

import torch

from digitalhuman_h3.inference.profiling import h3_nvtx_range

from .parallel import ParallelContext, current_parallel_context

_FUSED_LAYOUT = os.environ.get("H3_ULYSSES_FUSED_LAYOUT", "0") == "1"


def _all_to_all_swap(
    value: torch.Tensor,
    *,
    world_size: int,
    group: object,
    sequence_to_heads: bool,
) -> torch.Tensor:
    """Swap sequence and head shards with one contiguous all-to-all."""

    if sequence_to_heads:
        if value.ndim < 3 or int(value.shape[-2]) % world_size:
            raise ValueError("Ulysses input heads must be divisible by its world size")
        prefix_ndim = value.ndim - 3
        local_heads = int(value.shape[-2]) // world_size
        split_shape = (*value.shape[:-2], world_size, local_heads, value.shape[-1])
        # Put the destination/head-shard axis first for all_to_all_single.
        inputs = (
            value.contiguous()
            .view(split_shape)
            .movedim(prefix_ndim + 1, 0)
            .contiguous()
        )
        outputs = torch.empty_like(inputs)
        with h3_nvtx_range(f"collective.ulysses_all_to_all.world{world_size}"):
            torch.distributed.all_to_all_single(outputs, inputs, group=group)
        # Incoming chunks are consecutive source sequence shards.
        outputs = outputs.movedim(0, prefix_ndim).contiguous()
        result_shape = (*value.shape[:-3], value.shape[-3] * world_size, local_heads, value.shape[-1])
        result = outputs.view(result_shape)
    else:
        if value.ndim < 3 or int(value.shape[-3]) % world_size:
            raise ValueError("Ulysses global rows must be divisible by its world size")
        local_rows = int(value.shape[-3]) // world_size
        split_shape = (*value.shape[:-3], world_size, local_rows, *value.shape[-2:])
        inputs = value.contiguous().view(split_shape).movedim(-4, 0).contiguous()
        outputs = torch.empty_like(inputs)
        with h3_nvtx_range(f"collective.ulysses_all_to_all.world{world_size}"):
            torch.distributed.all_to_all_single(outputs, inputs, group=group)
        # Incoming chunks are different head shards for this local sequence.
        outputs = outputs.movedim(0, -3).contiguous()
        result_shape = (*value.shape[:-3], local_rows, value.shape[-2] * world_size, value.shape[-1])
        result = outputs.view(result_shape)
    return result


class _UlyssesAllToAll(torch.autograd.Function):
    """Autograd-safe sequence/head exchange matching the validated LTX path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: object,
        value: torch.Tensor,
        world_size: int,
        group: object,
        sequence_to_heads: bool,
    ) -> torch.Tensor:
        ctx.world_size = world_size
        ctx.group = group
        ctx.sequence_to_heads = sequence_to_heads
        # heads->sequence is the final SP collective in a forward block.
        return _all_to_all_swap(
            value,
            world_size=world_size,
            group=group,
            sequence_to_heads=sequence_to_heads,
        )

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[object, ...]:
        # The reverse of sequence->heads is the final SP collective in a
        # backward block. Fence there before FSDP can enter the next block.
        grad_input = _all_to_all_swap(
            grad_output.contiguous(),
            world_size=ctx.world_size,
            group=ctx.group,
            sequence_to_heads=not ctx.sequence_to_heads,
        )
        return grad_input, None, None, None


def _swap(
    value: torch.Tensor,
    *,
    context: ParallelContext,
    sequence_to_heads: bool,
) -> torch.Tensor:
    world_size = context.ulysses_world_size
    if world_size == 1:
        return value
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Ulysses all-to-all requires initialized torch.distributed")
    if context.ulysses_group is None:
        raise RuntimeError("Ulysses process group is absent")
    if sequence_to_heads:
        padding = (-int(value.shape[-2])) % world_size
        if padding:
            value = torch.nn.functional.pad(value, (0, 0, 0, padding))
    return _UlyssesAllToAll.apply(
        value,
        world_size,
        context.ulysses_group,
        sequence_to_heads,
    )


def sequence_to_heads(
    value: torch.Tensor,
    *,
    context: ParallelContext | None = None,
) -> torch.Tensor:
    """Trade Ulysses-local sequence rows for attention heads."""

    context = context or current_parallel_context()
    return _swap(value, context=context, sequence_to_heads=True)


def heads_to_sequence(
    value: torch.Tensor,
    *,
    context: ParallelContext | None = None,
    num_heads: int | None = None,
) -> torch.Tensor:
    """Inverse of :func:`sequence_to_heads`."""

    context = context or current_parallel_context()
    if (
        _FUSED_LAYOUT and value.is_cuda and not value.requires_grad
        and value.ndim == 3 and context.ulysses_world_size > 1
    ):
        from .ulysses_layout import attention_receive_layout

        world = context.ulysses_world_size
        inputs = value.contiguous().view(
            world, value.shape[0] // world, value.shape[1], value.shape[2]
        )
        outputs = torch.empty_like(inputs)
        with h3_nvtx_range(f"collective.ulysses_all_to_all.world{world}"):
            torch.distributed.all_to_all_single(outputs, inputs, group=context.ulysses_group)
        return attention_receive_layout(
            outputs, num_heads=num_heads if num_heads is not None else int(value.shape[1]) * world
        )
    output = _swap(value, context=context, sequence_to_heads=False)
    return output if num_heads is None else output[..., :num_heads, :]


def packed_qkv_sequence_to_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    context: ParallelContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K/V into one Ulysses collective, then restore three views."""

    context = context or current_parallel_context()
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Ulysses packed Q/K/V shapes differ")
    packed = torch.stack((q, k, v), dim=0)
    exchanged = _swap(packed, context=context, sequence_to_heads=True)
    return tuple(exchanged.unbind(dim=0))  # type: ignore[return-value]


def _merged_qkv_send_layout(
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    world_size: int,
) -> torch.Tensor:
    tokens = int(value.shape[0])
    local_heads = num_heads // world_size
    return (
        value.view(tokens, 3, num_heads, head_dim)
        .view(tokens, 3, world_size, local_heads, head_dim)
        .permute(2, 1, 0, 3, 4)
        .contiguous()
    )


def merged_qkv_sequence_to_heads(
    value: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    context: ParallelContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange an inference QKV projection without first stacking Q/K/V.

    ``value`` keeps the merged-linear layout ``[tokens, 3 * heads * head_dim]``.
    The regular path materializes ``torch.stack((q, k, v))`` and then copies it
    again into the all-to-all send layout.  This inference-only path writes that
    send layout directly, while preserving the exact collective and output
    layout used by :func:`packed_qkv_sequence_to_heads`.
    """

    context = context or current_parallel_context()
    world_size = context.ulysses_world_size
    tokens = int(value.shape[0])
    if int(value.shape[-1]) != 3 * num_heads * head_dim:
        raise ValueError("merged QKV width differs from heads * head_dim")
    packed = value.view(tokens, 3, num_heads, head_dim)
    if world_size == 1:
        return tuple(packed.unbind(dim=1))  # type: ignore[return-value]
    use_fused_layout = _FUSED_LAYOUT and value.is_cuda and not value.requires_grad
    padding = (-num_heads) % world_size
    if padding and not use_fused_layout:
        packed = torch.nn.functional.pad(packed, (0, 0, 0, padding))
        num_heads += padding
        value = packed.reshape(tokens, 3 * num_heads * head_dim)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Ulysses all-to-all requires initialized torch.distributed")
    if context.ulysses_group is None:
        raise RuntimeError("Ulysses process group is absent")

    local_heads = (num_heads + world_size - 1) // world_size
    if use_fused_layout:
        from .ulysses_layout import qkv_send_layout

        inputs = qkv_send_layout(
            value, num_heads=num_heads, head_dim=head_dim, world_size=world_size
        )
    else:
        inputs = _merged_qkv_send_layout(
            value, num_heads=num_heads, head_dim=head_dim, world_size=world_size
        )
    outputs = torch.empty_like(inputs)
    with h3_nvtx_range(f"collective.ulysses_all_to_all.world{world_size}"):
        torch.distributed.all_to_all_single(
            outputs,
            inputs,
            group=context.ulysses_group,
        )
    exchanged = (
        outputs.movedim(0, 1)
        .contiguous()
        .view(3, tokens * world_size, local_heads, head_dim)
    )
    return tuple(exchanged.unbind(dim=0))  # type: ignore[return-value]


__all__ = [
    "heads_to_sequence",
    "merged_qkv_sequence_to_heads",
    "packed_qkv_sequence_to_heads",
    "sequence_to_heads",
]
