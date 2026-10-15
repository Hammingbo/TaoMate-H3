# SPDX-License-Identifier: Apache-2.0
"""Rectangular MiniMax-H3 adapter for NVLabs SOL-Attn.

The released SOL-Attn public API is square-only. HiAR instead evaluates current
media queries against condition, current media, and persistent history K/V.
This module adapts the official Triton reference algorithm to distinct query
and K/V lengths. The caller packs all quality-critical K/V into one exact
sink; the HiAR integration routes only persistent historical video blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Sequence

from .profiling import h3_nvtx_range


BLOCK = 64
GROUP = 32


@dataclass(frozen=True)
class SolStaticKVPart:
    """One immutable K/V part and its pair-owned summary cache."""

    start_token: int
    key: Any
    value: Any
    summary_cache: MutableMapping[tuple[int, int], Any]


@dataclass(frozen=True)
class _StaticBlockPlan:
    part_index: int
    global_block_start: int
    block_count: int
    local_token_start: int


@dataclass(frozen=True)
class SolAttentionCallReceipt:
    static_cache_hits: int = 0
    static_cache_misses: int = 0
    static_summary_reused_blocks: int = 0
    static_summary_built_blocks: int = 0
    dynamic_or_boundary_summary_blocks: int = 0


@dataclass(frozen=True)
class _CachedBlockSummary:
    key_centroids: Any
    value_sums: Any
    key_first_moment: Any
    key_second_moment: Any


def _plan_static_summary_blocks(
    *,
    key_tokens: int,
    static_ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[_StaticBlockPlan, ...], tuple[int, ...]]:
    """Plan exact summary reuse without changing concatenated 64-row blocks.

    A block is reusable only when all 64 of its rows belong to one immutable
    part. Blocks crossing either edge of a part remain in the recompute set.
    """

    key_blocks = (int(key_tokens) + BLOCK - 1) // BLOCK
    covered: set[int] = set()
    plans: list[_StaticBlockPlan] = []
    for part_index, (start_token, token_count) in enumerate(static_ranges):
        start_token = int(start_token)
        token_count = int(token_count)
        stop_token = start_token + token_count
        first_block = (start_token + BLOCK - 1) // BLOCK
        stop_block = stop_token // BLOCK
        block_count = max(0, stop_block - first_block)
        if block_count == 0:
            continue
        plans.append(
            _StaticBlockPlan(
                part_index=part_index,
                global_block_start=first_block,
                block_count=block_count,
                local_token_start=first_block * BLOCK - start_token,
            )
        )
        covered.update(range(first_block, stop_block))
    uncached = tuple(block for block in range(key_blocks) if block not in covered)
    return tuple(plans), uncached


def _triton_kernels() -> tuple[Any, Any, Any, Any]:
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_kv(
        k,
        v,
        kc,
        vc,
        tokens,
        padded_blocks,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block, batch_head = tl.program_id(0), tl.program_id(1)
        batch, head = batch_head // H, batch_head % H
        rows = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        dims = tl.arange(0, D)
        valid = rows < tokens
        offsets = (
            ((batch * tokens + rows[:, None]).to(tl.int64) * H + head) * D
            + dims[None, :]
        )
        key = tl.load(k + offsets, mask=valid[:, None], other=0.0)
        value = tl.load(v + offsets, mask=valid[:, None], other=0.0)
        length = tl.minimum(BLOCK_SIZE, tokens - block * BLOCK_SIZE).to(tl.float32)
        summary = ((batch * padded_blocks + block) * H + head) * D + dims
        tl.store(kc + summary, tl.sum(key, axis=0) / length)
        tl.store(vc + summary, tl.sum(value, axis=0))

    @triton.jit
    def reduce_selected_kv(
        k,
        v,
        kc,
        vc,
        block_indices,
        tokens,
        padded_blocks,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        selected_block, batch_head = tl.program_id(0), tl.program_id(1)
        block = tl.load(block_indices + selected_block)
        batch, head = batch_head // H, batch_head % H
        rows = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        dims = tl.arange(0, D)
        valid = rows < tokens
        offsets = (
            ((batch * tokens + rows[:, None]).to(tl.int64) * H + head) * D
            + dims[None, :]
        )
        key = tl.load(k + offsets, mask=valid[:, None], other=0.0)
        value = tl.load(v + offsets, mask=valid[:, None], other=0.0)
        length = tl.minimum(BLOCK_SIZE, tokens - block * BLOCK_SIZE).to(tl.float32)
        summary = ((batch * padded_blocks + block) * H + head) * D + dims
        tl.store(kc + summary, tl.sum(key, axis=0) / length)
        tl.store(vc + summary, tl.sum(value, axis=0))

    @triton.jit
    def pool_query(
        q,
        qbar,
        tokens,
        blocks,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block, batch_head = tl.program_id(0), tl.program_id(1)
        batch, head = batch_head // H, batch_head % H
        rows = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        dims = tl.arange(0, D)
        valid = rows < tokens
        offsets = (
            ((batch * tokens + rows[:, None]).to(tl.int64) * H + head) * D
            + dims[None, :]
        )
        values = tl.load(q + offsets, mask=valid[:, None], other=0.0)
        length = tl.minimum(BLOCK_SIZE, tokens - block * BLOCK_SIZE).to(tl.float32)
        output = ((batch * blocks + block) * H + head) * D + dims
        tl.store(qbar + output, tl.sum(values.to(tl.float32), axis=0) / length)

    @triton.jit
    def rectangular_forward(
        q,
        k,
        v,
        kc,
        vc,
        threshold,
        output,
        scale,
        query_tokens,
        key_tokens,
        padded_key_blocks,
        sink_start_block,
        sink_end_block,
        telemetry,
        H: tl.constexpr,
        D: tl.constexpr,
        QUERY_BLOCKS: tl.constexpr,
        KEY_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        COLLECT_TELEMETRY: tl.constexpr,
    ):
        query_block, batch_head = tl.program_id(0), tl.program_id(1)
        batch, head = batch_head // H, batch_head % H
        token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
        group_offsets = tl.max_contiguous(tl.arange(0, GROUP_SIZE), GROUP_SIZE)
        dims = tl.arange(0, D)
        query_rows = query_block * BLOCK_SIZE + token_offsets
        query_valid = query_rows < query_tokens
        query_offsets = (
            ((batch * query_tokens + query_rows[:, None]).to(tl.int64) * H + head)
            * D
            + dims[None, :]
        )
        query = tl.load(q + query_offsets, mask=query_valid[:, None], other=0.0)
        query_length = tl.minimum(
            BLOCK_SIZE, query_tokens - query_block * BLOCK_SIZE
        ).to(tl.float32)

        accumulator = tl.zeros((BLOCK_SIZE, D), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        row_max = tl.full((BLOCK_SIZE,), -float("inf"), dtype=tl.float32)
        scale_log2 = scale * 1.4426950408889634
        route_threshold = tl.load(
            threshold + (batch * QUERY_BLOCKS + query_block) * H + head
        )
        route_selected_total = tl.zeros((1,), dtype=tl.int64)
        exact_total = tl.zeros((1,), dtype=tl.int64)
        routable_total = tl.zeros((1,), dtype=tl.int64)
        candidate_total = tl.zeros((1,), dtype=tl.int64)

        for group_start in range(0, KEY_BLOCKS, GROUP_SIZE):
            block_indices = group_start + group_offsets
            valid_blocks = block_indices < KEY_BLOCKS
            summary_offsets = (
                ((batch * padded_key_blocks + block_indices[:, None]) * H + head)
                * D
                + dims[None, :]
            )
            key_centroids = tl.load(kc + summary_offsets)
            value_sums = tl.load(vc + summary_offsets)
            proxy_scores = (
                tl.dot(query, key_centroids.T).to(tl.float32) * scale_log2
            )
            route_selected = (
                tl.sum(proxy_scores, axis=0) / query_length > route_threshold
            )
            sink = (
                (block_indices >= sink_start_block)
                & (block_indices < sink_end_block)
            )
            sink = sink & valid_blocks
            routable = valid_blocks & ~sink
            route_selected = route_selected & routable
            exact = route_selected | sink
            if COLLECT_TELEMETRY:
                route_selected_total += tl.sum(
                    route_selected.to(tl.int32), axis=0
                ).to(tl.int64)
                exact_total += tl.sum(exact.to(tl.int32), axis=0).to(tl.int64)
                routable_total += tl.sum(
                    routable.to(tl.int32), axis=0
                ).to(tl.int64)
                candidate_total += tl.sum(
                    valid_blocks.to(tl.int32), axis=0
                ).to(tl.int64)

            approximate = valid_blocks & ~exact
            has_approximate = tl.sum(approximate.to(tl.int32), axis=0) > 0
            approximate_scores = tl.where(
                approximate[None, :], proxy_scores, -float("inf")
            )
            safe_scores = tl.where(has_approximate, approximate_scores, 0.0)
            candidate_max = tl.maximum(row_max, tl.max(safe_scores, axis=1))
            next_max = tl.where(has_approximate, candidate_max, row_max)
            alpha = tl.math.exp2(
                tl.where(has_approximate, row_max - next_max, 0.0)
            )
            approximate_probability = tl.math.exp2(
                safe_scores
                - tl.where(has_approximate, next_max, 0.0)[:, None]
            )
            approximate_probability = tl.where(
                has_approximate & approximate[None, :],
                approximate_probability,
                0.0,
            )
            accumulator = accumulator * alpha[:, None] + tl.dot(
                approximate_probability.to(value_sums.dtype), value_sums
            )
            lengths = tl.minimum(
                BLOCK_SIZE,
                tl.maximum(0, key_tokens - block_indices * BLOCK_SIZE),
            ).to(tl.float32)
            row_sum = row_sum * alpha + tl.sum(
                approximate_probability * lengths[None, :], axis=1
            )
            row_max = next_max

            exact_offsets = tl.where(exact, group_offsets, GROUP_SIZE)
            exact_count = tl.sum(exact.to(tl.int32), axis=0)
            for _ in range(exact_count):
                offset = tl.min(exact_offsets)
                key_block = group_start + offset
                exact_offsets = tl.where(
                    group_offsets == offset, GROUP_SIZE, exact_offsets
                )
                key_rows = key_block * BLOCK_SIZE + token_offsets
                key_valid = key_rows < key_tokens
                key_offsets = (
                    ((batch * key_tokens + key_rows[:, None]).to(tl.int64) * H + head)
                    * D
                    + dims[None, :]
                )
                exact_key = tl.load(
                    k + key_offsets, mask=key_valid[:, None], other=0.0
                )
                exact_scores = (
                    tl.dot(query, exact_key.T).to(tl.float32) * scale_log2
                )
                exact_scores += tl.where(
                    key_valid[None, :], 0.0, -float("inf")
                )
                next_max = tl.maximum(row_max, tl.max(exact_scores, axis=1))
                alpha = tl.math.exp2(row_max - next_max)
                exact_probability = tl.math.exp2(
                    exact_scores - next_max[:, None]
                )
                row_sum = row_sum * alpha + tl.sum(
                    exact_probability, axis=1
                )
                exact_value = tl.load(
                    v + key_offsets, mask=key_valid[:, None], other=0.0
                )
                accumulator = accumulator * alpha[:, None] + tl.dot(
                    exact_probability.to(exact_value.dtype), exact_value
                )
                row_max = next_max

        if COLLECT_TELEMETRY:
            tl.atomic_add(telemetry, route_selected_total)
            tl.atomic_add(telemetry + 1, exact_total)
            tl.atomic_add(telemetry + 2, routable_total)
            tl.atomic_add(telemetry + 3, candidate_total)

        tl.store(
            output + query_offsets,
            (accumulator / row_sum[:, None]).to(tl.bfloat16),
            mask=query_valid[:, None],
        )

    return reduce_kv, reduce_selected_kv, pool_query, rectangular_forward


_KERNELS: tuple[Any, Any, Any, Any] | None = None


def _kernels() -> tuple[Any, Any, Any, Any]:
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = _triton_kernels()
    return _KERNELS


def sol_attn_rectangular(
    query: Any,
    key: Any,
    value: Any,
    *,
    scale: float,
    tau: float,
    sink_tokens: int,
    static_parts: Sequence[SolStaticKVPart] = (),
    telemetry_counts: Any | None = None,
    return_receipt: bool = False,
) -> Any:
    """Run rectangular SOL-Attn over complete HiAR-visible K/V."""

    import torch
    import triton

    query = query.unsqueeze(0).contiguous()
    key = key.unsqueeze(0).contiguous()
    value = value.unsqueeze(0).contiguous()
    batch, query_tokens, heads, head_dim = query.shape
    key_tokens = int(key.shape[1])
    query_blocks = triton.cdiv(query_tokens, BLOCK)
    key_blocks = triton.cdiv(key_tokens, BLOCK)
    padded_key_blocks = triton.cdiv(key_blocks, GROUP) * GROUP
    reduce_kv, reduce_selected_kv, pool_query, rectangular_forward = _kernels()

    key_centroids = torch.zeros(
        (batch, padded_key_blocks, heads, head_dim),
        device=key.device,
        dtype=torch.bfloat16,
    )
    value_sums = torch.zeros_like(key_centroids)
    if static_parts:
        static_plans, uncached_blocks = _plan_static_summary_blocks(
            key_tokens=key_tokens,
            static_ranges=tuple(
                (part.start_token, int(part.key.shape[0]))
                for part in static_parts
            ),
        )
    else:
        static_plans = ()
        uncached_blocks = tuple(range(key_blocks)) if return_receipt else ()
    cache_hits = 0
    cache_misses = 0
    reused_blocks = 0
    built_blocks = 0
    cached_summaries: list[_CachedBlockSummary] = []
    with h3_nvtx_range("sol_attn.reduce_kv"):
        if not static_plans:
            reduce_kv[(key_blocks, batch * heads)](
                key,
                value,
                key_centroids,
                value_sums,
                key_tokens,
                padded_key_blocks,
                H=heads,
                D=head_dim,
                BLOCK_SIZE=BLOCK,
                num_warps=4,
                num_stages=1,
            )
        else:
            if uncached_blocks:
                uncached_indices = torch.tensor(
                    uncached_blocks,
                    dtype=torch.long,
                    device=key.device,
                )
                reduce_selected_kv[(len(uncached_blocks), batch * heads)](
                    key,
                    value,
                    key_centroids,
                    value_sums,
                    uncached_indices,
                    key_tokens,
                    padded_key_blocks,
                    H=heads,
                    D=head_dim,
                    BLOCK_SIZE=BLOCK,
                    num_warps=4,
                    num_stages=1,
                )
            for plan in static_plans:
                part = static_parts[plan.part_index]
                cache_key = (plan.local_token_start, plan.block_count)
                cached = part.summary_cache.get(cache_key)
                if cached is None:
                    local_key = part.key.narrow(
                        0,
                        plan.local_token_start,
                        plan.block_count * BLOCK,
                    ).unsqueeze(0).contiguous()
                    local_value = part.value.narrow(
                        0,
                        plan.local_token_start,
                        plan.block_count * BLOCK,
                    ).unsqueeze(0).contiguous()
                    local_key_centroids = torch.empty(
                        (batch, plan.block_count, heads, head_dim),
                        device=key.device,
                        dtype=torch.bfloat16,
                    )
                    local_value_sums = torch.empty_like(local_key_centroids)
                    reduce_kv[(plan.block_count, batch * heads)](
                        local_key,
                        local_value,
                        local_key_centroids,
                        local_value_sums,
                        plan.block_count * BLOCK,
                        plan.block_count,
                        H=heads,
                        D=head_dim,
                        BLOCK_SIZE=BLOCK,
                        num_warps=4,
                        num_stages=1,
                    )
                    local_centroids_fp32 = local_key_centroids.float()
                    cached = _CachedBlockSummary(
                        key_centroids=local_key_centroids,
                        value_sums=local_value_sums,
                        key_first_moment=local_centroids_fp32.sum(dim=1),
                        key_second_moment=local_centroids_fp32.square().sum(dim=1),
                    )
                    part.summary_cache[cache_key] = cached
                    cache_misses += 1
                    built_blocks += plan.block_count
                else:
                    cache_hits += 1
                    reused_blocks += plan.block_count
                start = plan.global_block_start
                stop = start + plan.block_count
                key_centroids[:, start:stop].copy_(cached.key_centroids)
                value_sums[:, start:stop].copy_(cached.value_sums)
                cached_summaries.append(cached)

    query_centroids = torch.empty(
        (batch, query_blocks, heads, head_dim),
        device=query.device,
        dtype=torch.bfloat16,
    )
    with h3_nvtx_range("sol_attn.route_threshold"):
        pool_query[(query_blocks, batch * heads)](
            query,
            query_centroids,
            query_tokens,
            query_blocks,
            H=heads,
            D=head_dim,
            BLOCK_SIZE=BLOCK,
            num_warps=4,
            num_stages=1,
        )
        if static_plans:
            first_moment = sum(
                (cached.key_first_moment for cached in cached_summaries),
                torch.zeros(
                    (batch, heads, head_dim),
                    device=key.device,
                    dtype=torch.float32,
                ),
            )
            second_moment = sum(
                (cached.key_second_moment for cached in cached_summaries),
                torch.zeros_like(first_moment),
            )
            if uncached_blocks:
                dynamic_centroids = key_centroids.index_select(
                    1, uncached_indices
                ).float()
                first_moment = first_moment + dynamic_centroids.sum(dim=1)
                second_moment = second_moment + dynamic_centroids.square().sum(
                    dim=1
                )
            key_mean = first_moment / key_blocks
            key_variance = second_moment / key_blocks - key_mean.square()
        else:
            valid_centroids = key_centroids[:, :key_blocks].float()
            key_mean = valid_centroids.mean(dim=1)
            key_variance = valid_centroids.var(dim=1, unbiased=False)
        query_centroids_fp32 = query_centroids.float()
        log2_scale = float(scale) * 1.4426950408889634
        proxy_mean = (
            query_centroids_fp32 * key_mean[:, None]
        ).sum(dim=-1) * log2_scale
        proxy_variance = (
            query_centroids_fp32.square() * key_variance[:, None]
        ).sum(dim=-1) * (log2_scale * log2_scale)
        threshold = proxy_mean + float(tau) * torch.sqrt(
            proxy_variance.clamp_min(0.0) + 1.0e-6
        )

    output = torch.empty_like(query)
    sink_start_block = 0
    sink_end_block = triton.cdiv(int(sink_tokens), BLOCK)
    with h3_nvtx_range("sol_attn.rectangular_forward"):
        rectangular_forward[(query_blocks, batch * heads)](
            query,
            key,
            value,
            key_centroids,
            value_sums,
            threshold,
            output,
            float(scale),
            query_tokens,
            key_tokens,
            padded_key_blocks,
            sink_start_block,
            sink_end_block,
            telemetry_counts if telemetry_counts is not None else key_centroids,
            H=heads,
            D=head_dim,
            QUERY_BLOCKS=query_blocks,
            KEY_BLOCKS=key_blocks,
            BLOCK_SIZE=BLOCK,
            GROUP_SIZE=GROUP,
            COLLECT_TELEMETRY=telemetry_counts is not None,
            num_warps=4,
            num_stages=1,
        )
    output = output.squeeze(0)
    if not return_receipt:
        return output
    return output, SolAttentionCallReceipt(
        static_cache_hits=cache_hits,
        static_cache_misses=cache_misses,
        static_summary_reused_blocks=reused_blocks,
        static_summary_built_blocks=built_blocks,
        dynamic_or_boundary_summary_blocks=len(uncached_blocks),
    )


__all__ = [
    "SolAttentionCallReceipt",
    "SolStaticKVPart",
    "sol_attn_rectangular",
]
