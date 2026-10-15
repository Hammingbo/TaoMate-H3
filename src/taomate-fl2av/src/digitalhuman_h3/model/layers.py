# Modified for the public FL2VA inference-only release: unused server/training routes removed.
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch MiniMax H3 DiT layers."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from digitalhuman_h3.distributed import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ParallelContext,
    RowParallelLinear,
    current_parallel_context,
    heads_to_sequence,
    merged_qkv_sequence_to_heads,
    packed_qkv_sequence_to_heads,
    tp_all_gather,
)
from digitalhuman_h3.inference.profiling import h3_nvtx_range

from .architecture import MINIMAX_H3_ADALN_MODALITY_NUM, MiniMaxH3Architecture


BF16 = torch.bfloat16
FP32 = torch.float32

_NATIVE_STREAM_HOOK: Any | None = None
_NATIVE_STREAM_METADATA: ContextVar[tuple[Any, Any] | None] = ContextVar(
    "digitalhuman_h3_native_stream_metadata", default=None
)
_NATIVE_STREAM_LIVE_DOCUMENTS_HINT: ContextVar[tuple[int, ...] | None] = (
    ContextVar("digitalhuman_h3_native_stream_live_documents_hint", default=None)
)
_NATIVE_STREAM_MLP_VIDEO_STRIDE: ContextVar[int | None] = ContextVar(
    "digitalhuman_h3_native_stream_mlp_video_stride", default=None
)
_NATIVE_STREAM_METADATA_FALLBACK: tuple[Any, Any] | None = None
_NATIVE_STREAM_MLP_VIDEO_STRIDE_FALLBACK: int | None = None


def active_native_stream_attention_metadata() -> tuple[Any, Any] | None:
    """Return metadata visible to ordinary and checkpoint-recompute forwards."""

    # Non-reentrant checkpoint recomputation may run in an autograd worker
    # context that does not inherit Python ContextVar state.  Streaming
    # metadata scopes forbid nesting, so their bounded process fallback keeps
    # recomputation on the same attention route as the original forward.
    metadata = _NATIVE_STREAM_METADATA.get()
    if metadata is not None:
        return metadata
    return _NATIVE_STREAM_METADATA_FALLBACK


def active_native_stream_mlp_video_stride() -> int:
    """Return the MLP stride bound to the active streaming metadata."""

    stride = _NATIVE_STREAM_MLP_VIDEO_STRIDE.get()
    if stride is not None:
        return stride
    fallback = _NATIVE_STREAM_MLP_VIDEO_STRIDE_FALLBACK
    return 1 if fallback is None else fallback
_INFERENCE_PACKED_DENSE_FLASH_ATTN3 = None


def set_inference_packed_dense_flash_attn3(enabled: bool) -> None:
    """Select dense FA3 or SDPA for packed attention outside stream hooks."""

    global _INFERENCE_PACKED_DENSE_FLASH_ATTN3
    if not enabled:
        _INFERENCE_PACKED_DENSE_FLASH_ATTN3 = None
        return
    from flash_attn_interface import flash_attn_func

    _INFERENCE_PACKED_DENSE_FLASH_ATTN3 = flash_attn_func


def set_native_stream_attention_hook(hook: Any | None) -> None:
    global _NATIVE_STREAM_HOOK
    if hook is not None and not callable(hook):
        raise TypeError("native H3 streaming hook must be callable")
    if hook is not None and _NATIVE_STREAM_HOOK is not None:
        raise RuntimeError("a native H3 streaming hook is already installed")
    _NATIVE_STREAM_HOOK = hook


@contextmanager
def native_stream_attention_metadata(
    token_tags: torch.Tensor,
    commit_mask: torch.Tensor,
    *,
    live_documents_hint: tuple[int, ...] | None = None,
    mlp_video_stride: int = 1,
) -> Iterator[None]:
    global _NATIVE_STREAM_METADATA_FALLBACK
    global _NATIVE_STREAM_MLP_VIDEO_STRIDE_FALLBACK

    if mlp_video_stride < 1:
        raise ValueError("native stream MLP video stride must be positive")
    if active_native_stream_attention_metadata() is not None:
        raise RuntimeError("native H3 streaming attention metadata is nested")
    metadata = (token_tags, commit_mask)
    stride = int(mlp_video_stride)
    token = _NATIVE_STREAM_METADATA.set(metadata)
    live_documents_token = _NATIVE_STREAM_LIVE_DOCUMENTS_HINT.set(
        live_documents_hint
    )
    stride_token = _NATIVE_STREAM_MLP_VIDEO_STRIDE.set(stride)
    _NATIVE_STREAM_METADATA_FALLBACK = metadata
    _NATIVE_STREAM_MLP_VIDEO_STRIDE_FALLBACK = stride
    try:
        yield
    finally:
        _NATIVE_STREAM_MLP_VIDEO_STRIDE.reset(stride_token)
        _NATIVE_STREAM_LIVE_DOCUMENTS_HINT.reset(live_documents_token)
        _NATIVE_STREAM_METADATA.reset(token)
        _NATIVE_STREAM_MLP_VIDEO_STRIDE_FALLBACK = None
        _NATIVE_STREAM_METADATA_FALLBACK = None


class MiniMaxH3RMSNorm(nn.RMSNorm):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        from digitalhuman_h3.inference.fused_kernels import (
            fused_rms_norm,
            h3_inference_fused_kernels_enabled,
        )

        if (
            h3_inference_fused_kernels_enabled()
            and value.is_cuda
            and value.dtype == BF16
            and self.weight.dtype == BF16
        ):
            return fused_rms_norm(value, self.weight, float(self.eps))
        return super().forward(value)


def _norm(size: int, *, eps: float, dtype: torch.dtype = BF16) -> nn.RMSNorm:
    return MiniMaxH3RMSNorm(size, eps=eps, dtype=dtype)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = torch.chunk(value, 2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def modulate_scale_shift(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    from digitalhuman_h3.inference.fused_kernels import (
        fused_modulate,
        h3_inference_fused_kernels_enabled,
    )

    if h3_inference_fused_kernels_enabled() and dtype == BF16:
        return fused_modulate(value, shift, scale, indices)
    return (
        value * (1.0 + scale.index_select(0, indices))
        + shift.index_select(0, indices)
    ).to(dtype)


def modulate_gate(
    value: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    from digitalhuman_h3.inference.fused_kernels import (
        fused_gate_add,
        h3_inference_fused_kernels_enabled,
    )

    if h3_inference_fused_kernels_enabled() and dtype == BF16:
        return fused_gate_add(value, gate, other, indices)
    return (value + gate.index_select(0, indices) * other).to(dtype)


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq",
            torch.empty(inv_freq_len, dtype=FP32),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        position = img_position_ids[0].to(FP32)
        per_axis = position.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        temporal, height, width = per_axis.unbind(dim=1)
        half = torch.cat((temporal, height, width), dim=-1)
        return torch.cat((half, half), dim=-1)


def rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    half = freqs.shape[-1] // 2
    return torch.cat(
        (torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])), dim=-1
    ).to(dtype=dtype, copy=False).contiguous()


def apply_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = cos_sin_cache.index_select(0, positions.to(torch.long))
    half = selected.shape[-1] // 2
    cos_half, sin_half = selected.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)

    from digitalhuman_h3.inference.fused_kernels import (
        fused_apply_rope,
        h3_inference_fused_kernels_enabled,
    )

    if h3_inference_fused_kernels_enabled():
        return (
            fused_apply_rope(q, cos.squeeze(1), sin.squeeze(1)),
            fused_apply_rope(k, cos.squeeze(1), sin.squeeze(1)),
        )

    def apply(value: torch.Tensor) -> torch.Tensor:
        rotating = value[..., : selected.shape[-1]]
        passthrough = value[..., selected.shape[-1] :]
        return torch.cat(
            (rotating * cos + _rotate_half(rotating) * sin, passthrough), dim=-1
        )

    return apply(q), apply(k)


def _packed_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    # The packed layout already carries its canonical host boundaries.  Do not
    # copy the identical device tensor back to CPU for every attention block:
    # that operation synchronizes all preceding CUDA/FSDP/Ulysses work and can
    # turn an otherwise asynchronous forward into a cross-process stall.
    bounds = (
        tuple(int(item) for item in cu_seqlens_host)
        if cu_seqlens_host is not None
        else tuple(int(item) for item in cu_seqlens.detach().cpu().tolist())
    )
    lengths = tuple(stop - start for start, stop in zip(bounds[:-1], bounds[1:]))
    if (
        len(bounds) < 2
        or bounds[0] != 0
        or bounds[-1] != int(q.shape[0])
        or any(length < 0 for length in lengths)
        or max_seqlen != max(lengths, default=0)
    ):
        raise ValueError("packed-attention cumulative lengths are invalid")
    segments: list[torch.Tensor] = []
    flash_attn_func = _INFERENCE_PACKED_DENSE_FLASH_ATTN3

    for start, stop in zip(bounds[:-1], bounds[1:]):
        if start == stop:
            continue
        if flash_attn_func is None:
            output = F.scaled_dot_product_attention(
                q[start:stop].transpose(0, 1).unsqueeze(0),
                k[start:stop].transpose(0, 1).unsqueeze(0),
                v[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
            )
            segments.append(output[0].transpose(0, 1))
            continue
        output = flash_attn_func(
            q[start:stop].unsqueeze(0),
            k[start:stop].unsqueeze(0),
            v[start:stop].unsqueeze(0),
            softmax_scale=scale,
            causal=False,
            num_splits=1,
        )
        if isinstance(output, tuple):
            output = output[0]
        segments.append(output.squeeze(0))
    if not segments:
        raise ValueError("packed attention has no live segment")
    return torch.cat(segments, dim=0)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3Architecture,
        *,
        context: ParallelContext,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = ColumnParallelLinear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            gather_output=False,
            dtype=FP32,
            context=context,
        )
        self.proj_out = RowParallelLinear(
            arch.time_embed_hidden_size,
            arch.time_embed_dim,
            bias=True,
            input_is_parallel=True,
            dtype=FP32,
            context=context,
        )
        self.register_buffer("_frequency_cache", None, persistent=False)
        self._inference_static_cache_enabled = False
        self._inference_static_cache: dict[tuple[float, ...], torch.Tensor] = {}
        self._inference_static_cache_hits = 0
        self._inference_static_cache_misses = 0

    def enable_inference_static_cache(self) -> None:
        if self.training:
            raise RuntimeError("timestep embedding static cache is inference-only")
        self._inference_static_cache_enabled = True
        self._inference_static_cache = {}
        self._inference_static_cache_hits = 0
        self._inference_static_cache_misses = 0

    def disable_inference_static_cache(self) -> None:
        self._inference_static_cache_enabled = False
        self._inference_static_cache = {}

    def inference_static_cache_stats(self) -> tuple[int, int]:
        return (
            self._inference_static_cache_hits,
            self._inference_static_cache_misses,
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if self._inference_static_cache_enabled:
            cache_key = tuple(
                float(value)
                for value in timestep.detach().to(torch.float32).cpu().tolist()
            )
            cached = self._inference_static_cache.get(cache_key)
            if cached is not None:
                self._inference_static_cache_hits += 1
                return cached
        half = self.frequency_embedding_size // 2
        frequencies = self._frequency_cache
        if frequencies is None or frequencies.device != timestep.device:
            frequencies = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=FP32, device=timestep.device)
                / half
            )
            self._frequency_cache = frequencies
        args = timestep.to(FP32)[:, None] * frequencies[None]
        encoded = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        output = self.proj_out(F.silu(self.proj_in(encoded)))
        if self._inference_static_cache_enabled:
            self._inference_static_cache[cache_key] = output
            self._inference_static_cache_misses += 1
        return output


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3Architecture,
        *,
        context: ParallelContext,
        load_time_w8a8_names: frozenset[str] = frozenset(),
        module_prefix: str | None = None,
    ) -> None:
        super().__init__()
        if arch.num_attention_heads % context.tp_world_size:
            raise ValueError("MiniMax H3 heads must be divisible by TP")
        self.parallel_context = context
        self.total_num_heads = arch.num_attention_heads
        self.num_heads = self.total_num_heads // context.tp_world_size
        self.head_dim = arch.attention_head_dim
        self.inner_dim = self.total_num_heads * self.head_dim
        self.local_inner_dim = self.num_heads * self.head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.layer_name: str | None = None
        qkv_name = f"{module_prefix}.qkv_proj"
        qkv_w8a8 = qkv_name in load_time_w8a8_names
        self.qkv_proj = MergedColumnParallelLinear(
            arch.hidden_size,
            [self.inner_dim] * 3,
            bias=False,
            gather_output=False,
            dtype=torch.int8 if qkv_w8a8 else BF16,
            weight_requires_grad=not qkv_w8a8,
            context=context,
        )
        if qkv_w8a8:
            from digitalhuman_h3.inference.w8a8 import (
                prepare_cutlass_w8a8_stream_load_slot_,
            )

            prepare_cutlass_w8a8_stream_load_slot_(
                torch,
                self.qkv_proj,
                name=qkv_name,
            )
        self.q_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(arch.attention_head_dim, eps=arch.qk_norm_eps)
        self.out_proj = RowParallelLinear(
            self.inner_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=BF16,
            context=context,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        ulysses_active: bool,
    ) -> torch.Tensor:
        total = value.shape[0]
        qkv = self.qkv_proj(value)
        from digitalhuman_h3.inference.fused_kernels import (
            fused_qk_norm_rope,
            fused_qkv_qk_norm_rope_,
            h3_fused_qkv_preprocess_enabled,
            h3_sol_engine_fusions_enabled,
        )

        fused_qkv_preprocess = bool(
            rope_cache is not None
            and h3_fused_qkv_preprocess_enabled()
            and not qkv.requires_grad
            and qkv.is_cuda
            and qkv.dtype == BF16
        )
        if fused_qkv_preprocess:
            cos_sin_cache, positions = rope_cache
            fused_qkv_qk_norm_rope_(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                positions,
                float(self.q_norm.eps),
            )
        q, k, v = qkv.split(self.local_inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)
        if not fused_qkv_preprocess:
            if rope_cache is not None and h3_sol_engine_fusions_enabled():
                cos_sin_cache, positions = rope_cache
                selected = cos_sin_cache.index_select(0, positions.to(torch.long))
                half = selected.shape[-1] // 2
                cos_half, sin_half = selected.split(half, dim=-1)
                cos = torch.cat((cos_half, cos_half), dim=-1)
                sin = torch.cat((sin_half, sin_half), dim=-1)
                q = fused_qk_norm_rope(
                    q, self.q_norm.weight, cos, sin, float(self.q_norm.eps)
                )
                k = fused_qk_norm_rope(
                    k, self.k_norm.weight, cos, sin, float(self.k_norm.eps)
                )
            else:
                q = self.q_norm(q)
                k = self.k_norm(k)
                if rope_cache is not None:
                    q, k = apply_rope_qk(q, k, *rope_cache)
        if ulysses_active:
            if fused_qkv_preprocess:
                q, k, v = merged_qkv_sequence_to_heads(
                    qkv,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    context=self.parallel_context,
                )
            else:
                q, k, v = packed_qkv_sequence_to_heads(
                    q, k, v, context=self.parallel_context
                )
        stream_metadata = active_native_stream_attention_metadata()
        if (
            self.layer_name is not None
            and _NATIVE_STREAM_HOOK is not None
            and stream_metadata is not None
        ):
            token_tags, commit_mask = stream_metadata
            output = _NATIVE_STREAM_HOOK(
                attention=self,
                layer_name=f"{self.layer_name}.attn",
                query=q,
                key=k,
                value=v,
                token_tags=token_tags,
                commit_mask=commit_mask,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                live_documents_hint=_NATIVE_STREAM_LIVE_DOCUMENTS_HINT.get(),
            )
        else:
            output = _packed_sdpa(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                scale=self.softmax_scale,
            )
        if ulysses_active:
            output = heads_to_sequence(
                output, context=self.parallel_context, num_heads=self.num_heads
            )
        return self.out_proj(output.reshape(total, self.local_inner_dim))


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3Architecture,
        *,
        context: ParallelContext,
        load_time_w8a8_names: frozenset[str] = frozenset(),
        module_prefix: str | None = None,
    ) -> None:
        super().__init__()
        fc1_name = f"{module_prefix}.fc1"
        fc1_w8a8 = fc1_name in load_time_w8a8_names
        self.fc1 = MergedColumnParallelLinear(
            arch.hidden_size,
            [arch.ffn_hidden_size] * 2,
            bias=False,
            gather_output=False,
            dtype=torch.int8 if fc1_w8a8 else BF16,
            weight_requires_grad=not fc1_w8a8,
            context=context,
        )
        if fc1_w8a8:
            from digitalhuman_h3.inference.w8a8 import (
                prepare_cutlass_w8a8_stream_load_slot_,
            )

            prepare_cutlass_w8a8_stream_load_slot_(
                torch,
                self.fc1,
                name=fc1_name,
            )
        self.fc2 = RowParallelLinear(
            arch.ffn_hidden_size,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=BF16,
            context=context,
        )
        self._activation_checkpoint_chunk_rows = 0

    def enable_activation_checkpoint_chunking(self, *, rows_per_chunk: int) -> None:
        if rows_per_chunk <= 0:
            raise ValueError("MLP checkpoint chunk rows must be positive")
        self._activation_checkpoint_chunk_rows = int(rows_per_chunk)

    def _forward_dense(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.fc1(value)
        from digitalhuman_h3.inference.fused_kernels import (
            fused_swiglu,
            h3_inference_fused_kernels_enabled,
        )

        if h3_inference_fused_kernels_enabled():
            return self.fc2(fused_swiglu(projected))
        gate, up = projected.chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * up)

    def _forward_sparse_video(
        self,
        value: torch.Tensor,
        *,
        token_tags: torch.Tensor,
        commit_mask: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        # The MLP is token-wise.  Preserve every non-video token exactly and
        # evaluate a regularly sampled set of video rows, then linearly restore
        # the skipped video rows to the original full sequence shape.  This is
        # inference-only structured token sparsity and requires matching
        # sparse-aware training before it can be treated as a quality route.
        from digitalhuman_h3.streaming.cache import VIDEO_TOKEN_TAG

        tags = token_tags.view(-1)
        if tags.device != value.device:
            tags = tags.to(device=value.device)
        if int(tags.numel()) != int(value.shape[0]):
            raise ValueError("stream MLP token tags do not cover the local sequence")
        target_mask = commit_mask.view(-1)
        if target_mask.device != value.device:
            target_mask = target_mask.to(device=value.device)
        if int(target_mask.numel()) != int(value.shape[0]):
            raise ValueError("stream MLP commit mask does not cover the local sequence")
        video_rows = torch.nonzero(
            target_mask & (tags == VIDEO_TOKEN_TAG),
            as_tuple=False,
        ).view(-1)
        if int(video_rows.numel()) <= stride:
            return self._forward_dense(value)

        selected_video_rows = video_rows[::stride]
        selected_mask = ~(target_mask & (tags == VIDEO_TOKEN_TAG))
        selected_mask[selected_video_rows] = True
        selected_rows = torch.nonzero(selected_mask, as_tuple=False).view(-1)
        selected_output = self._forward_dense(value.index_select(0, selected_rows))

        output = torch.empty_like(value)
        output.index_copy_(0, selected_rows, selected_output)
        selected_is_target_video = (
            target_mask.index_select(0, selected_rows)
            & (tags.index_select(0, selected_rows) == VIDEO_TOKEN_TAG)
        )
        sampled_output = selected_output.index_select(
            0,
            torch.nonzero(
                selected_is_target_video,
                as_tuple=False,
            ).view(-1),
        )
        ordinals = torch.arange(
            int(video_rows.numel()),
            device=value.device,
            dtype=torch.long,
        )
        left = torch.div(ordinals, stride, rounding_mode="floor")
        right = torch.clamp(left + 1, max=int(sampled_output.shape[0]) - 1)
        alpha = (ordinals.remainder(stride).to(torch.float32) / float(stride)).to(
            dtype=value.dtype
        )
        restored_video = torch.lerp(
            sampled_output.index_select(0, left),
            sampled_output.index_select(0, right),
            alpha.unsqueeze(-1),
        )
        output.index_copy_(0, video_rows, restored_video)
        return output

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        stream_metadata = active_native_stream_attention_metadata()
        stream_stride = active_native_stream_mlp_video_stride()
        if (
            stream_stride > 1
            and stream_metadata is not None
            and value.ndim == 2
        ):
            token_tags, commit_mask = stream_metadata
            return self._forward_sparse_video(
                value,
                token_tags=token_tags,
                commit_mask=commit_mask,
                stride=stream_stride,
            )
        rows_per_chunk = self._activation_checkpoint_chunk_rows
        if rows_per_chunk <= 0 or value.ndim != 2 or value.shape[0] <= rows_per_chunk:
            return self._forward_dense(value)
        if not torch.is_grad_enabled():
            return torch.cat(
                [
                    self._forward_dense(value[start : start + rows_per_chunk])
                    for start in range(0, int(value.shape[0]), rows_per_chunk)
                ],
                dim=0,
            )
        outputs = [
            checkpoint(
                self._forward_dense,
                value[start : start + rows_per_chunk],
                use_reentrant=False,
            )
            for start in range(0, int(value.shape[0]), rows_per_chunk)
        ]
        return torch.cat(outputs, dim=0)


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3Architecture,
        out_features: int,
        *,
        expand_ratio: int,
        modality_num: int,
        context: ParallelContext,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError("MiniMax H3 AdaLN output contract differs")
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.parallel_context = context
        self.linear = ColumnParallelLinear(
            arch.time_embed_dim,
            out_features,
            bias=True,
            gather_output=False,
            dtype=BF16,
            context=context,
        )
        self._inference_static_cache_enabled = False
        self._inference_static_cache: dict[
            int, tuple[torch.Tensor, ...]
        ] = {}
        self._inference_static_cache_hits = 0
        self._inference_static_cache_misses = 0

    def enable_inference_static_cache(self) -> None:
        if self.training:
            raise RuntimeError("AdaLN static cache is inference-only")
        self._inference_static_cache_enabled = True
        self._inference_static_cache = {}
        self._inference_static_cache_hits = 0
        self._inference_static_cache_misses = 0

    def disable_inference_static_cache(self) -> None:
        self._inference_static_cache_enabled = False
        self._inference_static_cache = {}

    def inference_static_cache_stats(self) -> tuple[int, int]:
        return (
            self._inference_static_cache_hits,
            self._inference_static_cache_misses,
        )

    def project_local(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    def split_output(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows = value.shape[0]
        value = value.view(
            rows * self.modality_num, self.expand_ratio * self.hidden_size
        )
        return tuple(value.chunk(self.expand_ratio, dim=-1))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self._inference_static_cache_enabled:
            cache_key = id(value)
            cached = self._inference_static_cache.get(cache_key)
            if cached is not None:
                self._inference_static_cache_hits += 1
                return cached
        projected = tp_all_gather(
            self.project_local(value), context=self.parallel_context
        )
        output = self.split_output(projected)
        if self._inference_static_cache_enabled:
            self._inference_static_cache[cache_key] = output
            self._inference_static_cache_misses += 1
        return output


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, arch: MiniMaxH3Architecture, *, context: ParallelContext) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, context=context)
        self.mlp = MiniMaxH3MLP(arch, context=context)

    def forward(
        self,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> torch.Tensor:
        value = value + self.attn(
            self.norm1(value),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ulysses_active=False,
        )
        return value + self.mlp(self.norm2(value))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, arch: MiniMaxH3Architecture, *, context: ParallelContext) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MiniMaxH3TokenRefinerBlock(arch, context=context)
            for _ in range(arch.token_refiner_num_layers)
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
    ) -> torch.Tensor:
        for block in self.blocks:
            value = block(
                value,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )
        return self.final_norm(value)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3Architecture,
        *,
        context: ParallelContext,
        load_time_w8a8_names: frozenset[str] = frozenset(),
        module_prefix: str | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(
            arch,
            context=context,
            load_time_w8a8_names=load_time_w8a8_names,
            module_prefix=(
                None if module_prefix is None else f"{module_prefix}.attn"
            ),
        )
        self.mlp = MiniMaxH3MLP(
            arch,
            context=context,
            load_time_w8a8_names=load_time_w8a8_names,
            module_prefix=(
                None if module_prefix is None else f"{module_prefix}.mlp"
            ),
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
            context=context,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        ulysses_active: bool,
    ) -> torch.Tensor:
        layer_label = self.attn.layer_name or "blocks.unknown"
        with h3_nvtx_range(f"dit.{layer_label}.adaln_norm"):
            parameters = self.adaln_proj(adaln_input)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = parameters
        from digitalhuman_h3.inference.fused_kernels import (
            fused_residual_gate_rms_norm_modulate,
            fused_rms_norm_modulate,
            h3_sol_engine_fusions_enabled,
        )

        sol_fusions = h3_sol_engine_fusions_enabled()
        with h3_nvtx_range(f"dit.{layer_label}.pre_attention_norm"):
            residual = value
            if sol_fusions:
                hidden = fused_rms_norm_modulate(
                    value,
                    self.norm1.weight,
                    scale_msa,
                    shift_msa,
                    combined_indices,
                    float(self.norm1.eps),
                )
            else:
                hidden = modulate_scale_shift(
                    self.norm1(value),
                    shift_msa,
                    scale_msa,
                    combined_indices,
                    dtype=BF16,
                )
        with h3_nvtx_range(f"dit.{layer_label}.attention"):
            hidden = self.attn(
                hidden,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=ulysses_active,
            )
        with h3_nvtx_range(f"dit.{layer_label}.post_attention_norm"):
            if sol_fusions:
                value, hidden = fused_residual_gate_rms_norm_modulate(
                    residual,
                    hidden,
                    gate_msa,
                    self.norm2.weight,
                    scale_mlp,
                    shift_mlp,
                    combined_indices,
                    float(self.norm2.eps),
                )
            else:
                value = modulate_gate(
                    residual, gate_msa, hidden, combined_indices, dtype=BF16
                )
                hidden = modulate_scale_shift(
                    self.norm2(value),
                    shift_mlp,
                    scale_mlp,
                    combined_indices,
                    dtype=BF16,
                )
        residual = value
        with h3_nvtx_range(f"dit.{layer_label}.mlp"):
            output = modulate_gate(
                residual,
                gate_mlp,
                self.mlp(hidden),
                combined_indices,
                dtype=BF16,
            )
        return output


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, arch: MiniMaxH3Architecture, *, context: ParallelContext) -> None:
        super().__init__()
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            expand_ratio=2,
            modality_num=1,
            context=context,
        )
        self.video_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.video_row_width,
            bias=True,
            gather_output=False,
            dtype=FP32,
            context=context,
        )
        self.audio_out = ColumnParallelLinear(
            arch.hidden_size,
            arch.audio_latent_channels,
            bias=True,
            gather_output=False,
            dtype=FP32,
            context=context,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        hidden = modulate_scale_shift(
            self.norm(value), shift, scale, inverse_indices, dtype=BF16
        ).to(FP32)
        return self.video_out(hidden), self.audio_out(hidden)

    def forward_audio_only(
        self,
        value: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project only audio velocity for the Base teacher pass."""

        shift, scale = self.adaln_proj(adaln_input)
        hidden = modulate_scale_shift(
            self.norm(value), shift, scale, inverse_indices, dtype=BF16
        ).to(FP32)
        return self.audio_out(hidden)

    def forward_video_only(
        self,
        value: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Project only video velocity when audio is supplied by Base10."""

        shift, scale = self.adaln_proj(adaln_input)
        hidden = modulate_scale_shift(
            self.norm(value), shift, scale, inverse_indices, dtype=BF16
        ).to(FP32)
        return self.video_out(hidden)


__all__ = [
    "BF16",
    "FP32",
    "MiniMaxH3Attention",
    "MiniMaxH3DiTBlock",
    "MiniMaxH3FinalLayer",
    "MiniMaxH3MLP",
    "MiniMaxH3Rope",
    "MiniMaxH3TimeEmbedder",
    "MiniMaxH3TokenRefiner",
    "active_native_stream_attention_metadata",
    "active_native_stream_mlp_video_stride",
    "apply_rope_qk",
    "set_inference_packed_dense_flash_attn3",
    "modulate_gate",
    "modulate_scale_shift",
    "native_stream_attention_metadata",
    "rope_cos_sin_cache",
    "set_native_stream_attention_hook",
]
