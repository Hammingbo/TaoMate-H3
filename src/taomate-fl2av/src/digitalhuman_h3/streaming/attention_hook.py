from __future__ import annotations

import math
import os
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cache import (
    AUDIO_TOKEN_TAG,
    MAIN_LAYER_NAMES,
    VIDEO_TOKEN_TAG,
    AVKV,
    CleanAVKVCache,
    TemporaryAVPrefix,
    TemporaryVideoPrefix,
    VideoKV,
)


class HookMode(str, Enum):
    IDLE = "idle"
    NOISY = "noisy"
    CLEAN_COMMIT = "clean_commit"
    PROGRESS_COMMIT = "progress_commit"
    FUSED_PROGRESS_DENOISE = "fused_progress_denoise"
    FULL_PREFIX_CAPTURE = "full_prefix_capture"


@dataclass(frozen=True)
class _KVPart:
    key: Any
    value: Any
    cached_pair: Any | None = None


@dataclass(frozen=True)
class _LiveDocumentKVPlan:
    condition: tuple[_KVPart, ...]
    media: tuple[_KVPart, ...]


@dataclass
class _DenseKVWorkspace:
    key: Any | None = None
    value: Any | None = None
    capacity_rows: int = 0
    tensor_spec: tuple[Any, ...] | None = None
    part_layout: tuple[Any, ...] | None = None
    cached_pair_refs: tuple[Any | None, ...] = ()


class _ReusableDenseKVWorkspacePool:
    """Reuse contiguous SageAttention K/V inputs without changing row order."""

    SCHEMA = "digitalhuman_h3.inference.direct_sage_dense_kv_workspace.v1"

    def __init__(self) -> None:
        self._workspaces: dict[str, _DenseKVWorkspace] = {}
        self.calls = 0
        self.allocation_count = 0
        self.layout_change_count = 0
        self.static_pair_copy_count = 0
        self.static_pair_reuse_count = 0
        self.dynamic_pair_copy_count = 0
        self.static_tokens_copied = 0
        self.static_tokens_reused = 0
        self.dynamic_tokens_copied = 0
        self.max_active_rows = 0

    @staticmethod
    def _tensor_layout(tensor: Any) -> tuple[Any, ...]:
        return (
            tuple(int(item) for item in tensor.shape),
            tuple(int(item) for item in tensor.stride()),
            int(tensor.storage_offset()),
            tensor.dtype,
            tensor.device,
        )

    def materialize(
        self,
        *,
        layer_name: str,
        parts: tuple[_KVPart, ...],
    ) -> tuple[Any, Any]:
        if not parts:
            raise ValueError("dense K/V workspace requires at least one part")
        first_key = parts[0].key
        first_value = parts[0].value
        tail = tuple(int(item) for item in first_key.shape[1:])
        tensor_spec = (
            tail,
            first_key.dtype,
            first_key.device,
            tuple(int(item) for item in first_value.shape[1:]),
            first_value.dtype,
            first_value.device,
        )
        total_rows = sum(int(part.key.shape[0]) for part in parts)
        part_layout = tuple(
            (
                self._tensor_layout(part.key),
                self._tensor_layout(part.value),
                part.cached_pair is not None,
            )
            for part in parts
        )

        workspace = self._workspaces.setdefault(layer_name, _DenseKVWorkspace())
        if (
            workspace.key is None
            or workspace.value is None
            or workspace.tensor_spec != tensor_spec
            or workspace.capacity_rows < total_rows
        ):
            workspace.key = first_key.new_empty((total_rows, *tail))
            workspace.value = first_value.new_empty(
                (total_rows, *tuple(int(item) for item in first_value.shape[1:]))
            )
            workspace.capacity_rows = total_rows
            workspace.tensor_spec = tensor_spec
            workspace.part_layout = None
            workspace.cached_pair_refs = ()
            self.allocation_count += 1

        if workspace.part_layout != part_layout:
            workspace.part_layout = part_layout
            workspace.cached_pair_refs = tuple(None for _ in parts)
            self.layout_change_count += 1

        cached_refs = list(workspace.cached_pair_refs)
        offset = 0
        for index, part in enumerate(parts):
            rows = int(part.key.shape[0])
            key_target = workspace.key.narrow(0, offset, rows)
            value_target = workspace.value.narrow(0, offset, rows)
            cached_pair = part.cached_pair
            cached_ref = cached_refs[index]
            if (
                cached_pair is not None
                and cached_ref is not None
                and cached_ref() is cached_pair
            ):
                self.static_pair_reuse_count += 1
                self.static_tokens_reused += rows
            else:
                key_target.copy_(part.key)
                value_target.copy_(part.value)
                if cached_pair is None:
                    self.dynamic_pair_copy_count += 1
                    self.dynamic_tokens_copied += rows
                else:
                    cached_refs[index] = weakref.ref(cached_pair)
                    self.static_pair_copy_count += 1
                    self.static_tokens_copied += rows
            offset += rows
        workspace.cached_pair_refs = tuple(cached_refs)
        self.calls += 1
        self.max_active_rows = max(self.max_active_rows, total_rows)
        return (
            workspace.key.narrow(0, 0, total_rows),
            workspace.value.narrow(0, 0, total_rows),
        )

    def receipt(self) -> dict[str, Any]:
        allocated_bytes = sum(
            int(workspace.key.numel()) * int(workspace.key.element_size())
            + int(workspace.value.numel()) * int(workspace.value.element_size())
            for workspace in self._workspaces.values()
            if workspace.key is not None and workspace.value is not None
        )
        return {
            "schema": self.SCHEMA,
            "enabled": True,
            "workspace_count": len(self._workspaces),
            "calls": self.calls,
            "allocation_count": self.allocation_count,
            "layout_change_count": self.layout_change_count,
            "static_pair_copy_count": self.static_pair_copy_count,
            "static_pair_reuse_count": self.static_pair_reuse_count,
            "dynamic_pair_copy_count": self.dynamic_pair_copy_count,
            "static_tokens_copied": self.static_tokens_copied,
            "static_tokens_reused": self.static_tokens_reused,
            "dynamic_tokens_copied": self.dynamic_tokens_copied,
            "max_active_rows": self.max_active_rows,
            "allocated_bytes": allocated_bytes,
            "content_order": "condition_cached_prefix_current_unchanged",
        }


def _plan_live_document_kv(
    *,
    condition_key: Any,
    condition_value: Any,
    allowed_prefixes: tuple[Any, ...],
    current_media_key: Any,
    current_media_value: Any,
    previous_media_key: Any | None = None,
    previous_media_value: Any | None = None,
) -> _LiveDocumentKVPlan:
    """Plan the same condition/media visibility used by Stage2/3 training."""

    condition = _KVPart(condition_key, condition_value)
    media = [
        condition,
        *(
            _KVPart(pair.key, pair.value, cached_pair=pair)
            for pair in allowed_prefixes
        ),
    ]
    if previous_media_key is not None:
        media.append(_KVPart(previous_media_key, previous_media_value))
    media.append(_KVPart(current_media_key, current_media_value))
    return _LiveDocumentKVPlan(condition=(condition,), media=tuple(media))


def _select_current_media_kv_indices(
    torch: Any,
    *,
    commit_mask_segment: Any,
    segment_tags: Any,
    video_stride: int,
) -> Any:
    """Select current media K/V, striding only committed video rows."""

    keep = commit_mask_segment.clone()
    if video_stride > 1:
        video_rows = torch.nonzero(
            commit_mask_segment & (segment_tags == VIDEO_TOKEN_TAG),
            as_tuple=False,
        ).view(-1)
        keep[video_rows] = False
        keep[video_rows[::video_stride]] = True
    return torch.nonzero(keep, as_tuple=False).view(-1)


def _select_fused_previous_media_indices(
    torch: Any,
    *,
    commit_mask_segment: Any,
    segment_tags: Any,
    video_only: bool,
) -> Any:
    """Select the previous fused document rows visible to the current one."""

    keep = commit_mask_segment
    if video_only:
        keep = keep & (segment_tags == VIDEO_TOKEN_TAG)
    return torch.nonzero(keep, as_tuple=False).view(-1)


def _select_full_prefix_capture_indices(
    torch: Any,
    *,
    token_tags: Any,
    commit_mask: Any,
    video_rows: int,
    audio_rows_per_channel: int | None,
) -> Any:
    """Select exact canonical AV rows while preserving channel-major ordering."""

    video_indices = torch.nonzero(
        commit_mask & (token_tags == VIDEO_TOKEN_TAG),
        as_tuple=False,
    ).view(-1)
    if int(video_indices.numel()) < video_rows:
        raise RuntimeError("full-prefix capture contains too few video rows")
    video_indices = video_indices[-video_rows:]
    if audio_rows_per_channel is None:
        return video_indices
    audio_indices = torch.nonzero(
        commit_mask & (token_tags == AUDIO_TOKEN_TAG),
        as_tuple=False,
    ).view(-1)
    if int(audio_indices.numel()) % 2:
        raise RuntimeError("full-prefix capture contains incomplete audio channels")
    available_audio_rows = int(audio_indices.numel()) // 2
    if available_audio_rows < audio_rows_per_channel:
        raise RuntimeError("full-prefix capture contains too few audio rows")
    audio_indices = audio_indices.reshape(2, available_audio_rows)[
        :, -audio_rows_per_channel:
    ].reshape(-1)
    return torch.cat((audio_indices, video_indices), dim=0)


def _sdpa_from_kv_parts(
    torch: Any,
    functional: Any,
    *,
    query: Any,
    parts: tuple[_KVPart, ...],
    scale: float,
) -> Any:
    """Execute exact SDPA over a reusable ordered K/V plan."""

    key = (
        parts[0].key
        if len(parts) == 1
        else torch.cat(tuple(part.key for part in parts), dim=0)
    )
    value = (
        parts[0].value
        if len(parts) == 1
        else torch.cat(tuple(part.value for part in parts), dim=0)
    )
    output = functional.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        key.transpose(0, 1).unsqueeze(0),
        value.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return output.squeeze(0).transpose(0, 1)


def _merge_segmented_flash_attention(
    torch: Any,
    partial_outputs: list[Any],
    partial_lse: list[Any],
    *,
    output_dtype: Any,
) -> Any:
    """Merge exact attention partitions from their per-query log normalizers."""

    lse = torch.stack(partial_lse, dim=0).to(torch.float32)
    normalizer = torch.logsumexp(lse, dim=0)
    weights = torch.exp(lse - normalizer.unsqueeze(0))
    stacked = torch.stack(partial_outputs, dim=0).to(torch.float32)
    return (
        stacked * weights.permute(0, 1, 3, 2).unsqueeze(-1)
    ).sum(dim=0).to(output_dtype)


def _live_documents_from_mask_or_hint(
    *,
    boundaries: tuple[int, ...],
    commit_mask: Any,
    live_documents_hint: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Resolve target documents without reading CUDA state when given a hint."""

    if live_documents_hint is None:
        return tuple(
            index
            for index, (start, stop) in enumerate(
                zip(boundaries, boundaries[1:])
            )
            if bool(commit_mask[start:stop].any().item())
        )
    if not isinstance(live_documents_hint, tuple):
        raise TypeError("live document hint must be a tuple")
    document_count = len(boundaries) - 1
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= document_count
        for index in live_documents_hint
    ):
        raise ValueError("live document hint contains an invalid document index")
    if tuple(sorted(set(live_documents_hint))) != live_documents_hint:
        raise ValueError("live document hint must be sorted and unique")
    if any(
        boundaries[index] == boundaries[index + 1]
        for index in live_documents_hint
    ):
        raise ValueError("live document hint references an empty document")
    return live_documents_hint


class H3StreamingAttentionHook:
    """Consumes official post-QK-norm/post-RoPE/post-Ulysses Q/K/V tensors."""

    def __init__(
        self,
        cache: CleanAVKVCache,
        *,
        attention_backend: str = "sdpa",
        history_kv_recent_full_blocks: int = 5,
        history_kv_older_video_stride: int = 1,
        history_kv_dynamic_density: float = 0.5,
        current_video_kv_stride: int = 1,
        dense_current_document_when_empty: bool = False,
        sol_attn_tau: float = 1.0,
        sol_attn_thresh_type: str = "diag",
        sol_attn_dense_layers: int = 2,
        sol_attn_dense_denoise_steps: int = 1,
        sol_attn_route_current_video: bool = False,
        direct_sage_kv_workspace: bool = False,
        sol_attn_static_summary_cache: bool = False,
    ) -> None:
        if attention_backend not in {
            "sdpa",
            "flash_attn3",
            "flash_attn3_fp8_dynamic",
            "flash_attn3_fp8_segmented",
            "flash_attn3_fp8_segmented_topk",
            "flex_dynamic_blocksparse",
            "sage_attn",
            "sol_attn",
        }:
            raise ValueError(
                f"unsupported H3 streaming attention backend: {attention_backend}"
            )
        self.cache = cache
        self.attention_backend = attention_backend
        self.history_kv_recent_full_blocks = history_kv_recent_full_blocks
        self.history_kv_older_video_stride = history_kv_older_video_stride
        if not 0.0 < history_kv_dynamic_density <= 1.0:
            raise ValueError("dynamic history KV density must be in (0, 1]")
        self.history_kv_dynamic_density = float(history_kv_dynamic_density)
        if current_video_kv_stride < 1:
            raise ValueError("current video KV stride must be positive")
        self.current_video_kv_stride = current_video_kv_stride
        self.dense_current_document_when_empty = bool(
            dense_current_document_when_empty
        )
        self.sol_attn_tau = float(sol_attn_tau)
        if sol_attn_thresh_type != "diag":
            raise ValueError("rectangular SOL-Attn currently uses diag routing")
        self.sol_attn_thresh_type = sol_attn_thresh_type
        self.sol_attn_dense_layers = int(sol_attn_dense_layers)
        self.sol_attn_dense_denoise_steps = int(sol_attn_dense_denoise_steps)
        self.sol_attn_route_current_video = bool(sol_attn_route_current_video)
        self.direct_sage_kv_workspace = bool(direct_sage_kv_workspace)
        if self.direct_sage_kv_workspace and attention_backend != "sage_attn":
            raise ValueError(
                "Direct dense K/V workspace requires SageAttention"
            )
        self.sol_attn_static_summary_cache = bool(
            sol_attn_static_summary_cache
        )
        if (
            self.dense_current_document_when_empty
            and current_video_kv_stride != 1
        ):
            raise ValueError(
                "dense current-document compatibility requires current video KV stride 1"
            )
        self._flash_attention_3 = None
        if attention_backend.startswith("flash_attn3") or attention_backend == "sol_attn":
            from flash_attn_interface import flash_attn_func

            self._flash_attention_3 = flash_attn_func
        self._sage_attention_sm90 = None
        if attention_backend == "sage_attn":
            from sageattention.core import sageattn_qk_int8_pv_fp8_cuda_sm90

            self._sage_attention_sm90 = sageattn_qk_int8_pv_fp8_cuda_sm90
        self.mode = HookMode.IDLE
        self.block_index: int | None = None
        self.full_prefix: TemporaryAVPrefix | TemporaryVideoPrefix | None = None
        self.fused_previous_full_prefix: (
            TemporaryAVPrefix | TemporaryVideoPrefix | None
        ) = None
        self.fused_previous_video_only = False
        self.fused_current_use_sink: bool | None = None
        self.fused_current_recent_blocks: int | None = None
        self._denoise_step_index: int | None = None
        self._capture_video_rows = 0
        self._capture_audio_rows_per_channel: int | None = None
        self._captured_full_prefix_layers: dict[str, AVKV | VideoKV] = {}
        self._live_documents: tuple[int, ...] | None = None
        self._live_document_boundaries: tuple[int, ...] | None = None
        self._fused_previous_media_indices: Any | None = None
        self._fused_previous_attention_media_indices: Any | None = None
        self._live_document_rows: dict[int, tuple[Any, Any]] = {}
        self._current_kv_indices: dict[int, Any] = {}
        self.fp8_history_reuse_count = 0
        self.fp8_runtime_kv_quantization_count = 0
        self.dynamic_block_sparse_call_count = 0
        self.dynamic_block_sparse_selected_blocks = 0
        self.dynamic_block_sparse_history_blocks = 0
        self.sol_attn_kernel_calls = 0
        self.sol_attn_dense_calls = 0
        self.sol_attn_query_tokens = 0
        self.sol_attn_key_tokens = 0
        self.sol_attn_static_cache_hits = 0
        self.sol_attn_static_cache_misses = 0
        self.sol_attn_static_summary_reused_blocks = 0
        self.sol_attn_static_summary_built_blocks = 0
        self.sol_attn_dynamic_or_boundary_summary_blocks = 0
        self._sol_attn_density_counts: Any | None = None
        self.sage_attn_calls = 0
        self.sage_attn_query_tokens = 0
        self.sage_attn_key_tokens = 0
        self._direct_sage_kv_workspaces = (
            _ReusableDenseKVWorkspacePool()
            if self.direct_sage_kv_workspace
            else None
        )

    def sol_attn_receipt(self) -> dict[str, Any] | None:
        if self.attention_backend != "sol_attn":
            return None
        density_counts = (
            None
            if self._sol_attn_density_counts is None
            else tuple(
                int(item)
                for item in self._sol_attn_density_counts.detach().cpu().tolist()
            )
        )
        route_selected_blocks, exact_blocks, routable_blocks, candidate_blocks = (
            density_counts or (0, 0, 0, 0)
        )
        return {
            "tau": self.sol_attn_tau,
            "thresh_type": self.sol_attn_thresh_type,
            "dense_layers": self.sol_attn_dense_layers,
            "dense_denoise_steps": self.sol_attn_dense_denoise_steps,
            "route_current_video": self.sol_attn_route_current_video,
            "static_summary_cache": self.sol_attn_static_summary_cache,
            "density_telemetry_enabled": self.sol_attn_static_summary_cache,
            "kernel_calls": self.sol_attn_kernel_calls,
            "dense_calls": self.sol_attn_dense_calls,
            "query_tokens": self.sol_attn_query_tokens,
            "key_tokens": self.sol_attn_key_tokens,
            "static_cache_hits": self.sol_attn_static_cache_hits,
            "static_cache_misses": self.sol_attn_static_cache_misses,
            "static_summary_reused_blocks": (
                self.sol_attn_static_summary_reused_blocks
            ),
            "static_summary_built_blocks": (
                self.sol_attn_static_summary_built_blocks
            ),
            "dynamic_or_boundary_summary_blocks": (
                self.sol_attn_dynamic_or_boundary_summary_blocks
            ),
            "route_selected_blocks": route_selected_blocks,
            "exact_blocks": exact_blocks,
            "routable_blocks": routable_blocks,
            "candidate_blocks": candidate_blocks,
            "selected_density": (
                None
                if routable_blocks == 0
                else route_selected_blocks / routable_blocks
            ),
            "exact_density": (
                None if candidate_blocks == 0 else exact_blocks / candidate_blocks
            ),
            "history_policy": (
                "rectangular_exact_condition_prefix_audio_sparse_current_"
                "and_history_video"
                if self.sol_attn_route_current_video
                else "rectangular_exact_condition_current_prefix_audio_"
                "sparse_history_video"
            ),
        }

    def sage_attn_receipt(self) -> dict[str, Any] | None:
        if self.attention_backend != "sage_attn":
            return None
        receipt = {
            "backend": "sageattn_qk_int8_pv_fp8_cuda_sm90",
            "tensor_layout": "NHD",
            "is_causal": False,
            "qk_quant_granularity": "per_thread",
            "pv_accum_dtype": "fp32+fp32",
            "smooth_k": True,
            "kernel_calls": self.sage_attn_calls,
            "query_tokens": self.sage_attn_query_tokens,
            "key_tokens": self.sage_attn_key_tokens,
        }
        if self._direct_sage_kv_workspaces is not None:
            receipt["direct_dense_kv_workspace"] = (
                self._direct_sage_kv_workspaces.receipt()
            )
        return receipt

    def attention_backend_receipt(self) -> dict[str, Any] | None:
        return self.sol_attn_receipt() or self.sage_attn_receipt()

    def activate(
        self,
        *,
        mode: HookMode,
        block_index: int,
        full_prefix: TemporaryAVPrefix | TemporaryVideoPrefix | None,
        fused_previous_full_prefix: (
            TemporaryAVPrefix | TemporaryVideoPrefix | None
        ) = None,
        fused_previous_video_only: bool = False,
        fused_current_use_sink: bool | None = None,
        fused_current_recent_blocks: int | None = None,
        denoise_step_index: int | None = None,
        phase_index: int | None = None,
    ) -> None:
        if self.mode is not HookMode.IDLE:
            raise RuntimeError("H3 streaming attention hook is already active")
        if block_index != self.cache.committed_blocks:
            raise ValueError("streaming attention block differs from clean history")
        if full_prefix is not None and full_prefix.contract != self.cache.contract:
            raise ValueError("temporary prefix and persistent cache contracts differ")
        if (
            fused_previous_full_prefix is not None
            and fused_previous_full_prefix.contract != self.cache.contract
        ):
            raise ValueError(
                "fused previous prefix and persistent cache contracts differ"
            )
        if (
            fused_previous_full_prefix is not None
            and mode is not HookMode.FUSED_PROGRESS_DENOISE
        ):
            raise ValueError(
                "fused previous prefix requires fused progress-denoise mode"
            )
        if fused_previous_video_only and mode is not HookMode.FUSED_PROGRESS_DENOISE:
            raise ValueError(
                "fused previous video-only visibility requires fused progress-denoise mode"
            )
        if (
            fused_current_use_sink is not None
            or fused_current_recent_blocks is not None
        ) and mode is not HookMode.FUSED_PROGRESS_DENOISE:
            raise ValueError(
                "fused current history policy requires fused progress-denoise mode"
            )
        if denoise_step_index is not None and mode not in {
            HookMode.NOISY,
            HookMode.FUSED_PROGRESS_DENOISE,
        }:
            raise ValueError(
                "denoise step index requires noisy or fused progress-denoise mode"
            )
        self.mode = mode
        self.block_index = block_index
        self.full_prefix = full_prefix
        self.fused_previous_full_prefix = fused_previous_full_prefix
        self.fused_previous_video_only = bool(fused_previous_video_only)
        self.fused_current_use_sink = fused_current_use_sink
        self.fused_current_recent_blocks = fused_current_recent_blocks
        self._denoise_step_index = denoise_step_index
        self._live_documents = None
        self._live_document_boundaries = None
        self._fused_previous_media_indices = None
        self._fused_previous_attention_media_indices = None
        self._live_document_rows = {}
        self._current_kv_indices = {}

    def deactivate(self) -> None:
        self.mode = HookMode.IDLE
        self.block_index = None
        self.full_prefix = None
        self.fused_previous_full_prefix = None
        self.fused_previous_video_only = False
        self.fused_current_use_sink = None
        self.fused_current_recent_blocks = None
        self._denoise_step_index = None
        self._capture_video_rows = 0
        self._capture_audio_rows_per_channel = None
        self._captured_full_prefix_layers = {}
        self._live_documents = None
        self._live_document_boundaries = None
        self._fused_previous_media_indices = None
        self._fused_previous_attention_media_indices = None
        self._live_document_rows = {}
        self._current_kv_indices = {}

    def begin_full_prefix_capture(
        self,
        *,
        video_rows: int,
        audio_rows_per_channel: int | None = None,
    ) -> None:
        if self.mode is not HookMode.IDLE:
            raise RuntimeError("H3 streaming attention hook is already active")
        if isinstance(video_rows, bool) or not isinstance(video_rows, int) or video_rows <= 0:
            raise ValueError("full-prefix capture video_rows must be positive")
        if audio_rows_per_channel is not None and (
            isinstance(audio_rows_per_channel, bool)
            or not isinstance(audio_rows_per_channel, int)
            or audio_rows_per_channel <= 0
        ):
            raise ValueError(
                "full-prefix capture audio_rows_per_channel must be positive"
            )
        self.mode = HookMode.FULL_PREFIX_CAPTURE
        self.block_index = self.cache.committed_blocks
        self._capture_video_rows = video_rows
        self._capture_audio_rows_per_channel = audio_rows_per_channel
        self._captured_full_prefix_layers = {}

    def _full_prefix_for_live_document(
        self,
        layer_name: str,
        *,
        document_index: int,
        previous_live_document: int | None,
    ) -> AVKV | VideoKV | None:
        """Select the prefix that existed when each fused document was run.

        In a lag-zero fused forward the previous document is a cache-recompute
        of block N-1 and must retain M_(N-2), while the current document must
        see the newly visible M_(N-1).  A single shared prefix would shift the
        previous document one block into the future at every boundary.
        """

        prefix = (
            self.fused_previous_full_prefix
            if (
                self.mode is HookMode.FUSED_PROGRESS_DENOISE
                and previous_live_document is not None
                and document_index == previous_live_document
            )
            else self.full_prefix
        )
        return None if prefix is None else prefix.get(layer_name)

    def finish_full_prefix_capture(
        self,
    ) -> TemporaryAVPrefix | TemporaryVideoPrefix:
        if self.mode is not HookMode.FULL_PREFIX_CAPTURE:
            raise RuntimeError("no H3 full-prefix capture is active")
        layers = self._captured_full_prefix_layers
        if tuple(layers) != MAIN_LAYER_NAMES:
            missing = [name for name in MAIN_LAYER_NAMES if name not in layers]
            self.deactivate()
            raise RuntimeError(
                f"full-prefix capture did not traverse all H3 layers: {missing}"
            )
        audio_rows_per_channel = self._capture_audio_rows_per_channel
        if audio_rows_per_channel is None:
            prefix = TemporaryVideoPrefix(layers, contract=self.cache.contract)
        else:
            prefix = TemporaryAVPrefix(
                layers,
                video_token_count=self._capture_video_rows,
                audio_token_count_per_channel=audio_rows_per_channel,
                contract=self.cache.contract,
            )
        self.deactivate()
        return prefix

    def __call__(
        self,
        *,
        attention: Any,
        layer_name: str,
        query: Any,
        key: Any,
        value: Any,
        token_tags: Any,
        commit_mask: Any,
        cu_seqlens: Any,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        live_documents_hint: tuple[int, ...] | None = None,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        del max_seqlen
        if self.mode is HookMode.IDLE:
            raise RuntimeError("patched H3 attention called outside a streaming step")
        if layer_name not in self.cache.contract.layer_names:
            raise ValueError(f"streaming hook reached a non-main attention layer: {layer_name}")
        if any(
            not torch.is_tensor(item)
            for item in (query, key, value, token_tags, commit_mask)
        ):
            raise TypeError("H3 streaming Q/K/V and token metadata must be tensors")
        expected_tail = (self.cache.contract.local_heads, self.cache.contract.head_dim)
        if (
            query.ndim != 3
            or key.shape != query.shape
            or value.shape != query.shape
            or tuple(query.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "post-Ulysses Q/K/V must be "
                f"[tokens, {expected_tail[0]}, {expected_tail[1]}]"
            )
        if query.dtype != torch.bfloat16 or not query.is_cuda:
            raise ValueError("official H3 streaming attention requires CUDA bfloat16 Q/K/V")
        if key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("H3 streaming Q/K/V dtypes differ")
        if key.device != query.device or value.device != query.device:
            raise ValueError("H3 streaming Q/K/V devices differ")
        if token_tags.ndim != 1 or token_tags.shape[0] != query.shape[0]:
            raise ValueError("post-Ulysses token tags do not cover Q/K/V rows")
        if (
            commit_mask.ndim != 1
            or commit_mask.dtype != torch.bool
            or commit_mask.shape[0] != query.shape[0]
        ):
            raise ValueError("post-Ulysses clean commit mask does not cover Q/K/V rows")
        if cu_seqlens_host is None:
            boundaries = tuple(int(item) for item in cu_seqlens.tolist())
        else:
            boundaries = tuple(int(item) for item in cu_seqlens_host)
        if (
            len(boundaries) < 2
            or boundaries[0] != 0
            or boundaries[-1] != query.shape[0]
            or any(left > right for left, right in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError("invalid packed H3 sequence boundaries")
        if self._live_documents is None:
            live_documents = _live_documents_from_mask_or_hint(
                boundaries=boundaries,
                commit_mask=commit_mask,
                live_documents_hint=live_documents_hint,
            )
            self._live_documents = live_documents
            self._live_document_boundaries = boundaries
        else:
            if boundaries != self._live_document_boundaries:
                raise ValueError("packed H3 boundaries changed within one streaming step")
            live_documents = self._live_documents
            if (
                live_documents_hint is not None
                and live_documents_hint != live_documents
            ):
                raise ValueError("live document hint changed within one streaming step")
        fused = self.mode is HookMode.FUSED_PROGRESS_DENOISE
        expected_live_documents = 2 if fused else 1
        if len(live_documents) != expected_live_documents:
            raise ValueError(
                "streaming session target-document count differs from hook mode: "
                f"expected={expected_live_documents}, actual={len(live_documents)}"
            )
        live_document = live_documents[-1]
        previous_live_document = live_documents[0] if fused else None

        capture = self.mode is HookMode.FULL_PREFIX_CAPTURE
        sparse_history = self.history_kv_older_video_stride > 1
        segmented_fp8 = self.attention_backend in {
            "flash_attn3_fp8_segmented",
            "flash_attn3_fp8_segmented_topk",
        }
        segmented_topk = self.attention_backend == "flash_attn3_fp8_segmented_topk"
        persistent = (
            None
            if capture or (segmented_fp8 and not sparse_history)
            else (
                self.cache.history_for_attention(
                    layer_name,
                    fused_current=False,
                    recent_full_blocks=self.history_kv_recent_full_blocks,
                    older_video_stride=self.history_kv_older_video_stride,
                )
                if sparse_history
                else self.cache.history(layer_name)
            )
        )
        outputs: list[Any] = []
        scale = float(attention.softmax_scale)
        layer_index = int(layer_name.split(".")[1])

        def exact_attention(
            q_segment: Any,
            parts: tuple[_KVPart, ...],
            *,
            force_dense: bool = False,
            sol_sink_tokens: int = 0,
        ) -> Any:
            if self.attention_backend == "sage_attn":
                use_dense_workspace = (
                    self._direct_sage_kv_workspaces is not None
                    and self.mode in {HookMode.NOISY, HookMode.CLEAN_COMMIT}
                    and len(parts) > 1
                    and any(part.cached_pair is not None for part in parts)
                )
                if use_dense_workspace:
                    key_segment, value_segment = (
                        self._direct_sage_kv_workspaces.materialize(
                            layer_name=layer_name,
                            parts=parts,
                        )
                    )
                else:
                    key_segment = (
                        parts[0].key
                        if len(parts) == 1
                        else torch.cat(tuple(part.key for part in parts), dim=0)
                    )
                    value_segment = (
                        parts[0].value
                        if len(parts) == 1
                        else torch.cat(tuple(part.value for part in parts), dim=0)
                    )
                if os.environ.get("H3_SAGE_FUSED_VALUE_PADDING") == "1":
                    from ..inference.sage_value_padding import sage_with_fused_value_padding
                    output = sage_with_fused_value_padding(
                        q_segment.unsqueeze(0), key_segment.unsqueeze(0),
                        value_segment.unsqueeze(0), sm_scale=scale)
                else:
                    output = self._sage_attention_sm90(
                        q_segment.unsqueeze(0),
                        key_segment.unsqueeze(0),
                        value_segment.unsqueeze(0),
                        tensor_layout="NHD",
                        is_causal=False,
                        qk_quant_gran="per_thread",
                        sm_scale=scale,
                        pv_accum_dtype="fp32+fp32",
                        smooth_k=True,
                        return_lse=False,
                    )
                self.sage_attn_calls += 1
                self.sage_attn_query_tokens += int(q_segment.shape[0])
                self.sage_attn_key_tokens += int(key_segment.shape[0])
                return output.squeeze(0)
            if self.attention_backend == "sol_attn":
                dense_mode = self.mode in {
                    HookMode.CLEAN_COMMIT,
                    HookMode.PROGRESS_COMMIT,
                    HookMode.FULL_PREFIX_CAPTURE,
                }
                dense_step = (
                    self.mode
                    in {HookMode.NOISY, HookMode.FUSED_PROGRESS_DENOISE}
                    and self._denoise_step_index is not None
                    and self._denoise_step_index
                    < self.sol_attn_dense_denoise_steps
                )
                if (
                    force_dense
                    or dense_mode
                    or dense_step
                    or layer_index < self.sol_attn_dense_layers
                ):
                    self.sol_attn_dense_calls += 1
                else:
                    key_offsets: list[int] = []
                    key_cursor = 0
                    for part in parts:
                        key_offsets.append(key_cursor)
                        key_cursor += int(part.key.shape[0])
                    key_segment = (
                        parts[0].key
                        if len(parts) == 1
                        else torch.cat(tuple(part.key for part in parts), dim=0)
                    )
                    value_segment = (
                        parts[0].value
                        if len(parts) == 1
                        else torch.cat(tuple(part.value for part in parts), dim=0)
                    )
                    from digitalhuman_h3.inference.sol_attention import (
                        SolStaticKVPart,
                        sol_attn_rectangular,
                    )

                    static_parts = ()
                    if self.sol_attn_static_summary_cache:
                        static_parts = tuple(
                            SolStaticKVPart(
                                start_token=key_offsets[index],
                                key=part.key,
                                value=part.value,
                                summary_cache=part.cached_pair.sol_summary_cache,
                            )
                            for index, part in enumerate(parts)
                            if part.cached_pair is not None
                        )
                        if self._sol_attn_density_counts is None:
                            self._sol_attn_density_counts = torch.zeros(
                                4,
                                dtype=torch.int64,
                                device=q_segment.device,
                            )

                    result = sol_attn_rectangular(
                        q_segment,
                        key_segment,
                        value_segment,
                        scale=scale,
                        tau=self.sol_attn_tau,
                        sink_tokens=sol_sink_tokens,
                        static_parts=static_parts,
                        telemetry_counts=self._sol_attn_density_counts,
                        return_receipt=self.sol_attn_static_summary_cache,
                    )
                    if self.sol_attn_static_summary_cache:
                        output, receipt = result
                        self.sol_attn_static_cache_hits += receipt.static_cache_hits
                        self.sol_attn_static_cache_misses += (
                            receipt.static_cache_misses
                        )
                        self.sol_attn_static_summary_reused_blocks += (
                            receipt.static_summary_reused_blocks
                        )
                        self.sol_attn_static_summary_built_blocks += (
                            receipt.static_summary_built_blocks
                        )
                        self.sol_attn_dynamic_or_boundary_summary_blocks += (
                            receipt.dynamic_or_boundary_summary_blocks
                        )
                    else:
                        output = result
                    query_tokens = int(q_segment.shape[0])
                    key_tokens = int(key_segment.shape[0])
                    self.sol_attn_kernel_calls += 1
                    self.sol_attn_query_tokens += query_tokens
                    self.sol_attn_key_tokens += key_tokens
                    return output
            segmented_call = segmented_fp8 and len(parts) > 1
            if self._flash_attention_3 is None:
                if segmented_call:
                    raise RuntimeError("segmented attention requires FlashAttention-3")
                return _sdpa_from_kv_parts(
                    torch,
                    functional,
                    query=q_segment,
                    parts=parts,
                    scale=scale,
                )

            q_input = q_segment.unsqueeze(0)
            if segmented_call:
                from digitalhuman_h3.inference.fused_kernels import (
                    quantize_fp8_e4m3_per_head,
                )

                q_input, q_descale = quantize_fp8_e4m3_per_head(q_input)
                partial_outputs: list[Any] = []
                partial_lse: list[Any] = []
                for part in parts:
                    cached_pair = part.cached_pair
                    if (
                        cached_pair is not None
                        and getattr(cached_pair, "fp8_key", None) is not None
                        and getattr(cached_pair, "fp8_value", None) is not None
                    ):
                        k_input = cached_pair.fp8_key.unsqueeze(0)
                        v_input = cached_pair.fp8_value.unsqueeze(0)
                        k_descale = cached_pair.fp8_key_descale
                        v_descale = cached_pair.fp8_value_descale
                        self.fp8_history_reuse_count += 1
                    else:
                        k_input, k_descale = quantize_fp8_e4m3_per_head(
                            part.key.unsqueeze(0)
                        )
                        v_input, v_descale = quantize_fp8_e4m3_per_head(
                            part.value.unsqueeze(0)
                        )
                        self.fp8_runtime_kv_quantization_count += 1
                    partial = self._flash_attention_3(
                        q_input,
                        k_input,
                        v_input,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        softmax_scale=scale,
                        causal=False,
                        num_splits=1,
                    )
                    if not isinstance(partial, tuple) or len(partial) < 2:
                        raise RuntimeError(
                            "segmented FP8 FlashAttention must return output and LSE"
                        )
                    partial_outputs.append(partial[0])
                    partial_lse.append(partial[1])
                return _merge_segmented_flash_attention(
                    torch,
                    partial_outputs,
                    partial_lse,
                    output_dtype=q_segment.dtype,
                ).squeeze(0)

            key_segment = (
                parts[0].key
                if len(parts) == 1
                else torch.cat(tuple(part.key for part in parts), dim=0)
            )
            value_segment = (
                parts[0].value
                if len(parts) == 1
                else torch.cat(tuple(part.value for part in parts), dim=0)
            )
            k_input = key_segment.unsqueeze(0)
            v_input = value_segment.unsqueeze(0)
            fp8_kwargs: dict[str, Any] = {}
            if self.attention_backend in {
                "flash_attn3_fp8_dynamic",
                "flash_attn3_fp8_segmented",
                "flash_attn3_fp8_segmented_topk",
            }:
                from digitalhuman_h3.inference.fused_kernels import (
                    quantize_fp8_e4m3_per_head,
                )

                q_input, q_descale = quantize_fp8_e4m3_per_head(q_input)
                k_input, k_descale = quantize_fp8_e4m3_per_head(k_input)
                v_input, v_descale = quantize_fp8_e4m3_per_head(v_input)
                self.fp8_runtime_kv_quantization_count += 1
                fp8_kwargs = {
                    "q_descale": q_descale,
                    "k_descale": k_descale,
                    "v_descale": v_descale,
                }
            output = self._flash_attention_3(
                q_input,
                k_input,
                v_input,
                **fp8_kwargs,
                softmax_scale=scale,
                causal=False,
                num_splits=1,
            )
            if isinstance(output, tuple):
                output = output[0]
            return output.squeeze(0)

        for index, (start, stop) in enumerate(zip(boundaries, boundaries[1:])):
            if start == stop:
                continue
            q_segment = query[start:stop]
            segment_key = key[start:stop]
            segment_value = value[start:stop]
            if index not in live_documents:
                outputs.append(
                    exact_attention(
                        q_segment,
                        (_KVPart(segment_key, segment_value),),
                        force_dense=True,
                    )
                )
                continue

            rows = self._live_document_rows.get(index)
            if rows is None:
                commit_mask_segment = commit_mask[start:stop]
                condition_indices = torch.nonzero(
                    ~commit_mask_segment,
                    as_tuple=False,
                ).view(-1)
                media_indices = torch.nonzero(
                    commit_mask_segment,
                    as_tuple=False,
                ).view(-1)
                rows = (condition_indices, media_indices)
                self._live_document_rows[index] = rows
            condition_indices, media_indices = rows

            current_media_indices = self._current_kv_indices.get(index)
            if current_media_indices is None:
                current_media_indices = (
                    media_indices
                    if self.current_video_kv_stride == 1
                    else _select_current_media_kv_indices(
                        torch,
                        commit_mask_segment=commit_mask[start:stop],
                        segment_tags=token_tags[start:stop],
                        video_stride=self.current_video_kv_stride,
                    )
                )
                self._current_kv_indices[index] = current_media_indices

            condition_query = q_segment.index_select(0, condition_indices)
            condition_key = segment_key.index_select(0, condition_indices)
            condition_value = segment_value.index_select(0, condition_indices)
            media_query = q_segment.index_select(0, media_indices)
            current_media_key = segment_key.index_select(
                0, current_media_indices
            )
            current_media_value = segment_value.index_select(
                0, current_media_indices
            )

            previous_media_key = None
            previous_media_value = None
            transient = (
                None
                if capture
                else self._full_prefix_for_live_document(
                    layer_name,
                    document_index=index,
                    previous_live_document=previous_live_document,
                )
            )
            if fused and index == live_document:
                assert previous_live_document is not None
                current_persistent = (
                    self.cache.history_for_attention(
                        layer_name,
                        fused_current=True,
                        recent_full_blocks=self.history_kv_recent_full_blocks,
                        older_video_stride=self.history_kv_older_video_stride,
                        **(
                            {}
                            if self.fused_current_recent_blocks is None
                            else {
                                "use_sink": bool(self.fused_current_use_sink),
                                "recent_blocks": self.fused_current_recent_blocks,
                            }
                        ),
                    )
                    if sparse_history
                    else self.cache.history_for_fused_current(
                        layer_name,
                        **(
                            {}
                            if self.fused_current_recent_blocks is None
                            else {
                                "use_sink": bool(self.fused_current_use_sink),
                                "recent_blocks": self.fused_current_recent_blocks,
                            }
                        ),
                    )
                )
                prefixes = tuple(
                    pair
                    for pair in (transient, current_persistent)
                    if pair is not None
                )
                previous_start = boundaries[previous_live_document]
                previous_stop = boundaries[previous_live_document + 1]
                if self._fused_previous_media_indices is None:
                    previous_commit_mask = commit_mask[
                        previous_start:previous_stop
                    ]
                    previous_tags = token_tags[previous_start:previous_stop]
                    previous_media = _select_fused_previous_media_indices(
                        torch,
                        commit_mask_segment=previous_commit_mask,
                        segment_tags=previous_tags,
                        video_only=False,
                    )
                    self._fused_previous_media_indices = previous_media.add(
                        previous_start
                    )
                    previous_attention_media = (
                        _select_fused_previous_media_indices(
                            torch,
                            commit_mask_segment=previous_commit_mask,
                            segment_tags=previous_tags,
                            video_only=True,
                        )
                        if self.fused_previous_video_only
                        else previous_media
                    )
                    self._fused_previous_attention_media_indices = (
                        previous_attention_media.add(previous_start)
                    )
                previous_media_indices = (
                    self._fused_previous_attention_media_indices
                )
                if previous_media_indices is None:
                    raise RuntimeError(
                        "fused H3 attention media indices were not resolved"
                    )
                previous_media_key = key.index_select(
                    0, previous_media_indices
                )
                previous_media_value = value.index_select(
                    0, previous_media_indices
                )
            elif segmented_fp8 and not sparse_history:
                prefix_list = [pair for pair in (transient,) if pair is not None]
                history_blocks = list(self.cache.history_blocks(layer_name))
                if segmented_topk and history_blocks:
                    selected_count = max(
                        1,
                        min(
                            len(history_blocks),
                            math.ceil(
                                len(history_blocks)
                                * self.history_kv_dynamic_density
                            ),
                        ),
                    )
                    if selected_count < len(history_blocks):
                        summaries = []
                        for pair in history_blocks:
                            if pair.key_summary is None:
                                raise RuntimeError(
                                    "dynamic segmented FP8 history lost its K summary"
                                )
                            summaries.append(pair.key_summary)
                        query_summary = media_query.float().mean(dim=0)
                        history_summary = torch.stack(summaries, dim=0)
                        scores = (
                            history_summary
                            .mul(query_summary.unsqueeze(0))
                            .sum(dim=-1)
                            .mean(dim=-1)
                        )
                        selected_indices = scores.topk(
                            selected_count,
                            sorted=False,
                        ).indices.sort().values.tolist()
                        history_blocks = [
                            history_blocks[int(selected)]
                            for selected in selected_indices
                        ]
                    self.dynamic_block_sparse_call_count += 1
                    self.dynamic_block_sparse_history_blocks += len(
                        self.cache.history_blocks(layer_name)
                    )
                    self.dynamic_block_sparse_selected_blocks += len(
                        history_blocks
                    )
                prefix_list.extend(history_blocks)
                prefixes = tuple(prefix_list)
            else:
                prefixes = tuple(
                    pair for pair in (transient, persistent) if pair is not None
                )

            if (
                self.dense_current_document_when_empty
                and self.mode is HookMode.NOISY
                and self.block_index == 0
                and not prefixes
                and previous_media_key is None
            ):
                # Match Direct inference for the first packed document: every
                # query sees the complete current document when no history or
                # temporary prefix exists. Later blocks keep the Stage3 mask.
                if self.attention_backend == "sol_attn":
                    full_document = (_KVPart(segment_key, segment_value),)
                    condition_output = exact_attention(
                        condition_query,
                        full_document,
                        force_dense=True,
                    )
                    media_output = exact_attention(
                        media_query,
                        full_document,
                        sol_sink_tokens=int(segment_key.shape[0]),
                    )
                    document_output = torch.zeros_like(q_segment)
                    document_output.index_copy_(
                        0, condition_indices, condition_output
                    )
                    document_output.index_copy_(0, media_indices, media_output)
                    outputs.append(document_output)
                else:
                    outputs.append(
                        exact_attention(
                            q_segment,
                            (_KVPart(segment_key, segment_value),),
                        )
                    )
                continue

            plan = _plan_live_document_kv(
                condition_key=condition_key,
                condition_value=condition_value,
                allowed_prefixes=prefixes,
                previous_media_key=previous_media_key,
                previous_media_value=previous_media_value,
                current_media_key=current_media_key,
                current_media_value=current_media_value,
            )
            condition_output = exact_attention(
                condition_query,
                plan.condition,
                force_dense=True,
            )

            if self.attention_backend == "sol_attn":
                audio_history, video_history = (
                    self.cache.sol_history_modality_parts(
                        layer_name,
                        fused_current=(fused and index == live_document),
                        **(
                            {}
                            if (
                                not fused
                                or index != live_document
                                or self.fused_current_recent_blocks is None
                            )
                            else {
                                "use_sink": bool(self.fused_current_use_sink),
                                "recent_blocks": self.fused_current_recent_blocks,
                            }
                        ),
                    )
                )
                if self.sol_attn_route_current_video and not fused:
                    segment_tags = token_tags[start:stop]
                    media_tags = segment_tags.index_select(0, media_indices)
                    current_tags = segment_tags.index_select(
                        0, current_media_indices
                    )
                    audio_query_indices = torch.nonzero(
                        media_tags == AUDIO_TOKEN_TAG,
                        as_tuple=False,
                    ).view(-1)
                    video_query_indices = torch.nonzero(
                        media_tags == VIDEO_TOKEN_TAG,
                        as_tuple=False,
                    ).view(-1)
                    current_audio_indices = torch.nonzero(
                        current_tags == AUDIO_TOKEN_TAG,
                        as_tuple=False,
                    ).view(-1)
                    current_video_indices = torch.nonzero(
                        current_tags == VIDEO_TOKEN_TAG,
                        as_tuple=False,
                    ).view(-1)
                    current_audio_key = current_media_key.index_select(
                        0, current_audio_indices
                    )
                    current_audio_value = current_media_value.index_select(
                        0, current_audio_indices
                    )
                    current_video_key = current_media_key.index_select(
                        0, current_video_indices
                    )
                    current_video_value = current_media_value.index_select(
                        0, current_video_indices
                    )
                    exact_parts = [
                        _KVPart(condition_key, condition_value),
                        _KVPart(current_audio_key, current_audio_value),
                    ]
                    if transient is not None:
                        exact_parts.append(
                            _KVPart(
                                transient.key,
                                transient.value,
                                cached_pair=transient,
                            )
                        )
                    exact_parts.extend(
                        _KVPart(history_key, history_value)
                        for history_key, history_value in audio_history
                    )
                    sol_parts = (
                        *exact_parts,
                        _KVPart(current_video_key, current_video_value),
                        *(
                            _KVPart(history_key, history_value)
                            for history_key, history_value in video_history
                        ),
                    )
                    audio_output = exact_attention(
                        media_query.index_select(0, audio_query_indices),
                        sol_parts,
                        force_dense=True,
                    )
                    video_output = exact_attention(
                        media_query.index_select(0, video_query_indices),
                        sol_parts,
                        sol_sink_tokens=sum(
                            int(part.key.shape[0]) for part in exact_parts
                        ),
                    )
                    media_output = torch.empty_like(media_query)
                    media_output.index_copy_(
                        0, audio_query_indices, audio_output
                    )
                    media_output.index_copy_(
                        0, video_query_indices, video_output
                    )
                else:
                    exact_parts = [
                        _KVPart(condition_key, condition_value),
                        _KVPart(current_media_key, current_media_value),
                    ]
                    if transient is not None:
                        exact_parts.append(
                            _KVPart(
                                transient.key,
                                transient.value,
                                cached_pair=transient,
                            )
                        )
                    if previous_media_key is not None:
                        exact_parts.append(
                            _KVPart(previous_media_key, previous_media_value)
                        )
                    exact_parts.extend(
                        _KVPart(history_key, history_value)
                        for history_key, history_value in audio_history
                    )
                    sol_parts = (
                        *exact_parts,
                        *(
                            _KVPart(history_key, history_value)
                            for history_key, history_value in video_history
                        ),
                    )
                    media_output = exact_attention(
                        media_query,
                        sol_parts,
                        force_dense=(fused and index == previous_live_document),
                        sol_sink_tokens=sum(
                            int(part.key.shape[0]) for part in exact_parts
                        ),
                    )
            else:
                media_output = None

            if (
                media_output is None
                and self.attention_backend == "flex_dynamic_blocksparse"
                and not fused
                and prefixes
            ):
                history_key = (
                    prefixes[0].key
                    if len(prefixes) == 1
                    else torch.cat(
                        tuple(pair.key for pair in prefixes), dim=0
                    )
                )
                if int(history_key.shape[0]) >= 128:
                    from digitalhuman_h3.inference.dynamic_block_sparse import (
                        dynamic_history_block_sparse_attention,
                    )

                    history_value = (
                        prefixes[0].value
                        if len(prefixes) == 1
                        else torch.cat(
                            tuple(pair.value for pair in prefixes), dim=0
                        )
                    )
                    dense_key = torch.cat(
                        (condition_key, current_media_key), dim=0
                    )
                    dense_value = torch.cat(
                        (condition_value, current_media_value), dim=0
                    )
                    media_output, receipt = dynamic_history_block_sparse_attention(
                        torch,
                        query=media_query,
                        history_key=history_key,
                        history_value=history_value,
                        current_key=dense_key,
                        current_value=dense_value,
                        softmax_scale=scale,
                        history_density=self.history_kv_dynamic_density,
                    )
                    self.dynamic_block_sparse_call_count += 1
                    self.dynamic_block_sparse_selected_blocks += int(
                        receipt["selected_history_blocks"]
                    )
                    self.dynamic_block_sparse_history_blocks += int(
                        receipt["history_blocks"]
                    )
            if media_output is None:
                media_output = exact_attention(
                    media_query,
                    plan.media,
                    force_dense=(fused and index == previous_live_document),
                )

            document_output = torch.zeros_like(q_segment)
            document_output.index_copy_(
                0, condition_indices, condition_output
            )
            document_output.index_copy_(0, media_indices, media_output)
            outputs.append(document_output)
        merged = torch.cat(outputs, dim=0)
        if merged.shape != query.shape:
            raise RuntimeError("streaming attention output shape differs from query")
        if capture:
            if layer_name in self._captured_full_prefix_layers:
                raise ValueError(f"duplicate full-prefix K/V layer: {layer_name}")
            capture_indices = _select_full_prefix_capture_indices(
                torch,
                token_tags=token_tags,
                commit_mask=commit_mask,
                video_rows=self._capture_video_rows,
                audio_rows_per_channel=self._capture_audio_rows_per_channel,
            )
            if self._capture_audio_rows_per_channel is None:
                self._captured_full_prefix_layers[layer_name] = VideoKV(
                    key=key.index_select(0, capture_indices).detach(),
                    value=value.index_select(0, capture_indices).detach(),
                )
            else:
                self._captured_full_prefix_layers[layer_name] = AVKV(
                    key=key.index_select(0, capture_indices).detach(),
                    value=value.index_select(0, capture_indices).detach(),
                )
        elif self.mode in {HookMode.CLEAN_COMMIT, HookMode.PROGRESS_COMMIT}:
            self.cache.stage(layer_name, key, value, token_tags, commit_mask)
        elif fused:
            assert previous_live_document is not None
            if self._fused_previous_media_indices is None:
                raise RuntimeError("fused H3 media indices were not resolved")
            self.cache.stage(
                layer_name,
                key,
                value,
                token_tags,
                commit_mask,
                commit_indices=self._fused_previous_media_indices,
            )
        return merged


__all__ = [
    "H3StreamingAttentionHook",
    "HookMode",
    "_live_documents_from_mask_or_hint",
    "_merge_segmented_flash_attention",
    "_plan_live_document_kv",
    "_sdpa_from_kv_parts",
    "_select_full_prefix_capture_indices",
    "_select_fused_previous_media_indices",
    "_select_current_media_kv_indices",
]
