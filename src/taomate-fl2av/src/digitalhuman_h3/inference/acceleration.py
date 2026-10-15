# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Mapping

from .fp8 import (
    H3_CALIBRATED_ONLY_FP8_POLICIES,
    H3_DYNAMIC_FP8_POLICIES,
)
from .w8a8 import H3_CUTLASS_W8A8_POLICIES

_DIT_LINEAR_PRECISIONS = frozenset(
    {
        "bf16",
        "fp8_dynamic",
        "fp8_dynamic_mlp",
        "fp8_calibrated",
    }
)
_STREAMING_ATTENTION_BACKENDS = frozenset(
    {
        "sdpa",
        "flash_attn3",
        "flash_attn3_fp8_segmented",
        "sage_attn",
        "sol_attn",
    }
)


@dataclass(frozen=True)
class InferenceAccelerationConfig:
    """Optional inference optimizations; every optimization defaults to off."""

    dit_linear_precision: str = "bf16"
    dit_fp8_policy: str | None = None
    dit_w8a8_policy: str | None = None
    dit_w8a8_workspace: bool = False
    streaming_attention_backend: str = "sdpa"
    direct_sage_kv_workspace: bool = False
    sol_attn_tau: float = 1.0
    sol_attn_dense_layers: int = 2
    sol_attn_dense_denoise_steps: int = 1
    sol_attn_route_current_video: bool = False
    sol_attn_static_summary_cache: bool = False
    fused_kernels: bool = False
    sol_engine_kernels: bool = False
    fused_qkv_preprocess: bool = False
    fused_lora_row_reduce: bool = False
    dit_lora_bf16_buffers: bool = False
    dit_lora_weight_merge: bool = False
    static_timestep_adaln_cache: bool = False
    packed_dense_flash_attn3: bool = False
    inplace_row_parallel_reduce: bool = False
    video_vae_fp8_linears: bool = False
    video_vae_fp8_attention: bool = False
    video_vae_exact_fusions: bool = False
    video_vae_batched_tiles: bool = False
    video_vae_balanced_cpu_output: bool = False
    video_vae_temporal_window_dp: bool = False
    video_vae_temporal_window_microbatch_size: int = 1
    video_vae_gpu_rgb24_output: bool = False
    video_vae_keep_on_device_after_decode: bool = False
    video_vae_decoder_tile_size: int | None = None
    video_vae_decoder_tile_overlap: int | None = None
    record_component_timings: bool = False

    def __post_init__(self) -> None:
        if self.dit_linear_precision not in _DIT_LINEAR_PRECISIONS:
            raise ValueError("unsupported DiT linear precision")
        if self.dit_fp8_policy is not None and (
            self.dit_fp8_policy not in H3_DYNAMIC_FP8_POLICIES
            or self.dit_linear_precision
            not in {"fp8_dynamic", "fp8_calibrated"}
        ):
            raise ValueError(
                "DiT FP8 policy requires fp8_dynamic or fp8_calibrated and a "
                "supported policy name"
            )
        if (
            self.dit_fp8_policy in H3_CALIBRATED_ONLY_FP8_POLICIES
            and self.dit_linear_precision != "fp8_calibrated"
        ):
            raise ValueError(
                f"{self.dit_fp8_policy} requires fp8_calibrated"
            )
        if (
            self.dit_w8a8_policy is not None
            and self.dit_w8a8_policy not in H3_CUTLASS_W8A8_POLICIES
        ):
            raise ValueError("unsupported DiT CUTLASS W8A8 policy")
        if (
            self.dit_w8a8_policy is not None
            and self.dit_linear_precision != "bf16"
        ):
            raise ValueError(
                "DiT CUTLASS W8A8 requires BF16 for unselected linears"
            )
        if self.dit_w8a8_policy is not None and self.dit_lora_weight_merge:
            raise ValueError("DiT CUTLASS W8A8 keeps LoRA online and cannot merge it")
        if self.dit_w8a8_workspace and self.dit_w8a8_policy is None:
            raise ValueError("DiT CUTLASS W8A8 workspace requires a W8A8 policy")
        if self.streaming_attention_backend not in _STREAMING_ATTENTION_BACKENDS:
            raise ValueError("unsupported streaming attention backend")
        if (
            self.direct_sage_kv_workspace
            and self.streaming_attention_backend != "sage_attn"
        ):
            raise ValueError(
                "Direct dense K/V workspace requires SageAttention"
            )
        if (
            self.sol_attn_static_summary_cache
            and self.streaming_attention_backend != "sol_attn"
        ):
            raise ValueError(
                "SOL static summary cache requires SOL-Attn"
            )
        if self.sol_attn_dense_layers < 0:
            raise ValueError("SOL-Attn dense layer count must be non-negative")
        if self.sol_attn_dense_denoise_steps < 0:
            raise ValueError("SOL-Attn dense denoise step count must be non-negative")
        if self.video_vae_decoder_tile_size is not None and (
            self.video_vae_decoder_tile_size <= 0
            or self.video_vae_decoder_tile_size % 16
        ):
            raise ValueError("Video VAE tile size must be a positive multiple of 16")
        if self.video_vae_decoder_tile_overlap is not None and (
            self.video_vae_decoder_tile_overlap < 0
            or self.video_vae_decoder_tile_overlap % 16
        ):
            raise ValueError(
                "Video VAE tile overlap must be a non-negative multiple of 16"
            )
        if (
            self.video_vae_temporal_window_dp
            and not self.video_vae_balanced_cpu_output
        ):
            raise ValueError(
                "Video VAE temporal-window DP requires balanced CPU output"
            )
        if self.video_vae_temporal_window_microbatch_size < 1:
            raise ValueError(
                "Video VAE temporal-window microbatch size must be positive"
            )
        if (
            self.video_vae_temporal_window_microbatch_size > 1
            and not self.video_vae_temporal_window_dp
        ):
            raise ValueError(
                "Video VAE temporal-window microbatching requires temporal-window DP"
            )
        if (
            self.video_vae_gpu_rgb24_output
            and not self.video_vae_temporal_window_dp
        ):
            raise ValueError(
                "Video VAE GPU RGB24 output requires temporal-window DP"
            )

    @property
    def enabled(self) -> bool:
        return any(
            (
                self.dit_linear_precision != "bf16",
                self.dit_w8a8_policy is not None,
                self.dit_w8a8_workspace,
                self.streaming_attention_backend != "sdpa",
                self.direct_sage_kv_workspace,
                self.sol_attn_static_summary_cache,
                self.fused_kernels,
                self.sol_engine_kernels,
                self.fused_qkv_preprocess,
                self.fused_lora_row_reduce,
                self.dit_lora_bf16_buffers,
                self.dit_lora_weight_merge,
                self.static_timestep_adaln_cache,
                self.packed_dense_flash_attn3,
                self.inplace_row_parallel_reduce,
                self.video_vae_fp8_linears,
                self.video_vae_fp8_attention,
                self.video_vae_exact_fusions,
                self.video_vae_batched_tiles,
                self.video_vae_balanced_cpu_output,
                self.video_vae_temporal_window_dp,
                self.video_vae_temporal_window_microbatch_size > 1,
                self.video_vae_gpu_rgb24_output,
                self.video_vae_keep_on_device_after_decode,
                self.video_vae_decoder_tile_size is not None,
                self.video_vae_decoder_tile_overlap is not None,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)







class InferenceComponentTimer:
    """Collect asynchronous CUDA-event and wall timings without per-call sync."""

    SCHEMA = "digitalhuman_h3.inference.component_timings.v1"

    def __init__(
        self,
        torch: Any,
        *,
        rank: int,
        world_size: int,
        split_streaming_dit: bool = False,
    ) -> None:
        self.torch = torch
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.split_streaming_dit = bool(
            split_streaming_dit or os.environ.get("H3_PROGRESSIVE_STAGE_TIMING") == "1"
        )
        self._events: dict[str, list[tuple[Any, Any]]] = {}
        self._wall_seconds: dict[str, float] = {}
        self._active_module_events: list[tuple[str, Any]] = []
        self._hook_handles: list[Any] = []

    def attach_transformer(self, transformer: Any) -> None:
        if self._hook_handles:
            raise RuntimeError("DiT timing hook is already attached")

        def before(_module: Any, _args: Any, _kwargs: Any) -> None:
            component = "dit"
            if self.split_streaming_dit:
                from digitalhuman_h3.streaming.capability import (
                    official_attention_denoise_step_index,
                    official_attention_hook_mode,
                )

                component = {
                    "noisy": "dit_denoise",
                    "clean_commit": "dit_clean_commit",
                    "progress_commit": "dit_progress_commit",
                    "fused_progress_denoise": "dit_fused_progress_denoise",
                    "full_prefix_capture": "dit_full_prefix_capture",
                }.get(official_attention_hook_mode(), "dit")
                if (
                    component in {"dit_denoise", "dit_fused_progress_denoise", "dit_progress_commit"}
                    and os.environ.get("H3_PROGRESSIVE_STAGE_TIMING") == "1"
                ):
                    component += {0: "_low0", 1: "_low16", 2: "_high33"}.get(
                        official_attention_denoise_step_index(), ""
                    )
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self._active_module_events.append((component, event))

        def after(_module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
            if not self._active_module_events:
                raise RuntimeError("DiT timing event stack is empty")
            component, start = self._active_module_events.pop()
            stop = self.torch.cuda.Event(enable_timing=True)
            stop.record()
            self._events.setdefault(component, []).append((start, stop))

        self._hook_handles = [
            transformer.register_forward_pre_hook(before, with_kwargs=True),
            transformer.register_forward_hook(after, with_kwargs=True),
        ]

    @contextmanager
    def region(self, component: str) -> Iterator[None]:
        start_event = self.torch.cuda.Event(enable_timing=True)
        stop_event = self.torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        try:
            yield
        finally:
            stop_event.record()
            self._events.setdefault(component, []).append((start_event, stop_event))
            self._wall_seconds[component] = self._wall_seconds.get(
                component, 0.0
            ) + (time.perf_counter() - wall_start)

    def receipt(self) -> dict[str, Any]:
        self.torch.cuda.synchronize()
        local: dict[str, Any] = {}
        for component in sorted(set(self._events) | set(self._wall_seconds)):
            events = self._events.get(component, ())
            local[component] = {
                "calls": len(events),
                "cuda_seconds": sum(
                    float(start.elapsed_time(stop)) for start, stop in events
                )
                / 1000.0,
                "wall_seconds": float(self._wall_seconds.get(component, 0.0)),
            }
        gathered: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(gathered, local)
        components: dict[str, Any] = {}
        names = sorted(
            {
                name
                for rank_payload in gathered
                for name in rank_payload
            }
        )
        for name in names:
            per_rank = [
                {
                    "rank": rank,
                    "calls": int(payload.get(name, {}).get("calls", 0)),
                    "cuda_seconds": float(
                        payload.get(name, {}).get("cuda_seconds", 0.0)
                    ),
                    "wall_seconds": float(
                        payload.get(name, {}).get("wall_seconds", 0.0)
                    ),
                }
                for rank, payload in enumerate(gathered)
            ]
            components[name] = {
                "calls_max": max(item["calls"] for item in per_rank),
                "cuda_seconds_max": max(
                    item["cuda_seconds"] for item in per_rank
                ),
                "cuda_seconds_mean": sum(
                    item["cuda_seconds"] for item in per_rank
                )
                / self.world_size,
                "wall_seconds_max": max(
                    item["wall_seconds"] for item in per_rank
                ),
                "per_rank": per_rank,
            }
        return {
            "schema": self.SCHEMA,
            "world_size": self.world_size,
            "streaming_dit_split": self.split_streaming_dit,
            "components": components,
        }


def apply_dit_inference_acceleration(
    torch: Any,
    transformer: Any,
    config: InferenceAccelerationConfig,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "digitalhuman_h3.inference.acceleration.v1",
        "config": config.to_dict(),
        "fused_kernels": None,
        "sol_engine_kernels": False,
        "fused_qkv_preprocess": False,
        "fused_lora_row_reduce": None,
        "dit_lora_bf16_buffers": None,
        "dit_lora_weight_merge": None,
        "dit_linear": None,
        "dit_w8a8": None,
        "dit_w8a8_workspace": None,
        "static_timestep_adaln_cache": False,
        "packed_dense_flash_attn3": False,
        "inplace_row_parallel_reduce": None,
    }
    if config.dit_w8a8_policy is not None:
        from digitalhuman_h3.inference.w8a8 import require_vllm_cutlass_w8a8

        # Resolve the optional H20 kernel before mutating transformer weights.
        require_vllm_cutlass_w8a8()
    if config.sol_engine_kernels:
        from digitalhuman_h3.inference.fused_kernels import (
            enable_h3_sol_engine_fusions,
        )

        enable_h3_sol_engine_fusions()
        receipt["fused_kernels"] = "enabled"
        receipt["sol_engine_kernels"] = True
    elif config.fused_kernels:
        from digitalhuman_h3.inference.fused_kernels import (
            enable_h3_inference_fused_kernels,
        )

        enable_h3_inference_fused_kernels()
        receipt["fused_kernels"] = "enabled"
    if config.fused_qkv_preprocess:
        from digitalhuman_h3.inference.fused_kernels import (
            enable_h3_fused_qkv_preprocess,
        )

        enable_h3_fused_qkv_preprocess()
        receipt["fused_qkv_preprocess"] = True
    if config.dit_lora_bf16_buffers:
        from digitalhuman_h3.inference.lora_checkpoint import (
            materialize_h3_lora_bf16_buffers_,
        )

        receipt["dit_lora_bf16_buffers"] = materialize_h3_lora_bf16_buffers_(
            transformer
        )
    if config.dit_lora_weight_merge:
        from digitalhuman_h3.inference.lora_checkpoint import (
            merge_h3_lora_weights_,
        )

        receipt["dit_lora_weight_merge"] = merge_h3_lora_weights_(transformer)
    if config.fused_lora_row_reduce and not config.dit_lora_weight_merge:
        from digitalhuman_h3.inference.lora_checkpoint import (
            enable_h3_fused_lora_row_reduce_,
        )

        receipt["fused_lora_row_reduce"] = enable_h3_fused_lora_row_reduce_(
            transformer
        )
    elif config.fused_lora_row_reduce:
        receipt["fused_lora_row_reduce"] = {
            "enabled": False,
            "disabled_reason": "online_lora_removed_by_weight_merge",
        }
    if config.dit_linear_precision != "bf16":
        from digitalhuman_h3.inference.fp8 import (
            quantize_h3_block_linears_calibrated_fp8_,
            quantize_h3_block_linears_dynamic_fp8_,
            quantize_h3_mlp_linears_dynamic_fp8_,
        )

        if config.dit_linear_precision == "fp8_calibrated":
            receipt["dit_linear"] = quantize_h3_block_linears_calibrated_fp8_(
                torch,
                transformer,
                policy=config.dit_fp8_policy,
            )
        elif config.dit_linear_precision == "fp8_dynamic_mlp":
            receipt["dit_linear"] = quantize_h3_mlp_linears_dynamic_fp8_(
                torch, transformer
            )
        else:
            receipt["dit_linear"] = quantize_h3_block_linears_dynamic_fp8_(
                torch,
                transformer,
                policy=config.dit_fp8_policy,
            )
    if config.dit_w8a8_policy is not None:
        from digitalhuman_h3.inference.w8a8 import (
            quantize_h3_interior_qkv_fc1_linears_cutlass_w8a8_,
        )

        w8a8_receipt = quantize_h3_interior_qkv_fc1_linears_cutlass_w8a8_(
            torch,
            transformer,
            policy=config.dit_w8a8_policy,
        )
        if config.dit_lora_bf16_buffers:
            w8a8_receipt["lora_branch"] = {
                "storage": "bfloat16_materialized_buffers",
                "compute": "bf16",
            }
        receipt["dit_w8a8"] = w8a8_receipt
    if config.dit_w8a8_workspace:
        from digitalhuman_h3.inference.w8a8 import (
            enable_h3_cutlass_w8a8_workspace_,
        )

        receipt["dit_w8a8_workspace"] = enable_h3_cutlass_w8a8_workspace_(
            torch,
            transformer,
        )
    if config.static_timestep_adaln_cache:
        transformer.enable_inference_static_timestep_cache()
        receipt["static_timestep_adaln_cache"] = True
    from digitalhuman_h3.model.layers import set_inference_packed_dense_flash_attn3

    set_inference_packed_dense_flash_attn3(config.packed_dense_flash_attn3)
    receipt["packed_dense_flash_attn3"] = config.packed_dense_flash_attn3
    from digitalhuman_h3.distributed.linear import RowParallelLinear

    row_modules = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, RowParallelLinear)
    ]
    for _, module in row_modules:
        module.inference_inplace_reduce_enabled = (
            config.inplace_row_parallel_reduce
        )
    if config.inplace_row_parallel_reduce:
        main_dit_modules = sum(
            name.startswith("blocks.") for name, _ in row_modules
        )
        receipt["inplace_row_parallel_reduce"] = {
            "schema": (
                "digitalhuman_h3.inference.inplace_row_parallel_reduce.v1"
            ),
            "module_count": len(row_modules),
            "main_dit_module_count": main_dit_modules,
            "main_dit_clones_avoided_per_forward": main_dit_modules,
        }
    return receipt




__all__ = [
    "InferenceAccelerationConfig",
    "InferenceComponentTimer",
    "add_inference_acceleration_arguments",
    "aggregate_component_timings",
    "apply_dit_inference_acceleration",
    "inference_acceleration_from_args",
]
