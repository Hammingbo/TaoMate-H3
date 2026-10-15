from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


INTERIOR_FC1_QKV_CUTLASS_W8A8_POLICY = "interior_fc1_qkv_v1"
H3_CUTLASS_W8A8_POLICIES = (INTERIOR_FC1_QKV_CUTLASS_W8A8_POLICY,)
_STREAM_LOAD_TEMPORARY_BYTES = 8 * 1024 * 1024


class CutlassW8A8Workspace:
    """Reuse exact dynamic-A8 and CUTLASS output storage per CUDA stream.

    The public vLLM helpers allocate activation, scale, and GEMM output tensors
    for every projection call.  H3 executes its QKV/FC1 projections serially on
    each rank, so the same storage can be reused after the preceding consumers
    have been enqueued on that stream.  Separate stream keys preserve this
    ordering guarantee when callers explicitly use more than one CUDA stream.
    """

    def __init__(self, torch: Any) -> None:
        namespace = getattr(getattr(torch, "ops", None), "_C", None)
        dynamic_quant = getattr(namespace, "dynamic_scaled_int8_quant", None)
        scaled_mm = getattr(namespace, "cutlass_scaled_mm", None)
        if not callable(dynamic_quant) or not callable(scaled_mm):
            raise RuntimeError(
                "CUTLASS W8A8 workspace requires vLLM dynamic INT8 quant and "
                "scaled-mm out operators"
            )
        self._torch = torch
        self._dynamic_quant = dynamic_quant
        self._scaled_mm = scaled_mm
        self._activation: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self._outputs: dict[tuple[Any, ...], Any] = {}

    def _stream_key(self, inputs: Any) -> tuple[Any, ...]:
        device = inputs.device
        if device.type != "cuda":
            return (device.type, device.index, 0)
        stream = self._torch.cuda.current_stream(device=device)
        return (device.type, device.index, int(stream.cuda_stream))

    def _activation_buffers(
        self,
        inputs: Any,
        stream_key: tuple[Any, ...],
    ) -> tuple[Any, Any]:
        rows, width = (int(inputs.shape[0]), int(inputs.shape[1]))
        key = (*stream_key, width)
        existing = self._activation.get(key)
        if existing is None or int(existing[0].shape[0]) < rows:
            quantized = self._torch.empty(
                (rows, width),
                device=inputs.device,
                dtype=self._torch.int8,
            )
            scales = self._torch.empty(
                (rows, 1),
                device=inputs.device,
                dtype=self._torch.float32,
            )
            existing = (quantized, scales)
            self._activation[key] = existing
        return existing[0][:rows], existing[1][:rows]

    def _output_buffer(
        self,
        inputs: Any,
        output_width: int,
        stream_key: tuple[Any, ...],
    ) -> Any:
        rows = int(inputs.shape[0])
        key = (*stream_key, int(output_width))
        existing = self._outputs.get(key)
        if existing is None or int(existing.shape[0]) < rows:
            existing = self._torch.empty(
                (rows, int(output_width)),
                device=inputs.device,
                dtype=self._torch.bfloat16,
            )
            self._outputs[key] = existing
        return existing[:rows]

    def linear(
        self,
        inputs: Any,
        weight_t: Any,
        weight_scale: Any,
    ) -> Any:
        stream_key = self._stream_key(inputs)
        quantized, activation_scale = self._activation_buffers(
            inputs,
            stream_key,
        )
        output = self._output_buffer(
            inputs,
            int(weight_t.shape[1]),
            stream_key,
        )
        self._dynamic_quant(
            quantized,
            inputs,
            activation_scale,
            None,
        )
        self._scaled_mm(
            output,
            quantized,
            weight_t,
            activation_scale,
            weight_scale,
            None,
        )
        return output


@lru_cache(maxsize=1)
def require_vllm_cutlass_w8a8() -> Any:
    """Load the vLLM CUTLASS INT8 kernels only for an enabled experiment."""

    try:
        backend = importlib.import_module("vllm._custom_ops")
    except ImportError as exc:
        raise RuntimeError(
            "CUTLASS W8A8 requires vLLM with its CUDA custom operations"
        ) from exc
    for name in ("scaled_int8_quant", "cutlass_scaled_mm"):
        if not callable(getattr(backend, name, None)):
            raise RuntimeError(f"vLLM CUDA custom operations are missing {name}")
    return backend


def resolve_h3_cutlass_w8a8_policy(
    linear_names: list[str],
    *,
    policy: str,
) -> dict[str, Any]:
    """Select the validated interior QKV/FC1 projections for W8A8."""

    if policy not in H3_CUTLASS_W8A8_POLICIES:
        raise ValueError("unsupported H3 CUTLASS W8A8 policy")
    projection_names = [
        name
        for name in linear_names
        if name.endswith((".attn.qkv_proj", ".mlp.fc1", ".mlp.fc2"))
    ]
    if not projection_names:
        raise ValueError("H3 CUTLASS W8A8 policy requires projection candidates")
    block_indices = sorted(
        {int(name.split(".", 2)[1]) for name in projection_names}
    )
    boundary_blocks = tuple((*block_indices[:2], *block_indices[-3:]))
    eligible_suffixes = (".attn.qkv_proj", ".mlp.fc1")
    quantized_names = [
        name
        for name in projection_names
        if name.endswith(eligible_suffixes)
        and int(name.split(".", 2)[1]) not in boundary_blocks
    ]
    quantized_set = set(quantized_names)
    return {
        "policy": policy,
        "quantized_linear_names": quantized_names,
        "bf16_protected_linear_names": [
            name for name in projection_names if name not in quantized_set
        ],
        "bf16_boundary_block_indices": list(boundary_blocks),
    }


def cutlass_w8a8_linear(
    torch: Any,
    inputs: Any,
    weight: Any,
    weight_scale: Any,
    *,
    bias: Any | None,
    workspace: CutlassW8A8Workspace | None = None,
    weight_t: Any | None = None,
    weight_scale_t: Any | None = None,
) -> Any:
    """Run dynamic-per-token A8 and per-output-channel W8 CUTLASS GEMM."""

    if bias is not None:
        raise RuntimeError("H3 CUTLASS W8A8 currently supports bias-free linears")
    shape = (*inputs.shape[:-1], int(weight.shape[0]))
    flattened = inputs.reshape(-1, int(inputs.shape[-1])).contiguous()
    if workspace is not None:
        output = workspace.linear(
            flattened,
            weight.t() if weight_t is None else weight_t,
            weight_scale.t() if weight_scale_t is None else weight_scale_t,
        )
        return output.reshape(shape)
    backend = require_vllm_cutlass_w8a8()
    quantized, activation_scale, _ = backend.scaled_int8_quant(flattened)
    output = backend.cutlass_scaled_mm(
        quantized,
        weight.t(),
        activation_scale,
        weight_scale.t(),
        torch.bfloat16,
    )
    return output.reshape(shape)


def enable_h3_cutlass_w8a8_workspace_(
    torch: Any,
    transformer: Any,
) -> dict[str, Any]:
    """Attach one lazy stream-aware workspace to all active H3 W8 linears."""

    require_vllm_cutlass_w8a8()
    selected = [
        (name, module)
        for name, module in transformer.named_modules()
        if name.startswith("blocks.")
        and getattr(module, "_cutlass_w8a8_weight_scale", None) is not None
    ]
    if not selected:
        raise RuntimeError("CUTLASS W8A8 workspace requires quantized H3 linears")
    workspace = CutlassW8A8Workspace(torch)
    for _, module in selected:
        module._cutlass_w8a8_workspace = workspace
        module._cutlass_w8a8_weight_t = module.weight.t()
        module._cutlass_w8a8_weight_scale_t = (
            module._cutlass_w8a8_weight_scale.t()
        )
    return {
        "schema": "digitalhuman_h3.inference.cutlass_w8a8_workspace.v1",
        "linear_count": len(selected),
        "workspace_count_per_rank": 1,
        "allocation_policy": "grow_only_per_cuda_stream_and_shape",
        "activation_quantization": "unchanged_dynamic_per_token_symmetric",
        "weight_quantization": "unchanged_static_per_output_channel_symmetric",
        "arithmetic_order_changed": False,
    }


def _quantize_cutlass_w8a8_weight(
    torch: Any,
    weight: Any,
) -> tuple[Any, Any]:
    """Shared exact W8 conversion used by post-load and streamed loading."""

    scale = torch.clamp(
        weight.abs().amax(dim=1, keepdim=True).to(torch.float32) / 127.0,
        min=1.0e-12,
    )
    quantized = torch.clamp(
        torch.round(weight / scale),
        -127,
        127,
    ).to(torch.int8)
    return quantized.contiguous(), scale.contiguous()


def prepare_cutlass_w8a8_stream_load_slot_(
    torch: Any,
    module: Any,
    *,
    name: str,
) -> None:
    """Replace one uninitialized BF16 weight with its final W8 storage."""

    if module.weight.dtype not in {torch.bfloat16, torch.int8}:
        raise TypeError(f"{name} expected an uninitialized BF16 or INT8 weight")
    if getattr(module, "bias", None) is not None:
        raise RuntimeError(f"{name} expected a bias-free H3 projection")
    if getattr(module, "_cutlass_w8a8_weight_scale", None) is not None:
        raise RuntimeError(f"{name} already owns CUTLASS W8A8 storage")
    weight = module.weight
    if weight.dtype == torch.bfloat16:
        module.weight = torch.nn.Parameter(
            torch.empty(weight.shape, dtype=torch.int8, device=weight.device),
            requires_grad=False,
        )
    elif weight.requires_grad:
        raise RuntimeError(f"{name} preallocated INT8 weight requires gradients")
    module.register_buffer(
        "_cutlass_w8a8_weight_scale",
        torch.empty(
            (int(weight.shape[0]), 1),
            dtype=torch.float32,
            device=weight.device,
        ),
        persistent=False,
    )
    module._cutlass_w8a8_stream_load_pending = True
    module._cutlass_w8a8_stream_loaded = False
    module._cutlass_w8a8_source_weight_bytes = int(weight.numel() * 2)


def load_cutlass_w8a8_stream_weight_(
    torch: Any,
    module: Any,
    source_weight: Any,
    *,
    name: str,
    maximum_temporary_bytes: int = _STREAM_LOAD_TEMPORARY_BYTES,
) -> None:
    """Quantize one BF16 source tensor into a prepared W8 slot by row chunks.

    CUTLASS uses one scale per output row, so splitting only the output-row
    dimension is mathematically identical to quantizing the complete matrix.
    The bounded chunks avoid materializing a full BF16 CUDA copy beside the
    final INT8 weight during distributed loading.
    """

    if getattr(module, "_cutlass_w8a8_stream_load_pending", None) is not True:
        raise RuntimeError(f"{name} has no pending CUTLASS W8A8 stream load")
    if module.weight.dtype != torch.int8 or source_weight.dtype != torch.bfloat16:
        raise TypeError(f"{name} streamed CUTLASS W8A8 dtype differs")
    if tuple(module.weight.shape) != tuple(source_weight.shape):
        raise ValueError(f"{name} streamed CUTLASS W8A8 shape differs")
    if source_weight.ndim != 2:
        raise ValueError(f"{name} streamed CUTLASS W8A8 weight must be 2D")
    if maximum_temporary_bytes <= 0:
        raise ValueError("maximum_temporary_bytes must be positive")
    bytes_per_row = int(source_weight.shape[1]) * int(
        source_weight.element_size()
    )
    rows_per_chunk = max(1, maximum_temporary_bytes // bytes_per_row)
    row_count = int(source_weight.shape[0])
    for start in range(0, row_count, rows_per_chunk):
        stop = min(start + rows_per_chunk, row_count)
        temporary = source_weight[start:stop].to(
            device=module.weight.device,
            dtype=torch.bfloat16,
        )
        quantized, scale = _quantize_cutlass_w8a8_weight(torch, temporary)
        module.weight[start:stop].copy_(quantized)
        module._cutlass_w8a8_weight_scale[start:stop].copy_(scale)
    module._cutlass_w8a8_stream_load_pending = False
    module._cutlass_w8a8_stream_loaded = True


def quantize_h3_interior_qkv_fc1_linears_cutlass_w8a8_(
    torch: Any,
    transformer: Any,
    *,
    policy: str,
) -> dict[str, Any]:
    """Quantize validated frozen interior QKV/FC1 base weights for inference."""

    require_vllm_cutlass_w8a8()
    from digitalhuman_h3.distributed import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    candidates: list[tuple[str, Any]] = []
    for name, module in transformer.named_modules():
        if not name.startswith("blocks.") or not name.endswith(
            (".attn.qkv_proj", ".mlp.fc1", ".mlp.fc2")
        ):
            continue
        # LoRA inference wraps the frozen TP linear in ``base``. Quantize only
        # that base so adapter enable/disable, including the Base audio teacher
        # bypass, keeps its existing semantics.
        selected = getattr(module, "base", module)
        if isinstance(selected, (ColumnParallelLinear, RowParallelLinear)):
            candidates.append((name, selected))
    selection = resolve_h3_cutlass_w8a8_policy(
        [name for name, _ in candidates],
        policy=policy,
    )
    selected_names = set(selection["quantized_linear_names"])
    selected_candidates = [
        (name, module)
        for name, module in candidates
        if name in selected_names
    ]
    if not selected_candidates:
        raise RuntimeError("H3 CUTLASS W8A8 policy selected no linears")

    # Validate the complete conversion set before replacing any parameter.
    # This keeps a repeated or mixed-precision call from leaving a half-
    # converted transformer when a later projection fails.
    streamed_names: list[str] = []
    for name, module in selected_candidates:
        weight_scale = getattr(module, "_cutlass_w8a8_weight_scale", None)
        if weight_scale is not None:
            if (
                module.weight.dtype != torch.int8
                or getattr(module, "_cutlass_w8a8_stream_load_pending", None)
                is not False
                or getattr(module, "_cutlass_w8a8_stream_loaded", None) is not True
            ):
                raise RuntimeError(f"{name} has incomplete CUTLASS W8A8 storage")
            streamed_names.append(name)
        elif module.weight.dtype != torch.bfloat16:
            raise TypeError(f"{name} expected BF16 weight before W8A8 conversion")
        if getattr(module, "bias", None) is not None:
            raise RuntimeError(f"{name} expected a bias-free H3 projection")

    converted: list[str] = []
    source_bytes = 0
    quantized_bytes = 0
    eligible_source_bytes = sum(
        int(
            getattr(
                module,
                "_cutlass_w8a8_source_weight_bytes",
                module.weight.numel() * module.weight.element_size(),
            )
        )
        for _, module in candidates
    )
    for name, module in selected_candidates:
        if name in streamed_names:
            source_bytes += int(module._cutlass_w8a8_source_weight_bytes)
            quantized_bytes += int(module.weight.numel() * module.weight.element_size())
        else:
            weight = module.weight.detach()
            quantized, scale = _quantize_cutlass_w8a8_weight(torch, weight)
            source_bytes += int(weight.numel() * weight.element_size())
            quantized_bytes += int(quantized.numel() * quantized.element_size())
            module.weight = torch.nn.Parameter(
                quantized,
                requires_grad=False,
            )
            module.register_buffer(
                "_cutlass_w8a8_weight_scale",
                scale,
                persistent=False,
            )
        converted.append(name)
    return {
        "schema": "digitalhuman_h3.inference.cutlass_w8a8.v1",
        "format": "int8_w8a8",
        "scope": "frozen_main_dit_base_weights_only",
        "policy": policy,
        "projection_set": "interior_qkv_and_fc1",
        "quantized_projection_suffixes": ["attn.qkv_proj", "mlp.fc1"],
        "activation_scale": "dynamic_per_token_symmetric",
        "weight_scale": "static_per_output_channel_symmetric",
        "kernel": "vllm_cutlass_scaled_mm",
        "eligible_linear_count": len(candidates),
        "linear_count": len(converted),
        "stream_load_linear_count": len(streamed_names),
        "bf16_protected_linear_count": len(
            selection["bf16_protected_linear_names"]
        ),
        "quantized_linear_names": converted,
        "bf16_protected_linear_names": selection[
            "bf16_protected_linear_names"
        ],
        "bf16_boundary_block_indices": selection[
            "bf16_boundary_block_indices"
        ],
        "source_weight_bytes": source_bytes,
        "quantized_weight_bytes": quantized_bytes,
        "eligible_source_weight_bytes": eligible_source_bytes,
        "eligible_weight_fraction": source_bytes / eligible_source_bytes,
        "bf16_projection_suffixes": ["attn.out_proj", "mlp.fc2"],
        "lora_branch": {
            "storage": "fp32_online_unmodified",
            "compute": "bf16",
        },
        "base_teacher_adapter_bypass": "lora_only",
        "base_teacher_base_precision": "int8_w8a8",
        "runtime_state": "transient_not_checkpoint_reconstructible",
    }


__all__ = [
    "CutlassW8A8Workspace",
    "H3_CUTLASS_W8A8_POLICIES",
    "INTERIOR_FC1_QKV_CUTLASS_W8A8_POLICY",
    "cutlass_w8a8_linear",
    "enable_h3_cutlass_w8a8_workspace_",
    "load_cutlass_w8a8_stream_weight_",
    "prepare_cutlass_w8a8_stream_load_slot_",
    "quantize_h3_interior_qkv_fc1_linears_cutlass_w8a8_",
    "require_vllm_cutlass_w8a8",
    "resolve_h3_cutlass_w8a8_policy",
]
