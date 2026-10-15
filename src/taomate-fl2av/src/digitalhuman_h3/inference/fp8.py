from __future__ import annotations

from typing import Any


QUALITY_FIRST_RESIDUAL_FP8_POLICY = "quality_first_residual_v1"
INTERIOR_FC1_ONLY_FP8_POLICY = "interior_fc1_only_v1"
INTERIOR_FC1_MID18_QKV_FP8_POLICY = "interior_fc1_mid18_qkv_v1"
H3_DYNAMIC_FP8_POLICIES = (
    QUALITY_FIRST_RESIDUAL_FP8_POLICY,
    INTERIOR_FC1_ONLY_FP8_POLICY,
    INTERIOR_FC1_MID18_QKV_FP8_POLICY,
)
H3_CALIBRATED_ONLY_FP8_POLICIES = (
    INTERIOR_FC1_ONLY_FP8_POLICY,
    INTERIOR_FC1_MID18_QKV_FP8_POLICY,
)
_INTERIOR_QKV_BLOCK_RANGES = {
    INTERIOR_FC1_MID18_QKV_FP8_POLICY: (16, 33),
}


def resolve_h3_dynamic_fp8_policy(
    linear_names: list[str],
    *,
    policy: str | None,
) -> dict[str, Any]:
    """Resolve an explicit policy against exact main-DiT runtime names.

    All explicit policies protect the two branch outputs that write into the
    residual stream and both depth boundaries of the 50-block transformer. The
    quality-first policy spends FP8 on QKV and the SwiGLU input projection;
    ``interior_fc1_only_v1`` keeps QKV in BF16, while
    ``interior_fc1_mid18_qkv_v1`` quantizes QKV only in blocks 16 through 33.
    """

    if policy is not None and policy not in H3_DYNAMIC_FP8_POLICIES:
        raise ValueError("unsupported H3 dynamic FP8 policy")
    if not linear_names:
        raise ValueError("H3 dynamic FP8 policy requires eligible linear names")
    if policy is None:
        return {
            "quantized_linear_names": list(linear_names),
            "bf16_protected_linear_names": [],
            "bf16_boundary_block_indices": [],
        }

    block_indices = sorted(
        {int(name.split(".", 2)[1]) for name in linear_names}
    )
    boundary_blocks = tuple((*block_indices[:2], *block_indices[-3:]))

    def should_quantize(name: str) -> bool:
        block_index = int(name.split(".", 2)[1])
        if block_index in boundary_blocks:
            return False
        if name.endswith(".mlp.fc1"):
            return True
        if not name.endswith(".attn.qkv_proj"):
            return False
        if policy == QUALITY_FIRST_RESIDUAL_FP8_POLICY:
            return True
        qkv_range = _INTERIOR_QKV_BLOCK_RANGES.get(policy)
        return (
            qkv_range is not None
            and qkv_range[0] <= block_index <= qkv_range[1]
        )

    quantized_names = [name for name in linear_names if should_quantize(name)]
    quantized_name_set = set(quantized_names)
    quantized_qkv_blocks = sorted(
        {
            int(name.split(".", 2)[1])
            for name in quantized_names
            if name.endswith(".attn.qkv_proj")
        }
    )
    all_qkv_blocks = {
        int(name.split(".", 2)[1])
        for name in linear_names
        if name.endswith(".attn.qkv_proj")
    }
    return {
        "quantized_linear_names": quantized_names,
        "bf16_protected_linear_names": [
            name for name in linear_names if name not in quantized_name_set
        ],
        "bf16_boundary_block_indices": list(boundary_blocks),
        "quantized_qkv_block_indices": quantized_qkv_blocks,
        "bf16_qkv_block_indices": sorted(
            all_qkv_blocks - set(quantized_qkv_blocks)
        ),
    }


class FP8DynamicLinear:
    """Inference-only FP8 replacement for an aligned ``torch.nn.Linear``."""

    @staticmethod
    def from_linear(torch: Any, source: Any) -> Any:
        class _FP8DynamicLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                weight = source.weight.detach()
                fp8_dtype = torch.float8_e4m3fn
                fp8_max = torch.finfo(fp8_dtype).max
                scale = torch.clamp(
                    weight.abs().amax().to(torch.float32) / fp8_max,
                    min=1.0e-12,
                )
                quantized = torch.clamp(
                    weight / scale, -fp8_max, fp8_max
                ).to(fp8_dtype)
                self.weight = torch.nn.Parameter(quantized, requires_grad=False)
                self.register_buffer("weight_scale", scale, persistent=True)
                self.bias = (
                    torch.nn.Parameter(
                        source.bias.detach().to(torch.float16),
                        requires_grad=False,
                    )
                    if source.bias is not None
                    else None
                )
                self.in_features = int(source.in_features)
                self.out_features = int(source.out_features)

            def forward(self, value: Any) -> Any:
                from digitalhuman_h3.inference.fused_kernels import (
                    quantize_fp8_e4m3_dynamic,
                )

                shape = (*value.shape[:-1], self.out_features)
                flattened = value.reshape(-1, self.in_features)
                quantized, input_scale = quantize_fp8_e4m3_dynamic(flattened)
                output = torch._scaled_mm(
                    quantized,
                    self.weight.t(),
                    input_scale,
                    self.weight_scale,
                    bias=self.bias,
                    out_dtype=torch.float16,
                    use_fast_accum=True,
                )
                return output.reshape(shape)

        return _FP8DynamicLinear()


def quantize_aligned_linears_dynamic_fp8_(
    torch: Any,
    module: Any,
) -> dict[str, Any]:
    """Replace Tensor-Core-aligned Linear children with dynamic FP8 modules."""

    names: list[str] = []

    def visit(parent: Any, prefix: str) -> None:
        for child_name, child in list(parent.named_children()):
            qualified = f"{prefix}.{child_name}" if prefix else child_name
            if (
                isinstance(child, torch.nn.Linear)
                and int(child.in_features) % 16 == 0
                and int(child.out_features) % 16 == 0
            ):
                setattr(
                    parent,
                    child_name,
                    FP8DynamicLinear.from_linear(torch, child),
                )
                names.append(qualified)
            else:
                visit(child, qualified)

    visit(module, "")
    if not names:
        raise RuntimeError("no aligned Linear modules were selected for FP8")
    return {
        "schema": "digitalhuman_h3.inference.module_fp8_linear.v1",
        "format": "float8_e4m3fn",
        "activation_scale": "dynamic_per_tensor",
        "weight_scale": "static_per_tensor",
        "linear_count": len(names),
    }


def _quantize_h3_block_linears_fp8_(
    torch: Any,
    transformer: Any,
    *,
    activation_calibration_forwards: int | None,
    activation_scale_margin: float,
    mlp_only: bool = False,
    policy: str | None = None,
) -> dict[str, Any]:
    """Quantize the dominant DiT block linears in place for inference only."""

    from digitalhuman_h3.distributed import (
        ColumnParallelLinear,
        RowParallelLinear,
    )

    if mlp_only and policy is not None:
        raise ValueError("H3 dynamic FP8 policy cannot be combined with MLP-only FP8")
    if (
        activation_calibration_forwards is None
        and policy in H3_CALIBRATED_ONLY_FP8_POLICIES
    ):
        raise ValueError(f"{policy} requires calibrated FP8 activation scales")

    suffixes = (
        (".mlp.fc1", ".mlp.fc2")
        if mlp_only
        else (
            ".attn.qkv_proj",
            ".attn.out_proj",
            ".mlp.fc1",
            ".mlp.fc2",
        )
    )
    candidates: list[tuple[str, Any]] = []
    for name, module in transformer.named_modules():
        if not name.startswith("blocks.") or not name.endswith(suffixes):
            continue
        # In LoRA inference the canonical projection is wrapped after the
        # checkpoint is loaded. Quantize the frozen parallel-linear base and
        # leave the adapter tensors untouched.
        selected = getattr(module, "base", module)
        if isinstance(selected, (ColumnParallelLinear, RowParallelLinear)):
            candidates.append((name, selected))
    if not candidates:
        raise RuntimeError("no H3 DiT block linears were selected for FP8")

    selection = resolve_h3_dynamic_fp8_policy(
        [name for name, _ in candidates],
        policy=policy,
    )
    selected_name_set = set(selection["quantized_linear_names"])
    selected_candidates = [
        (name, module)
        for name, module in candidates
        if name in selected_name_set
    ]
    if not selected_candidates:
        raise RuntimeError("H3 dynamic FP8 policy selected no block linears")

    names: list[str] = []
    source_bytes = 0
    quantized_bytes = 0
    eligible_source_bytes = sum(
        int(module.weight.numel() * module.weight.element_size())
        for _, module in candidates
    )
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    for name, selected in selected_candidates:
        weight = selected.weight.detach()
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"{name} expected BF16 weight before FP8 conversion")
        scale = torch.clamp(
            weight.abs().amax().to(torch.float32) / fp8_max,
            min=1.0e-12,
        )
        quantized = torch.clamp(weight / scale, -fp8_max, fp8_max).to(fp8_dtype)
        source_bytes += int(weight.numel() * weight.element_size())
        quantized_bytes += int(quantized.numel() * quantized.element_size())
        selected.weight = torch.nn.Parameter(quantized, requires_grad=False)
        selected.register_buffer("_fp8_weight_scale", scale, persistent=True)
        if activation_calibration_forwards is not None:
            selected.register_buffer(
                "_fp8_activation_scale",
                torch.zeros((), dtype=torch.float32, device=weight.device),
                persistent=False,
            )
            selected._fp8_activation_calibration_remaining = (
                activation_calibration_forwards
            )
            selected._fp8_activation_scale_margin = activation_scale_margin
        names.append(name)
    protected_names = selection["bf16_protected_linear_names"]
    if mlp_only:
        projection_set = "mlp_only"
    else:
        projection_set = {
            INTERIOR_FC1_ONLY_FP8_POLICY: "mlp_input_interior_blocks",
            INTERIOR_FC1_MID18_QKV_FP8_POLICY: (
                "mlp_input_interior_and_mid18_qkv_blocks"
            ),
            QUALITY_FIRST_RESIDUAL_FP8_POLICY: (
                "qkv_and_mlp_input_interior_blocks"
            ),
        }.get(policy, "attention_and_mlp")
    if policy == INTERIOR_FC1_ONLY_FP8_POLICY:
        policy_reason = (
            "protect attention, residual-write projections, and both "
            "representation boundaries (first two and last three blocks)"
        )
    elif policy in _INTERIOR_QKV_BLOCK_RANGES:
        qkv_start, qkv_stop = _INTERIOR_QKV_BLOCK_RANGES[policy]
        policy_reason = (
            f"protect QKV outside blocks {qkv_start} through {qkv_stop}, "
            "all residual-write projections, and both representation boundaries"
        )
    else:
        policy_reason = (
            "protect residual-write projections and both representation "
            "boundaries (first two and last three blocks)"
        )
    policy_rules = (
        {
            "quantized_projection_suffixes": (
                ["mlp.fc1"]
                if policy == INTERIOR_FC1_ONLY_FP8_POLICY
                else ["attn.qkv_proj", "mlp.fc1"]
            ),
            "bf16_projection_suffixes": (
                ["attn.qkv_proj", "attn.out_proj", "mlp.fc2"]
                if policy == INTERIOR_FC1_ONLY_FP8_POLICY
                else ["attn.out_proj", "mlp.fc2"]
            ),
            "bf16_boundary_block_indices": selection[
                "bf16_boundary_block_indices"
            ],
            "quantized_qkv_block_indices": selection[
                "quantized_qkv_block_indices"
            ],
            "bf16_qkv_block_indices": selection["bf16_qkv_block_indices"],
            "reason": policy_reason,
        }
        if policy in H3_DYNAMIC_FP8_POLICIES
        else None
    )
    return {
        "schema": "digitalhuman_h3.inference.fp8.v3",
        "scope": "frozen_main_dit_base_weights_only",
        "excluded_components": [
            "lora_adapter_tensors",
            "patch_projections",
            "timestep_projections",
            "final_output_projections",
            "token_refiner",
            "condition_and_adaln",
        ],
        "format": "float8_e4m3fn",
        "activation_scale": (
            "dynamic_per_tensor"
            if activation_calibration_forwards is None
            else "calibrated_static_per_tensor"
        ),
        "activation_calibration_forwards": activation_calibration_forwards,
        "activation_scale_margin": activation_scale_margin,
        "activation_calibration_scope": (
            None
            if activation_calibration_forwards is None
            else "student_av_forwards_only"
        ),
        "base_model_bypass_forwards": (
            None
            if activation_calibration_forwards is None
            else "per_forward_scale_without_consuming_calibration"
        ),
        "weight_scale": "static_per_tensor",
        "projection_set": projection_set,
        "policy": policy or ("mlp_only_v1" if mlp_only else "all_main_blocks_v1"),
        "policy_rules": policy_rules,
        "eligible_linear_count": len(candidates),
        "linear_count": len(names),
        "bf16_protected_linear_count": len(protected_names),
        "quantized_linear_names": names,
        "bf16_protected_linear_names": protected_names,
        "eligible_source_weight_bytes": eligible_source_bytes,
        "source_weight_bytes": source_bytes,
        "quantized_weight_bytes": quantized_bytes,
        "eligible_weight_fraction": source_bytes / eligible_source_bytes,
    }


def quantize_h3_block_linears_dynamic_fp8_(
    torch: Any,
    transformer: Any,
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    return _quantize_h3_block_linears_fp8_(
        torch,
        transformer,
        activation_calibration_forwards=None,
        activation_scale_margin=1.0,
        policy=policy,
    )


def quantize_h3_mlp_linears_dynamic_fp8_(
    torch: Any,
    transformer: Any,
) -> dict[str, Any]:
    """Quantize only MLP projections, leaving Q/K/V and attention output BF16."""

    return _quantize_h3_block_linears_fp8_(
        torch,
        transformer,
        activation_calibration_forwards=None,
        activation_scale_margin=1.0,
        mlp_only=True,
    )


def quantize_h3_block_linears_calibrated_fp8_(
    torch: Any,
    transformer: Any,
    *,
    calibration_forwards: int = 5,
    scale_margin: float = 1.25,
    policy: str | None = None,
) -> dict[str, Any]:
    if calibration_forwards < 1:
        raise ValueError("FP8 calibration forwards must be positive")
    if scale_margin < 1.0:
        raise ValueError("FP8 activation scale margin must be at least one")
    return _quantize_h3_block_linears_fp8_(
        torch,
        transformer,
        activation_calibration_forwards=calibration_forwards,
        activation_scale_margin=scale_margin,
        policy=policy,
    )


def synchronize_h3_block_fp8_activation_scales_(
    torch: Any,
    transformer: Any,
    *,
    group: Any,
) -> dict[str, Any]:
    """Publish one shared clean-prefix scale per calibrated block linear."""

    selected: list[tuple[Any, Any]] = []
    for name, module in transformer.named_modules():
        scale = getattr(module, "_fp8_activation_scale", None)
        if not name.startswith("blocks.") or scale is None:
            continue
        if int(getattr(module, "_fp8_activation_calibration_remaining", 0)) != 0:
            raise RuntimeError(
                "calibrated FP8 activation scales must be frozen before "
                "cross-stage synchronization"
            )
        selected.append((module, scale))
    if not selected:
        raise RuntimeError("no calibrated H3 block-linear activation scales found")

    packed = torch.stack(
        [scale.detach().to(dtype=torch.float32) for _, scale in selected]
    )
    torch.distributed.all_reduce(
        packed,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )
    for index, (module, _) in enumerate(selected):
        prefix_scale = getattr(module, "_fp8_prefix_activation_scale", None)
        if prefix_scale is None:
            module.register_buffer(
                "_fp8_prefix_activation_scale",
                packed[index].detach().clone(),
                persistent=False,
            )
        else:
            prefix_scale.copy_(packed[index])
    return {
        "enabled": True,
        "schema": "digitalhuman_h3.inference.fp8_stage_scale_sync.v2",
        "linear_count": len(selected),
        "collective_count": 1,
        "reduction": "max",
        "group_scope": "corresponding_tp_lane_across_three_denoise_stages",
        "calibration_state": "frozen_before_sync",
        "target": "clean_prefix_only",
        "current_stage_scales_changed": False,
    }


__all__ = [
    "FP8DynamicLinear",
    "H3_CALIBRATED_ONLY_FP8_POLICIES",
    "H3_DYNAMIC_FP8_POLICIES",
    "INTERIOR_FC1_MID18_QKV_FP8_POLICY",
    "INTERIOR_FC1_ONLY_FP8_POLICY",
    "QUALITY_FIRST_RESIDUAL_FP8_POLICY",
    "quantize_aligned_linears_dynamic_fp8_",
    "quantize_h3_block_linears_calibrated_fp8_",
    "quantize_h3_block_linears_dynamic_fp8_",
    "quantize_h3_mlp_linears_dynamic_fp8_",
    "resolve_h3_dynamic_fp8_policy",
    "synchronize_h3_block_fp8_activation_scales_",
]
