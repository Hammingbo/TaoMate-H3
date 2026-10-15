from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .collectives import tp_all_gather, tp_all_reduce, tp_all_reduce_inplace_
from .parallel import ParallelContext, current_parallel_context


_FP8_ACTIVATION_SCALE_POLICY: ContextVar[str] = ContextVar(
    "digitalhuman_h3_fp8_activation_scale_policy",
    default="current",
)
_FP8_ACTIVATION_CALIBRATION_SUSPENDED: ContextVar[bool] = ContextVar(
    "digitalhuman_h3_fp8_activation_calibration_suspended",
    default=False,
)


@contextmanager
def use_fp8_activation_scale_policy(policy: str) -> Iterator[None]:
    """Select local-current or cross-stage-shared calibrated FP8 scales."""

    if policy not in {"current", "prefix_shared"}:
        raise ValueError("unsupported FP8 activation scale policy")
    token = _FP8_ACTIVATION_SCALE_POLICY.set(policy)
    try:
        yield
    finally:
        _FP8_ACTIVATION_SCALE_POLICY.reset(token)


@contextmanager
def suspend_fp8_activation_calibration() -> Iterator[None]:
    """Run FP8 with per-forward scales without consuming calibration samples."""

    token = _FP8_ACTIVATION_CALIBRATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _FP8_ACTIVATION_CALIBRATION_SUSPENDED.reset(token)


def _linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    fp8_weight_scale: torch.Tensor | None,
    fp8_input_scale: torch.Tensor | None = None,
    cutlass_w8a8_weight_scale: torch.Tensor | None = None,
    cutlass_w8a8_workspace: object | None = None,
    cutlass_w8a8_weight_t: torch.Tensor | None = None,
    cutlass_w8a8_weight_scale_t: torch.Tensor | None = None,
) -> torch.Tensor:
    if cutlass_w8a8_weight_scale is not None:
        if fp8_weight_scale is not None:
            raise RuntimeError(
                "a linear cannot combine CUTLASS W8A8 with another quantizer"
            )
        from digitalhuman_h3.inference.w8a8 import cutlass_w8a8_linear

        return cutlass_w8a8_linear(
            torch,
            inputs,
            weight,
            cutlass_w8a8_weight_scale,
            bias=bias,
            workspace=cutlass_w8a8_workspace,
            weight_t=cutlass_w8a8_weight_t,
            weight_scale_t=cutlass_w8a8_weight_scale_t,
        )
    if fp8_weight_scale is None:
        return F.linear(inputs, weight, bias)
    shape = (*inputs.shape[:-1], int(weight.shape[0]))
    flattened = inputs.reshape(-1, int(inputs.shape[-1]))
    input_scale = fp8_input_scale
    from digitalhuman_h3.inference.fused_kernels import (
        quantize_fp8_e4m3,
        quantize_fp8_e4m3_dynamic,
    )

    if input_scale is None:
        quantized, input_scale = quantize_fp8_e4m3_dynamic(flattened)
    else:
        quantized = quantize_fp8_e4m3(flattened, input_scale)
    output = torch._scaled_mm(
        quantized,
        weight.t(),
        input_scale,
        fp8_weight_scale,
        bias=bias,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )
    return output.reshape(shape)


def _module_fp8_input_scale(
    module: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor | None:
    scale = getattr(module, "_fp8_activation_scale", None)
    if scale is None:
        return None
    calibration_suspended = _FP8_ACTIVATION_CALIBRATION_SUSPENDED.get()
    remaining = int(getattr(module, "_fp8_activation_calibration_remaining", 0))
    if remaining > 0 or calibration_suspended:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        observed = torch.clamp(
            inputs.detach().abs().amax().to(torch.float32) / fp8_max,
            min=1.0e-12,
        )
        if remaining > 0 and not calibration_suspended:
            torch.maximum(scale, observed, out=scale)
            module._fp8_activation_calibration_remaining = remaining - 1
        return observed
    margin = float(getattr(module, "_fp8_activation_scale_margin", 1.0))
    if _FP8_ACTIVATION_SCALE_POLICY.get() == "prefix_shared":
        prefix_scale = getattr(module, "_fp8_prefix_activation_scale", None)
        if prefix_scale is not None:
            return prefix_scale * margin
    return scale * margin


class ColumnParallelLinear(nn.Module):
    """Output-column sharded linear layer with checkpoint-compatible metadata."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        gather_output: bool,
        dtype: torch.dtype,
        weight_requires_grad: bool = True,
        context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = context or current_parallel_context()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.tp_size = self.parallel_context.tp_world_size
        self.tp_rank = self.parallel_context.tp_rank
        self.gather_output = bool(gather_output)
        if self.output_size % self.tp_size:
            raise ValueError(
                f"column output {self.output_size} is not divisible by TP={self.tp_size}"
            )
        self.local_output_size = self.output_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.local_output_size, self.input_size, dtype=dtype),
            requires_grad=weight_requires_grad,
        )
        self.bias = (
            nn.Parameter(torch.empty(self.local_output_size, dtype=dtype))
            if bias
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = _linear(
            inputs,
            self.weight,
            self.bias,
            getattr(self, "_fp8_weight_scale", None),
            _module_fp8_input_scale(self, inputs),
            getattr(self, "_cutlass_w8a8_weight_scale", None),
            getattr(self, "_cutlass_w8a8_workspace", None),
            getattr(self, "_cutlass_w8a8_weight_t", None),
            getattr(self, "_cutlass_w8a8_weight_scale_t", None),
        )
        if self.gather_output:
            output = tp_all_gather(output, context=self.parallel_context)
        return output

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full column-parallel weight shape differs")
        return value.narrow(
            0,
            self.tp_rank * self.local_output_size,
            self.local_output_size,
        ).contiguous()

    def full_bias_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size,):
            raise ValueError("full column-parallel bias shape differs")
        return value.narrow(
            0,
            self.tp_rank * self.local_output_size,
            self.local_output_size,
        ).contiguous()


class MergedColumnParallelLinear(ColumnParallelLinear):
    """Column sharding applied independently to fused logical projections."""

    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        *,
        bias: bool,
        gather_output: bool,
        dtype: torch.dtype,
        weight_requires_grad: bool = True,
        context: ParallelContext | None = None,
    ) -> None:
        self.output_sizes = tuple(int(value) for value in output_sizes)
        if not self.output_sizes or any(value <= 0 for value in self.output_sizes):
            raise ValueError("merged column output sizes must be positive")
        active = context or current_parallel_context()
        if any(value % active.tp_world_size for value in self.output_sizes):
            raise ValueError("every merged output must be divisible by TP")
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            dtype=dtype,
            weight_requires_grad=weight_requires_grad,
            context=active,
        )
        self.local_output_sizes = tuple(value // self.tp_size for value in self.output_sizes)

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full merged-column weight shape differs")
        pieces: list[torch.Tensor] = []
        offset = 0
        for size, local_size in zip(self.output_sizes, self.local_output_sizes):
            pieces.append(
                value.narrow(0, offset + self.tp_rank * local_size, local_size)
            )
            offset += size
        return torch.cat(pieces, dim=0).contiguous()

    def full_bias_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size,):
            raise ValueError("full merged-column bias shape differs")
        pieces: list[torch.Tensor] = []
        offset = 0
        for size, local_size in zip(self.output_sizes, self.local_output_sizes):
            pieces.append(
                value.narrow(0, offset + self.tp_rank * local_size, local_size)
            )
            offset += size
        return torch.cat(pieces, dim=0).contiguous()


class RowParallelLinear(nn.Module):
    """Input-row sharded linear layer with summed TP partial outputs."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        bias: bool,
        input_is_parallel: bool,
        dtype: torch.dtype,
        weight_requires_grad: bool = True,
        context: ParallelContext | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = context or current_parallel_context()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.tp_size = self.parallel_context.tp_world_size
        self.tp_rank = self.parallel_context.tp_rank
        self.input_is_parallel = bool(input_is_parallel)
        self.inference_inplace_reduce_enabled = False
        if self.input_size % self.tp_size:
            raise ValueError(
                f"row input {self.input_size} is not divisible by TP={self.tp_size}"
            )
        self.local_input_size = self.input_size // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(self.output_size, self.local_input_size, dtype=dtype),
            requires_grad=weight_requires_grad,
        )
        self.bias = (
            nn.Parameter(torch.empty(self.output_size, dtype=dtype)) if bias else None
        )

    def forward_local(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return this TP rank's partial output before reduction and bias."""

        if self.input_is_parallel:
            local_inputs = inputs
        else:
            local_inputs = inputs.narrow(
                -1,
                self.tp_rank * self.local_input_size,
                self.local_input_size,
            )
        return _linear(
            local_inputs,
            self.weight,
            None,
            getattr(self, "_fp8_weight_scale", None),
            _module_fp8_input_scale(self, local_inputs),
            getattr(self, "_cutlass_w8a8_weight_scale", None),
            getattr(self, "_cutlass_w8a8_workspace", None),
            getattr(self, "_cutlass_w8a8_weight_t", None),
            getattr(self, "_cutlass_w8a8_weight_scale_t", None),
        )

    def reduce_local_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.inference_inplace_reduce_enabled and not torch.is_grad_enabled():
            return tp_all_reduce_inplace_(output, context=self.parallel_context)
        return tp_all_reduce(output, context=self.parallel_context)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.forward_local(inputs)
        output = self.reduce_local_output(output)
        if self.bias is not None:
            output = output + self.bias
        return output

    def full_weight_slice(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape) != (self.output_size, self.input_size):
            raise ValueError("full row-parallel weight shape differs")
        return value.narrow(
            1,
            self.tp_rank * self.local_input_size,
            self.local_input_size,
        ).contiguous()


__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "RowParallelLinear",
    "use_fp8_activation_scale_policy",
    "suspend_fp8_activation_calibration",
]
