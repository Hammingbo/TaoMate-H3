"""Native model-parallel state owned by Digitalhuman H3."""

from .collectives import tp_all_gather, tp_all_reduce, ulysses_all_gather_rows
from .linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    suspend_fp8_activation_calibration,
    use_fp8_activation_scale_policy,
)
from .parallel import ParallelContext, current_parallel_context, use_parallel_context
from .ulysses import (
    heads_to_sequence,
    merged_qkv_sequence_to_heads,
    packed_qkv_sequence_to_heads,
    sequence_to_heads,
)

__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "ParallelContext",
    "RowParallelLinear",
    "current_parallel_context",
    "heads_to_sequence",
    "merged_qkv_sequence_to_heads",
    "packed_qkv_sequence_to_heads",
    "sequence_to_heads",
    "tp_all_gather",
    "tp_all_reduce",
    "ulysses_all_gather_rows",
    "suspend_fp8_activation_calibration",
    "use_fp8_activation_scale_policy",
    "use_parallel_context",
]
