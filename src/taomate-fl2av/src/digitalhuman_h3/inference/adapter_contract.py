# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations
from typing import Any, Mapping

MODEL_REVISION = "29139ad62f28479297e305d690ee1521042133d4"
SOURCE_REVISION = "digitalhuman_h3_native_v1"
STAGE1B_TARGET_SOURCE = "ode_trajectory_intervals"
STAGE3_RUNTIME_SCHEMA = "digitalhuman_h3.training.stage3_runtime.v6"
STAGE3_INITIALIZED_EVALUATION_SCHEMA = "digitalhuman_h3.training.stage3_initialized_evaluation.v1"
STAGE3_INITIALIZED_EVALUATION_ENV = "H3_STAGE3_INITIALIZED_EVAL_CHECKPOINT"
_H3_LORA_TARGET_SUFFIXES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

def canonical_h3_lora_targets() -> tuple[str, ...]:
    """Return the exact 52-block × 4-projection H3 adapter inventory."""

    blocks = [f"token_refiner.blocks.{index}" for index in range(2)]
    blocks.extend(f"blocks.{index}" for index in range(50))
    return tuple(
        f"{block}.{suffix}"
        for block in blocks
        for suffix in _H3_LORA_TARGET_SUFFIXES
    )

def is_stage3_initialized_evaluation_marker(marker: Mapping[str, Any]) -> bool:
    """Recognize an initialized LoRA snapshot, never a trained/resumable state."""
    contract = marker.get("contract", {})
    runtime = contract.get("runtime_contract", {}) if isinstance(contract, Mapping) else {}
    return (
        marker.get("schema") == STAGE3_INITIALIZED_EVALUATION_SCHEMA
        and marker.get("status") == "complete"
        and marker.get("qualification_status") == "initialized_evaluation_only"
        and marker.get("evaluation_only") is True
        and type(marker.get("optimizer_step")) is int
        and marker["optimizer_step"] == 0
        and isinstance(contract, Mapping)
        and contract.get("stage") == "stage3"
        and contract.get("trainable_mode") == "lora"
        and isinstance(runtime, Mapping)
        and runtime.get("schema") == STAGE3_RUNTIME_SCHEMA
    )
