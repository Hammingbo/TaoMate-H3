"""Capability checks for the native H3 streaming attention path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


STREAM_HOOK_ABI = 3
_OFFICIAL_ATTENTION_HOOK: Any | None = None


class StreamingCapabilityError(RuntimeError):
    pass


def _streaming_topology_mismatches(server_args: Any) -> dict[str, object]:
    """Validate TP2 with single-host or padded-head multi-host Ulysses."""

    num_gpus = getattr(server_args, "num_gpus", None)
    pair = (getattr(server_args, "tp_size", None), getattr(server_args, "ulysses_degree", None))
    single_gpu_low_stage = (
        pair == (1, 1)
        and os.environ.get("H3_PROGRESSIVE_PP_STAGE") in {"0", "1"}
    )
    selected_topology = single_gpu_low_stage
    expected_tp = pair[0] if selected_topology else 2
    allowed_worlds = {expected_tp * pair[1]} if selected_topology else {2, 4, 8, 16}
    expected_ulysses = num_gpus // expected_tp if num_gpus in allowed_worlds else None
    mismatches: dict[str, object] = {}
    if num_gpus not in allowed_worlds:
        mismatches["num_gpus"] = num_gpus
    expected = {
        "tp_size": expected_tp,
        "ulysses_degree": expected_ulysses,
        "ring_degree": 1,
        "dp_size": 1,
        "enable_cfg_parallel": False,
        "cfg_parallel_degree": 1,
        "enable_torch_compile": False,
    }
    mismatches.update(
        {
            name: getattr(server_args, name, None)
            for name, value in expected.items()
            if getattr(server_args, name, None) != value
        }
    )
    return mismatches


@dataclass(frozen=True)
class StreamingCapability:
    backend: str
    hook_abi: int
    hook_setter: str
    layer_count: int | None
    topology_verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def require_streaming_capability(
    *,
    model: Any | None = None,
    server_args: Any | None = None,
) -> StreamingCapability:
    try:
        from ..model.layers import set_native_stream_attention_hook
    except ImportError as exc:
        raise StreamingCapabilityError("native H3 attention hook is unavailable") from exc
    if not callable(set_native_stream_attention_hook):
        raise StreamingCapabilityError("native H3 attention hook setter is unavailable")
    layer_count: int | None = None
    if model is not None:
        blocks = getattr(model, "blocks", None)
        if blocks is None or len(blocks) != 50:
            raise StreamingCapabilityError("H3 streaming requires 50 main DiT layers")
        names = tuple(getattr(block.attn, "layer_name", None) for block in blocks)
        if names != tuple(f"blocks.{index}" for index in range(50)):
            raise StreamingCapabilityError("native H3 attention layer identities differ")
        layer_count = len(blocks)
    topology_verified = False
    if server_args is not None:
        mismatches = _streaming_topology_mismatches(server_args)
        if mismatches:
            raise StreamingCapabilityError(
                f"native H3 streaming topology differs: {mismatches}"
            )
        topology_verified = True
    return StreamingCapability(
        backend="digitalhuman_h3_native_torch",
        hook_abi=STREAM_HOOK_ABI,
        hook_setter="digitalhuman_h3.model.layers.set_native_stream_attention_hook",
        layer_count=layer_count,
        topology_verified=topology_verified,
    )


def set_official_attention_hook(hook: Any | None) -> None:
    global _OFFICIAL_ATTENTION_HOOK

    from ..model.layers import set_native_stream_attention_hook

    set_native_stream_attention_hook(hook)
    _OFFICIAL_ATTENTION_HOOK = hook


def official_attention_hook_mode() -> str | None:
    hook = _OFFICIAL_ATTENTION_HOOK
    if hook is None:
        return None
    mode = getattr(hook, "mode", None)
    return None if mode is None else str(getattr(mode, "value", mode))


def official_attention_denoise_step_index() -> int | None:
    return getattr(_OFFICIAL_ATTENTION_HOOK, "_denoise_step_index", None)


__all__ = [
    "STREAM_HOOK_ABI",
    "StreamingCapability",
    "StreamingCapabilityError",
    "require_streaming_capability",
    "official_attention_hook_mode",
    "official_attention_denoise_step_index",
    "set_official_attention_hook",
]
