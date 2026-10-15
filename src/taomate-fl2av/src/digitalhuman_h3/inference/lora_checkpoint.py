# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import importlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .adapter_contract import (
    MODEL_REVISION, SOURCE_REVISION, STAGE1B_TARGET_SOURCE,
    canonical_h3_lora_targets,
    STAGE3_INITIALIZED_EVALUATION_ENV,
    is_stage3_initialized_evaluation_marker,
)


class H3LoRACheckpointError(RuntimeError):
    pass


H3_DIFFUSIONNFT_LOCAL_STAGE3_FORMAL_ENV = (
    "H3_DIFFUSIONNFT_LOCAL_STAGE3_FORMAL"
)
H3_DIFFUSIONNFT_LOCAL_STAGE3_PHYSICAL_ENV = (
    "H3_DIFFUSIONNFT_LOCAL_STAGE3_PHYSICAL"
)
_STAGE3_CACHE_FILES = ("complete.json", "adapter_optimizer.pt")


@dataclass(frozen=True)
class LoadedH3LoRA:
    checkpoint_dir: Path
    stage: str
    optimizer_step: int
    role: str
    rank: int
    alpha: float
    state: Mapping[str, Any]
    loaded_tensor_count: int
    missing_tensor_count: int
    redundant_tensor_count: int

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


def read_h3_adapter_config(root: Path) -> Mapping[str, Any]:
    """Read the inference-only adapter configuration used by the public release."""
    path = root / "config.json"
    if not path.is_file():
        path = root / "adapter_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise H3LoRACheckpointError("H3 LoRA adapter config must be an object")
    return config


def _validate_rank_alpha(rank: Any, alpha: Any) -> None:
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank <= 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
    ):
        raise H3LoRACheckpointError("H3 LoRA rank/alpha is invalid")


def _select_lora_state(
    state: Mapping[str, Any], *, prefix: str, rank: int, torch: Any,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for target in canonical_h3_lora_targets():
        for leaf in ("lora_a", "lora_b"):
            name = f"{target}.{leaf}"
            stored_name = f"{prefix}{name}"
            if stored_name not in state:
                continue
            tensor = state[stored_name]
            if (
                not torch.is_tensor(tensor)
                or tensor.ndim != 2
                or tensor.dtype != torch.float32
                or (leaf == "lora_a" and int(tensor.shape[0]) != rank)
                or (leaf == "lora_b" and int(tensor.shape[1]) != rank)
            ):
                raise H3LoRACheckpointError(f"H3 LoRA tensor shape/dtype differs: {name}")
            selected[name] = tensor.detach().contiguous()
    return selected


def _load_safetensors_checkpoint(root: Path, *, role: str, torch: Any) -> LoadedH3LoRA:
    config = read_h3_adapter_config(root)
    if role != "generator" or config.get("weight_source") != "generator":
        raise H3LoRACheckpointError("FL2VA release requires online generator weights")
    if (
        config.get("stage") != "stage3"
        or config.get("model_revision") != MODEL_REVISION
        or config.get("source_revision") != SOURCE_REVISION
    ):
        raise H3LoRACheckpointError("H3 LoRA adapter binding differs")
    rank, alpha = config.get("rank"), config.get("alpha")
    _validate_rank_alpha(rank, alpha)
    step = config.get("optimizer_step")
    if type(step) is not int or step <= 0:
        raise H3LoRACheckpointError("H3 LoRA optimizer step is invalid")
    load_file = importlib.import_module("safetensors.torch").load_file
    state = load_file(str(root / "adapter_model.safetensors"), device="cpu")
    expected = {
        f"{target}.{leaf}"
        for target in canonical_h3_lora_targets()
        for leaf in ("lora_a", "lora_b")
    }
    present = set(state)
    if present != expected:
        raise H3LoRACheckpointError(
            "H3 LoRA tensor inventory differs: "
            f"missing={len(expected - present)}, unexpected={len(present - expected)}"
        )
    selected = _select_lora_state(state, prefix="", rank=rank, torch=torch)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"[adapter-load] H3 LoRA tensors loaded={len(selected)}", flush=True)
    return LoadedH3LoRA(
        checkpoint_dir=root, stage="stage3", optimizer_step=step, role=role,
        rank=rank, alpha=float(alpha), state=selected,
        loaded_tensor_count=len(selected), missing_tensor_count=0, redundant_tensor_count=0,
    )


def _checkpoint_read_roots(checkpoint_dir: Path) -> tuple[Path, Path]:
    """Resolve the logical checkpoint and its optional node-local read root."""

    logical = checkpoint_dir.expanduser().resolve(strict=True)
    formal_value = os.environ.get(H3_DIFFUSIONNFT_LOCAL_STAGE3_FORMAL_ENV)
    physical_value = os.environ.get(H3_DIFFUSIONNFT_LOCAL_STAGE3_PHYSICAL_ENV)
    if formal_value is None and physical_value is None:
        return logical, logical
    if formal_value is None or physical_value is None:
        raise H3LoRACheckpointError(
            "local Stage3 checkpoint binding is incomplete"
        )

    formal = Path(formal_value).expanduser()
    physical = Path(physical_value).expanduser()
    if not formal.is_absolute() or not physical.is_absolute():
        raise H3LoRACheckpointError(
            "local Stage3 checkpoint binding must use absolute paths"
        )
    formal = formal.resolve(strict=True)
    if logical != formal:
        return logical, logical
    if physical.is_symlink() or not physical.is_dir():
        raise H3LoRACheckpointError(
            "local Stage3 checkpoint physical root is not a directory"
        )

    for name in _STAGE3_CACHE_FILES:
        source = formal / name
        cached = physical / name
        if (
            not source.is_file()
            or cached.is_symlink()
            or not cached.is_file()
        ):
            raise H3LoRACheckpointError(
                f"local Stage3 checkpoint file is absent: {name}"
            )
        source_stat = source.stat()
        cached_stat = cached.stat()
        if (
            source_stat.st_size != cached_stat.st_size
            or source_stat.st_mtime_ns != cached_stat.st_mtime_ns
        ):
            raise H3LoRACheckpointError(
                f"local Stage3 checkpoint file identity differs: {name}"
            )
    return logical, physical


def _read_marker(
    checkpoint_dir: Path,
    *,
    allow_intermediate: bool = False,
) -> tuple[Path, Path, Mapping[str, Any]]:
    logical_root, read_root = _checkpoint_read_roots(checkpoint_dir)
    if not logical_root.is_dir():
        raise H3LoRACheckpointError("H3 LoRA checkpoint root is not a directory")
    marker_path = read_root / "complete.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3LoRACheckpointError(
            f"cannot read H3 LoRA completion marker: {exc}"
        ) from exc
    if not isinstance(marker, Mapping) or marker.get("status") != "complete":
        raise H3LoRACheckpointError("H3 LoRA checkpoint is not complete")
    qualification = marker.get("qualification_status")
    initialized_snapshot = is_stage3_initialized_evaluation_marker(marker)
    initialized_path = os.environ.get(STAGE3_INITIALIZED_EVALUATION_ENV)
    explicit_initialized_evaluation = (
        allow_intermediate and initialized_snapshot and initialized_path is not None
        and Path(initialized_path).expanduser().resolve() == logical_root
    )
    if not (
        isinstance(qualification, str)
        and (
            qualification.endswith("_qualified")
            or (allow_intermediate and qualification == "intermediate_restartable")
            or explicit_initialized_evaluation
        )
    ):
        raise H3LoRACheckpointError("H3 LoRA checkpoint is not GPU-qualified")
    return logical_root, read_root, marker


def load_h3_lora_checkpoint(
    checkpoint_dir: str | Path,
    *,
    role: str = "generator",
    torch_module: Any | None = None,
    allow_intermediate: bool = False,
) -> LoadedH3LoRA:
    """Load a public safetensors adapter or a qualified training checkpoint.

    DMD Stage1 and Stage3 checkpoints contain independent ``generator`` and
    ``fake_score`` banks.  AR-flow Stage1b and Stage2 contain only the causal
    generator bank.  Production generation defaults to ``generator``.
    The R5 child may explicitly bind one initialized evaluation-only Stage3
    snapshot; it is not treated as a qualified or resumable checkpoint.
    """

    if role not in {"generator", "fake_score"}:
        raise H3LoRACheckpointError("H3 LoRA role must be generator or fake_score")
    torch = torch_module or importlib.import_module("torch")
    root = Path(checkpoint_dir).expanduser().resolve(strict=True)
    if (root / "adapter_model.safetensors").is_file():
        return _load_safetensors_checkpoint(root, role=role, torch=torch)
    root, read_root, marker = _read_marker(
        Path(checkpoint_dir),
        allow_intermediate=allow_intermediate,
    )
    initialized_evaluation = is_stage3_initialized_evaluation_marker(marker)
    if initialized_evaluation and role != "generator":
        raise H3LoRACheckpointError("initialized evaluation snapshot exposes only the generator")
    contract = marker.get("contract")
    if not isinstance(contract, Mapping):
        raise H3LoRACheckpointError("H3 LoRA checkpoint contract is absent")
    stage = contract.get("stage")
    trainable = contract.get("trainable_config")
    if (
        stage not in {"stage1", "stage2", "stage3"}
        or contract.get("trainable_mode") != "lora"
        or contract.get("model_revision") != MODEL_REVISION
        or contract.get("source_revision") != SOURCE_REVISION
        or not isinstance(trainable, Mapping)
        or trainable.get("mode") != "lora"
    ):
        raise H3LoRACheckpointError("H3 LoRA checkpoint binding differs")
    lora = trainable.get("lora")
    if not isinstance(lora, Mapping):
        raise H3LoRACheckpointError("H3 LoRA checkpoint config is absent")
    stage1_variant = contract.get("stage1_variant")
    objective = contract.get("objective")
    stage1b_single_bank = (
        stage == "stage1"
        and (
            stage1_variant == "ar_flow_matching"
            or (
                stage1_variant is None
                and isinstance(objective, Mapping)
                and objective.get("target_source") == STAGE1B_TARGET_SOURCE
            )
        )
    )
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    _validate_rank_alpha(rank, alpha)
    if (stage == "stage2" or stage1b_single_bank) and role != "generator":
        raise H3LoRACheckpointError(
            "causal single-bank checkpoints expose only the generator adapter"
        )

    files = marker.get("files")
    record = files.get("adapter_optimizer.pt") if isinstance(files, Mapping) else None
    payload_path = read_root / "adapter_optimizer.pt"
    try:
        payload_size = payload_path.stat().st_size
    except OSError as exc:
        raise H3LoRACheckpointError(
            f"cannot inspect H3 LoRA tensor payload: {exc}"
        ) from exc
    if (
        not isinstance(record, Mapping)
        or record.get("size_bytes") != payload_size
    ):
        raise H3LoRACheckpointError("H3 LoRA tensor payload record differs")
    try:
        payload = torch.load(
            payload_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise H3LoRACheckpointError(f"cannot load H3 LoRA tensor payload: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"model", "optimizer"}:
        raise H3LoRACheckpointError("H3 LoRA tensor payload keys differ")
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise H3LoRACheckpointError("H3 LoRA model state is absent")

    targets = canonical_h3_lora_targets()
    leaves = ("lora_a", "lora_b")
    if stage in {"stage1", "stage3"} and not stage1b_single_bank:
        banks = ("generator", "fake_score")
        if stage == "stage3" and contract.get("objective", {}).get("anchor_video_weight", 0):
            banks += ("anchor_score",)
        expected = {
            f"{bank}.{target}.{leaf}"
            for bank in banks
            for target in targets
            for leaf in leaves
        }
        prefix = f"{role}."
    else:
        expected = {f"{target}.{leaf}" for target in targets for leaf in leaves}
        prefix = ""
    canonical_state: dict[str, Any] = {}
    for raw_name, tensor in model_state.items():
        if not isinstance(raw_name, str):
            continue
        name = raw_name.replace("._checkpoint_wrapped_module.", ".")
        if name in canonical_state:
            raise H3LoRACheckpointError(
                f"H3 LoRA tensor name collides after wrapper removal: {name}"
            )
        canonical_state[name] = tensor

    loaded_names = expected.intersection(canonical_state)
    missing_names = expected.difference(canonical_state)
    redundant_names = set(canonical_state).difference(expected)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(
            "[checkpoint-load] H3 LoRA tensors "
            f"loaded={len(loaded_names)} "
            f"missing={len(missing_names)} "
            f"redundant={len(redundant_names)}",
            flush=True,
        )
    selected = _select_lora_state(canonical_state, prefix=prefix, rank=rank, torch=torch)

    optimizer_step = marker.get("optimizer_step")
    if (isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int)
            or optimizer_step < 0 or (optimizer_step == 0 and not initialized_evaluation)):
        raise H3LoRACheckpointError("H3 LoRA optimizer step is invalid")
    return LoadedH3LoRA(
        checkpoint_dir=root,
        stage=str(stage),
        optimizer_step=optimizer_step,
        role=role,
        rank=rank,
        alpha=float(alpha),
        state=selected,
        loaded_tensor_count=len(loaded_names),
        missing_tensor_count=len(missing_names),
        redundant_tensor_count=len(redundant_names),
    )


def _tensor_parallel_coordinates(module: Any) -> tuple[int, int]:
    tp_size = int(module.tp_size)
    tp_rank = int(module.tp_rank)
    if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
        raise H3LoRACheckpointError("H3 inference LoRA TP coordinates are invalid")
    return tp_size, tp_rank


def _column_local_b(full_b: Any, module: Any) -> Any:
    torch = importlib.import_module("torch")
    tp_size, tp_rank = _tensor_parallel_coordinates(module)
    output_sizes = getattr(module, "output_sizes", None)
    if output_sizes is None:
        if int(full_b.shape[0]) % tp_size:
            raise H3LoRACheckpointError("H3 column LoRA output is not TP-divisible")
        shard = int(full_b.shape[0]) // tp_size
        return full_b.narrow(0, tp_rank * shard, shard).contiguous()
    pieces: list[Any] = []
    offset = 0
    for raw_size in output_sizes:
        size = int(raw_size)
        if size % tp_size:
            raise H3LoRACheckpointError("H3 merged-column LoRA output is not TP-divisible")
        shard = size // tp_size
        pieces.append(full_b.narrow(0, offset + tp_rank * shard, shard))
        offset += size
    if offset != int(full_b.shape[0]):
        raise H3LoRACheckpointError("H3 merged-column LoRA output inventory differs")
    return torch.cat(pieces, dim=0).contiguous()


def _row_local_a(full_a: Any, module: Any) -> Any:
    tp_size, tp_rank = _tensor_parallel_coordinates(module)
    input_size = int(full_a.shape[1])
    if input_size % tp_size:
        raise H3LoRACheckpointError("H3 row LoRA input is not TP-divisible")
    shard = input_size // tp_size
    return full_a.narrow(1, tp_rank * shard, shard).contiguous()


def apply_h3_lora_checkpoint(
    transformer: Any,
    checkpoint_dir: str | Path,
    *,
    role: str = "generator",
    allow_intermediate: bool = False,
) -> Mapping[str, Any]:
    """Install a TP-aware, inference-only LoRA bank on the live H3 DiT."""

    torch = importlib.import_module("torch")
    linear = importlib.import_module("digitalhuman_h3.distributed.linear")
    collectives = importlib.import_module("digitalhuman_h3.distributed.collectives")
    loaded = load_h3_lora_checkpoint(
        checkpoint_dir,
        role=role,
        torch_module=torch,
        allow_intermediate=allow_intermediate,
    )
    targets = canonical_h3_lora_targets()
    modules = dict(transformer.named_modules())
    supported_column = (linear.ColumnParallelLinear, linear.MergedColumnParallelLinear)
    supported_row = (linear.RowParallelLinear,)

    class H3InferenceParallelLinearLoRA(torch.nn.Module):
        def __init__(self, base: Any, *, kind: str, a: Any, b: Any) -> None:
            super().__init__()
            self.base = base
            self.kind = kind
            self.scale = loaded.scale
            self.adapter_enabled = True
            self.fused_row_reduce_enabled = False
            self._digitalhuman_h3_inference_lora = True
            self.register_buffer("lora_a", a, persistent=False)
            self.register_buffer("lora_b", b, persistent=False)

        def forward(self, inputs: Any) -> Any:
            if (
                self.adapter_enabled
                and self.kind == "row"
                and self.fused_row_reduce_enabled
                and int(self.base.tp_size) > 1
            ):
                base_partial = self.base.forward_local(inputs)
                compute_dtype = base_partial.dtype
                update_partial = torch.nn.functional.linear(
                    torch.nn.functional.linear(
                        inputs.to(compute_dtype), self.lora_a.to(compute_dtype)
                    ),
                    self.lora_b.to(compute_dtype),
                )
                output = self.base.reduce_local_output(
                    base_partial + update_partial * self.scale
                )
                if self.base.bias is not None:
                    output = output + self.base.bias
                return output
            output = self.base(inputs)
            if not self.adapter_enabled:
                return output
            compute_dtype = output.dtype
            update = torch.nn.functional.linear(
                torch.nn.functional.linear(
                    inputs.to(compute_dtype), self.lora_a.to(compute_dtype)
                ),
                self.lora_b.to(compute_dtype),
            )
            if self.kind == "row" and int(self.base.tp_size) > 1:
                update = collectives.tp_all_reduce(
                    update, context=self.base.parallel_context
                )
            if tuple(update.shape) != tuple(output.shape):
                raise H3LoRACheckpointError("H3 inference LoRA output shape differs")
            return output + update * self.scale

    replacements: list[tuple[str, Any, Any]] = []
    local_numel = 0
    available_targets = [
        target
        for target in targets
        if all(f"{target}.{leaf}" in loaded.state for leaf in ("lora_a", "lora_b"))
    ]
    if not available_targets:
        raise H3LoRACheckpointError("H3 LoRA checkpoint contains no loadable target")
    for target in available_targets:
        module = modules.get(target)
        if type(module) in supported_column:
            if bool(getattr(module, "gather_output", False)):
                raise H3LoRACheckpointError("gathered H3 column output is not qualified")
            kind = "column"
        elif type(module) in supported_row:
            if not bool(getattr(module, "input_is_parallel", False)):
                raise H3LoRACheckpointError("H3 row-parallel topology differs")
            kind = "row"
        else:
            raise H3LoRACheckpointError(f"H3 inference LoRA target type differs: {target}")

        full_a = loaded.state[f"{target}.lora_a"]
        full_b = loaded.state[f"{target}.lora_b"]
        if (
            int(full_a.shape[1]) != int(module.input_size)
            or int(full_b.shape[0]) != int(module.output_size)
        ):
            raise H3LoRACheckpointError(f"H3 inference LoRA full shape differs: {target}")
        device = module.weight.device
        if kind == "column":
            local_a = full_a.to(device=device, dtype=torch.float32)
            local_b = _column_local_b(full_b, module).to(
                device=device, dtype=torch.float32
            )
        else:
            local_a = _row_local_a(full_a, module).to(
                device=device, dtype=torch.float32
            )
            local_b = full_b.to(device=device, dtype=torch.float32)
        if (
            int(local_a.shape[1]) != int(module.weight.shape[1])
            or int(local_b.shape[0]) != int(module.weight.shape[0])
        ):
            raise H3LoRACheckpointError(f"H3 inference LoRA local shape differs: {target}")
        replacement = H3InferenceParallelLinearLoRA(
            module, kind=kind, a=local_a, b=local_b
        ).eval()
        replacements.append((target, module, replacement))
        local_numel += int(local_a.numel() + local_b.numel())

    for target, expected, replacement in replacements:
        parent = transformer
        parts = target.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        if getattr(parent, parts[-1]) is not expected:
            raise H3LoRACheckpointError(f"H3 inference LoRA target changed: {target}")
        setattr(parent, parts[-1], replacement)

    return {
        "schema": "digitalhuman_h3.inference.lora_checkpoint.v1",
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "stage": loaded.stage,
        "optimizer_step": loaded.optimizer_step,
        "role": loaded.role,
        "rank": loaded.rank,
        "alpha": loaded.alpha,
        "scale": loaded.scale,
        "target_count": len(replacements),
        "expected_target_count": len(targets),
        "loaded_tensor_count": loaded.loaded_tensor_count,
        "missing_tensor_count": loaded.missing_tensor_count,
        "redundant_tensor_count": loaded.redundant_tensor_count,
        "local_adapter_numel": local_numel,
        "tensor_parallel_size": int(replacements[0][1].tp_size),
    }


def enable_h3_fused_lora_row_reduce_(transformer: Any) -> Mapping[str, Any]:
    """Fuse each inference row-LoRA update into its base TP reduction."""

    row_modules = [
        (name, module)
        for name, module in transformer.named_modules()
        if getattr(module, "_digitalhuman_h3_inference_lora", False)
        and getattr(module, "kind", None) == "row"
    ]
    if not row_modules:
        raise H3LoRACheckpointError(
            "fused LoRA row reduction requires a loaded inference LoRA adapter"
        )
    for _, module in row_modules:
        module.fused_row_reduce_enabled = True
    main_dit_modules = sum(name.startswith("blocks.") for name, _ in row_modules)
    return {
        "schema": "digitalhuman_h3.inference.fused_lora_row_reduce.v1",
        "row_parallel_module_count": len(row_modules),
        "main_dit_row_parallel_module_count": main_dit_modules,
        "main_dit_tp_all_reduce_calls_saved_per_forward": main_dit_modules,
    }


def materialize_h3_lora_bf16_buffers_(transformer: Any) -> Mapping[str, Any]:
    """Keep inference LoRA A/B buffers in their actual BF16 compute dtype.

    H3 inference projections return BF16 even when the frozen base weight uses
    FP8.  The canonical wrapper therefore converted every FP32 A/B buffer to
    BF16 on every forward.  Materializing the same conversion once preserves
    both LoRA GEMMs and their accumulation order while removing those repeated
    device casts.
    """

    torch = importlib.import_module("torch")
    adapters = [
        (name, module)
        for name, module in transformer.named_modules()
        if getattr(module, "_digitalhuman_h3_inference_lora", False)
    ]
    if not adapters:
        raise H3LoRACheckpointError(
            "BF16 LoRA buffers require a loaded inference LoRA adapter"
        )

    converted_tensor_count = 0
    converted_numel = 0
    for _name, module in adapters:
        for buffer_name in ("lora_a", "lora_b"):
            value = getattr(module, buffer_name)
            if value.dtype == torch.bfloat16:
                continue
            converted_tensor_count += 1
            converted_numel += int(value.numel())
            setattr(
                module,
                buffer_name,
                value.to(device=value.device, dtype=torch.bfloat16),
            )

    return {
        "schema": "digitalhuman_h3.inference.lora_bf16_buffers.v1",
        "target_count": len(adapters),
        "buffer_count": 2 * len(adapters),
        "converted_tensor_count": converted_tensor_count,
        "converted_numel": converted_numel,
        "dtype": "bfloat16",
        "online_lora_gemms_removed_per_forward": 0,
        "arithmetic_order_changed": False,
    }


def merge_h3_lora_weights_(transformer: Any) -> Mapping[str, Any]:
    """Fold a loaded inference LoRA bank into its frozen local base weights.

    The merge is evaluated in FP32 and rounded once to the base weight dtype.
    A later FP8 conversion therefore quantizes the effective Stage3 weight and
    every target projection avoids both online LoRA GEMMs.
    """

    torch = importlib.import_module("torch")
    adapters = [
        (name, module)
        for name, module in transformer.named_modules()
        if getattr(module, "_digitalhuman_h3_inference_lora", False)
    ]
    if not adapters:
        raise H3LoRACheckpointError(
            "LoRA weight merge requires a loaded inference LoRA adapter"
        )

    merged_numel = 0
    released_adapter_numel = 0
    for name, module in adapters:
        if not bool(module.adapter_enabled):
            raise H3LoRACheckpointError(
                f"H3 inference LoRA target is already disabled: {name}"
            )
        base = module.base
        weight = base.weight.detach()
        if weight.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise H3LoRACheckpointError(
                f"H3 inference LoRA base weight cannot be merged: {name}"
            )
        if (
            int(module.lora_a.shape[1]) != int(weight.shape[1])
            or int(module.lora_b.shape[0]) != int(weight.shape[0])
        ):
            raise H3LoRACheckpointError(
                f"H3 inference LoRA local merge shape differs: {name}"
            )

        delta = torch.mm(
            module.lora_b.to(dtype=torch.float32),
            module.lora_a.to(dtype=torch.float32),
        )
        delta.mul_(float(module.scale))
        merged = weight.to(dtype=torch.float32).add_(delta).to(dtype=weight.dtype)
        base.weight = torch.nn.Parameter(merged, requires_grad=False)
        merged_numel += int(merged.numel())
        released_adapter_numel += int(
            module.lora_a.numel() + module.lora_b.numel()
        )
        module.adapter_enabled = False
        module._digitalhuman_h3_lora_merged = True
        del module.lora_a
        del module.lora_b

    return {
        "schema": "digitalhuman_h3.inference.lora_weight_merge.v1",
        "arithmetic": "fp32_delta_and_accumulate_then_round_to_base_dtype",
        "target_count": len(adapters),
        "merged_weight_numel": merged_numel,
        "released_adapter_numel": released_adapter_numel,
        "online_lora_gemms_removed_per_forward": 2 * len(adapters),
        "reversible": False,
    }


@contextmanager
def h3_base_model_inference(transformer: Any):
    """Temporarily bypass LoRA without consuming student FP8 calibration."""

    from digitalhuman_h3.distributed import suspend_fp8_activation_calibration

    adapters = [
        module
        for module in transformer.modules()
        if getattr(module, "_digitalhuman_h3_inference_lora", False)
    ]
    if not adapters:
        raise H3LoRACheckpointError(
            "Base audio teacher requires an inference LoRA checkpoint"
        )
    if any(
        getattr(module, "_digitalhuman_h3_lora_merged", False)
        for module in adapters
    ):
        raise H3LoRACheckpointError(
            "Base audio teacher cannot bypass an already merged LoRA weight"
        )
    previous = [bool(module.adapter_enabled) for module in adapters]
    for module in adapters:
        module.adapter_enabled = False
    try:
        with suspend_fp8_activation_calibration():
            yield
    finally:
        for module, enabled in zip(adapters, previous):
            module.adapter_enabled = enabled


__all__ = [
    "H3_DIFFUSIONNFT_LOCAL_STAGE3_FORMAL_ENV",
    "H3_DIFFUSIONNFT_LOCAL_STAGE3_PHYSICAL_ENV",
    "H3LoRACheckpointError",
    "LoadedH3LoRA",
    "apply_h3_lora_checkpoint",
    "enable_h3_fused_lora_row_reduce_",
    "h3_base_model_inference",
    "load_h3_lora_checkpoint",
    "materialize_h3_lora_bf16_buffers_",
    "merge_h3_lora_weights_",
    "read_h3_adapter_config",
]
