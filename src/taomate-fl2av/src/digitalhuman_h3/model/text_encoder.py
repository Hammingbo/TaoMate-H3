# SPDX-License-Identifier: Apache-2.0
"""TP2 Qwen3-VL layer-50 feature encoder used by MiniMax H3."""

from __future__ import annotations

import importlib
import inspect
import json
import re
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from digitalhuman_h3.distributed import (
    ColumnParallelLinear,
    ParallelContext,
    RowParallelLinear,
    current_parallel_context,
)


SELECTED_LANGUAGE_LAYERS = 50
HIDDEN_SIZE = 5120
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class H3TextEncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class H3TextWeightReceipt:
    text_encoder_root: str
    shard_count: int
    tensor_count: int
    selected_language_layers: int
    tp_world_size: int
    tp_rank: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _transformers_qwen_symbols() -> tuple[Any, Any, Any, Any, Any]:
    try:
        config_mod = importlib.import_module(
            "transformers.models.qwen3_vl.configuration_qwen3_vl"
        )
        model_mod = importlib.import_module(
            "transformers.models.qwen3_vl.modeling_qwen3_vl"
        )
    except ImportError as exc:
        raise H3TextEncoderError(
            "MiniMax H3 requires a Transformers build with Qwen3-VL"
        ) from exc
    return (
        config_mod.Qwen3VLConfig,
        model_mod.Qwen3VLModel,
        model_mod.Qwen3VLVisionModel,
        model_mod.Qwen3VLTextRMSNorm,
        model_mod.Qwen3VLTextRotaryEmbedding,
    )


def _apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    model_mod = importlib.import_module(
        "transformers.models.qwen3_vl.modeling_qwen3_vl"
    )
    return model_mod.apply_rotary_pos_emb(q, k, cos, sin)


def _repeat_kv(value: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return value
    batch, heads, sequence, width = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, width)
        .reshape(batch, heads * repeats, sequence, width)
    )


class _TextAttention(nn.Module):
    def __init__(self, config: Any, layer_index: int, context: ParallelContext) -> None:
        super().__init__()
        if config.num_attention_heads % context.tp_world_size:
            raise ValueError("Qwen3-VL query heads must be divisible by TP")
        if config.num_key_value_heads % context.tp_world_size:
            raise ValueError("Qwen3-VL key/value heads must be divisible by TP")
        self.head_dim = int(config.head_dim)
        self.query_heads = int(config.num_attention_heads) // context.tp_world_size
        self.kv_heads = int(config.num_key_value_heads) // context.tp_world_size
        self.kv_repeats = self.query_heads // self.kv_heads
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_index
        self.q_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        _, _, _, rms_norm, _ = _transformers_qwen_symbols()
        self.q_norm = rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = rms_norm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,
            context=context,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        q = self.q_norm(
            self.q_proj(hidden).view(batch, sequence, self.query_heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_norm(
            self.k_proj(hidden).view(batch, sequence, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(hidden).view(
            batch, sequence, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = _apply_rotary(q, k, cos, sin)
        k = _repeat_kv(k, self.kv_repeats)
        v = _repeat_kv(v, self.kv_repeats)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
            scale=self.scaling,
        )
        return self.o_proj(output.transpose(1, 2).reshape(batch, sequence, -1))


class _TextMLP(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=torch.bfloat16,
            context=context,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=torch.bfloat16,
            context=context,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class _TextLayer(nn.Module):
    def __init__(self, config: Any, layer_index: int, context: ParallelContext) -> None:
        super().__init__()
        _, _, _, rms_norm, _ = _transformers_qwen_symbols()
        self.self_attn = _TextAttention(config, layer_index, context)
        self.mlp = _TextMLP(config, context)
        self.input_layernorm = rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), position_embeddings=position_embeddings
        )
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class _LanguageModel(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            getattr(config, "pad_token_id", None),
            dtype=torch.bfloat16,
        )
        self.layers = nn.ModuleList(
            _TextLayer(config, index, context)
            for index in range(SELECTED_LANGUAGE_LAYERS)
        )
        _, _, _, _, rotary = _transformers_qwen_symbols()
        self.rotary_emb = rotary(config=config)

    @staticmethod
    def _inject_visual(
        hidden: torch.Tensor,
        mask: torch.Tensor,
        visual: torch.Tensor,
    ) -> torch.Tensor:
        selected = hidden[mask]
        if selected.shape != visual.shape:
            raise ValueError(
                "Qwen3-VL DeepStack visual feature shape differs: "
                f"{tuple(visual.shape)} != {tuple(selected.shape)}"
            )
        hidden = hidden.clone()
        hidden[mask] = selected + visual.to(selected)
        return hidden

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        visual_mask: torch.Tensor | None = None,
        deepstack_visual: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, position_embeddings=position_embeddings)
            if (
                visual_mask is not None
                and deepstack_visual is not None
                and index < len(deepstack_visual)
            ):
                hidden = self._inject_visual(
                    hidden, visual_mask, deepstack_visual[index]
                )
        return hidden


class _Backbone(nn.Module):
    def __init__(self, config: Any, context: ParallelContext) -> None:
        super().__init__()
        _, _, vision_model, _, _ = _transformers_qwen_symbols()
        self.language_model = _LanguageModel(config.text_config, context)
        self.visual = vision_model(config.vision_config)


class MiniMaxH3TextEncoder(nn.Module):
    """Qwen3-VL multimodal backbone ending at unnormalized layer 50."""

    def __init__(self, config: Any, *, context: ParallelContext | None = None) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = context or current_parallel_context()
        if int(config.text_config.hidden_size) != HIDDEN_SIZE:
            raise ValueError("MiniMax H3 Qwen3-VL hidden size must be 5120")
        if int(config.text_config.num_hidden_layers) < SELECTED_LANGUAGE_LAYERS:
            raise ValueError("Qwen3-VL checkpoint has fewer than 50 language layers")
        self.model = _Backbone(config, self.parallel_context)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)

    @classmethod
    def allocate(
        cls,
        text_encoder_root: str | Path,
        *,
        device: str | torch.device,
        context: ParallelContext | None = None,
    ) -> "MiniMaxH3TextEncoder":
        root = Path(text_encoder_root).resolve(strict=True)
        config_type, _, _, _, _ = _transformers_qwen_symbols()
        config = config_type.from_pretrained(str(root), local_files_only=True)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device(device):
                return cls(config, context=context)
        finally:
            torch.set_default_dtype(previous_dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _position_ids(
        self,
        input_ids: torch.Tensor,
        *,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_grid_thw is None and video_grid_thw is None:
            return torch.arange(input_ids.shape[1], dtype=torch.long)[None, None].expand(
                3, input_ids.shape[0], -1
            )
        _, helper_type, _, _, _ = _transformers_qwen_symbols()
        helper = types.SimpleNamespace(config=self.config)
        if hasattr(helper_type, "get_vision_position_ids"):
            helper.get_vision_position_ids = types.MethodType(
                helper_type.get_vision_position_ids, helper
            )
        method = helper_type.get_rope_index
        parameters = inspect.signature(method).parameters
        attention_mask = torch.ones_like(input_ids)
        if "mm_token_type_ids" in parameters:
            mm_types = torch.zeros_like(input_ids, dtype=torch.int32)
            mm_types[input_ids == self.image_token_id] = 1
            mm_types[input_ids == self.video_token_id] = 2
            position_ids, _ = method(
                helper,
                input_ids,
                mm_types,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            position_ids, _ = method(
                helper,
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )
        return position_ids

    @torch.no_grad()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be supplied together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be supplied together"
            )
        host_ids = input_ids.to(device="cpu", dtype=torch.long)[None]
        host_image_grid = (
            image_grid_thw.to(device="cpu", dtype=torch.long)
            if image_grid_thw is not None
            else None
        )
        host_video_grid = (
            video_grid_thw.to(device="cpu", dtype=torch.long)
            if video_grid_thw is not None
            else None
        )
        position_ids = self._position_ids(
            host_ids,
            image_grid_thw=host_image_grid,
            video_grid_thw=host_video_grid,
        ).to(self.device)
        ids = host_ids.to(self.device)
        embeddings = self.model.language_model.embed_tokens(ids)
        visual_mask: torch.Tensor | None = None
        deepstack: list[torch.Tensor] | None = None

        if pixel_values is not None:
            grid = host_image_grid.to(self.device)
            visual, deepstack_features = self.model.visual(
                pixel_values.to(self.device, dtype=self.model.visual.dtype),
                grid_thw=grid,
            )
            visual = visual.to(embeddings)
            visual_mask = ids == self.image_token_id
            if int(visual_mask.sum()) != int(visual.shape[0]):
                raise ValueError("Qwen3-VL image token count differs from visual features")
            embeddings = embeddings.clone()
            embeddings[visual_mask] = visual
            deepstack = [item.to(embeddings) for item in deepstack_features]
        elif pixel_values_videos is not None:
            grid = host_video_grid.to(self.device)
            visual, deepstack_features = self.model.visual(
                pixel_values_videos.to(self.device, dtype=self.model.visual.dtype),
                grid_thw=grid,
            )
            visual = visual.to(embeddings)
            visual_mask = ids == self.video_token_id
            if int(visual_mask.sum()) != int(visual.shape[0]):
                raise ValueError("Qwen3-VL video token count differs from visual features")
            embeddings = embeddings.clone()
            embeddings[visual_mask] = visual
            deepstack = [item.to(embeddings) for item in deepstack_features]

        hidden = self.model.language_model(
            inputs_embeds=embeddings,
            position_ids=position_ids,
            visual_mask=visual_mask,
            deepstack_visual=deepstack,
        )[0]
        if tuple(hidden.shape) != (int(ids.shape[1]), HIDDEN_SIZE):
            raise ValueError(f"unexpected Qwen3-VL layer-50 shape {tuple(hidden.shape)}")
        return hidden.to(torch.bfloat16)


def _index(root: Path) -> Mapping[str, str]:
    path = root / "model.safetensors.index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3TextEncoderError(f"cannot read Qwen3-VL weight index: {exc}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise H3TextEncoderError("Qwen3-VL weight index has no weight_map")
    return dict(weight_map)


def _selected_weight(name: str) -> bool:
    if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
        return False
    match = _LAYER_RE.match(name)
    return not (match and int(match.group(1)) >= SELECTED_LANGUAGE_LAYERS)


def load_minimax_h3_text_weights(
    model: MiniMaxH3TextEncoder,
    text_encoder_root: str | Path,
) -> H3TextWeightReceipt:
    root = Path(text_encoder_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise H3TextEncoderError("Qwen3-VL weight root must be a real directory")
    full_map = _index(root)
    weight_map = {name: file for name, file in full_map.items() if _selected_weight(name)}
    state = model.state_dict(keep_vars=True)
    expected, present = set(state), set(weight_map)
    if expected != present:
        raise H3TextEncoderError(
            "Qwen3-VL selected tensor inventory differs: "
            f"missing={sorted(expected - present)[:8]}, "
            f"unexpected={sorted(present - expected)[:8]}"
        )
    modules = dict(model.named_modules())
    shards: dict[Path, list[str]] = {}
    for name, filename in weight_map.items():
        shard = (root / filename).resolve(strict=False)
        if shard.parent != root or shard.name != filename or not shard.is_file() or shard.is_symlink():
            raise H3TextEncoderError(f"unsafe Qwen3-VL shard path: {filename}")
        shards.setdefault(shard, []).append(name)
    safe_open = importlib.import_module("safetensors").safe_open
    with torch.no_grad():
        for shard, names in sorted(shards.items(), key=lambda item: item[0].name):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                for name in sorted(names):
                    value = handle.get_tensor(name)
                    owner_name, _, leaf = name.rpartition(".")
                    owner = modules.get(owner_name)
                    if owner is None:
                        raise H3TextEncoderError(f"Qwen3-VL state owner is absent: {name}")
                    if isinstance(owner, ColumnParallelLinear):
                        value = (
                            owner.full_weight_slice(value)
                            if leaf == "weight"
                            else owner.full_bias_slice(value)
                        )
                    elif isinstance(owner, RowParallelLinear) and leaf == "weight":
                        value = owner.full_weight_slice(value)
                    destination = state[name]
                    if tuple(value.shape) != tuple(destination.shape):
                        raise H3TextEncoderError(
                            f"Qwen3-VL local tensor shape differs for {name}: "
                            f"{tuple(value.shape)} != {tuple(destination.shape)}"
                        )
                    destination.copy_(
                        value.to(device=destination.device, dtype=destination.dtype)
                    )
    model.eval()
    return H3TextWeightReceipt(
        text_encoder_root=str(root),
        shard_count=len(shards),
        tensor_count=len(state),
        selected_language_layers=SELECTED_LANGUAGE_LAYERS,
        tp_world_size=model.parallel_context.tp_world_size,
        tp_rank=model.parallel_context.tp_rank,
    )


__all__ = [
    "H3TextEncoderError",
    "H3TextWeightReceipt",
    "MiniMaxH3TextEncoder",
    "SELECTED_LANGUAGE_LAYERS",
    "load_minimax_h3_text_weights",
]
