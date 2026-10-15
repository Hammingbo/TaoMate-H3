# SPDX-License-Identifier: Apache-2.0
"""Native in-process MiniMax H3 audio-video DiT."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from digitalhuman_h3.distributed import (
    ColumnParallelLinear,
    ParallelContext,
    current_parallel_context,
    tp_all_gather,
    ulysses_all_gather_rows,
)

from .architecture import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT,
    MiniMaxH3Architecture,
)
from .layers import (
    BF16,
    FP32,
    MiniMaxH3DiTBlock,
    MiniMaxH3FinalLayer,
    MiniMaxH3Rope,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenRefiner,
    rope_cos_sin_cache,
)


MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})

_FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "rope_cache",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "block_token_tags",
        "block_combined_indices",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "local_embedding_layout",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "streaming_cache_only",
        "skip_audio_output_head",
    }
)


def _required(kwargs: dict[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiT.forward requires kwarg {key!r}")
    return kwargs[key]


class MiniMaxH3DiT(nn.Module):
    """Checkpoint-key-compatible MiniMax H3 DiT with native PyTorch kernels."""

    def __init__(
        self,
        architecture: MiniMaxH3Architecture | None = None,
        *,
        parallel_context: ParallelContext | None = None,
        load_time_w8a8_policy: str | None = None,
    ) -> None:
        super().__init__()
        self.arch = architecture or MiniMaxH3Architecture()
        self.parallel_context = parallel_context or current_parallel_context()
        self.hidden_size = self.arch.hidden_size
        self.num_attention_heads = self.arch.num_attention_heads
        self.num_channels_latents = self.arch.video_latent_channels
        self._validate_topology()
        context = self.parallel_context
        self.video_patch_proj = ColumnParallelLinear(
            self.arch.video_row_width,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=FP32,
            context=context,
        )
        self.audio_patch_proj = ColumnParallelLinear(
            self.arch.audio_latent_channels,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=FP32,
            context=context,
        )
        self.condition_proj = ColumnParallelLinear(
            self.arch.text_dim,
            self.arch.hidden_size,
            bias=True,
            gather_output=True,
            dtype=BF16,
            context=context,
        )
        self.time_embedder = MiniMaxH3TimeEmbedder(self.arch, context=context)
        self.rope = MiniMaxH3Rope(self.arch.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(self.arch, context=context)
        load_time_w8a8_names: frozenset[str] = frozenset()
        if load_time_w8a8_policy is not None:
            from digitalhuman_h3.inference.w8a8 import (
                require_vllm_cutlass_w8a8,
                resolve_h3_cutlass_w8a8_policy,
            )

            require_vllm_cutlass_w8a8()
            candidate_names = [
                f"blocks.{index}.{projection}"
                for index in range(self.arch.num_layers)
                for projection in (
                    "attn.qkv_proj",
                    "mlp.fc1",
                )
            ]
            selection = resolve_h3_cutlass_w8a8_policy(
                candidate_names,
                policy=load_time_w8a8_policy,
            )
            load_time_w8a8_names = frozenset(
                selection["quantized_linear_names"]
            )

        self.blocks = nn.ModuleList()
        for index in range(self.arch.num_layers):
            block = MiniMaxH3DiTBlock(
                self.arch,
                context=context,
                load_time_w8a8_names=load_time_w8a8_names,
                module_prefix=f"blocks.{index}",
            )
            block.attn.layer_name = f"blocks.{index}"
            self.blocks.append(block)
        self._load_time_w8a8_policy = load_time_w8a8_policy
        self.layer_names = ["blocks"]
        self.final_layer = MiniMaxH3FinalLayer(self.arch, context=context)
        self._inference_static_timestep_cache_enabled = False
        self._inference_static_adaln_input_cache: dict[int, torch.Tensor] = {}
        self._inference_static_adaln_input_hits = 0
        self._inference_static_adaln_input_misses = 0

    def enable_inference_static_timestep_cache(self) -> None:
        """Reuse exact timestep/AdaLN outputs for repeated inference schedules."""

        if self.training:
            raise RuntimeError("static timestep cache is inference-only")
        self._inference_static_timestep_cache_enabled = True
        self._inference_static_adaln_input_cache = {}
        self._inference_static_adaln_input_hits = 0
        self._inference_static_adaln_input_misses = 0
        self.time_embedder.enable_inference_static_cache()
        for block in self.blocks:
            block.adaln_proj.enable_inference_static_cache()
        self.final_layer.adaln_proj.enable_inference_static_cache()

    def inference_static_timestep_cache_stats(self) -> dict[str, int]:
        time_hits, time_misses = self.time_embedder.inference_static_cache_stats()
        adaln_hits = 0
        adaln_misses = 0
        for module in [
            *(block.adaln_proj for block in self.blocks),
            self.final_layer.adaln_proj,
        ]:
            hits, misses = module.inference_static_cache_stats()
            adaln_hits += hits
            adaln_misses += misses
        return {
            "time_embedding_hits": time_hits,
            "time_embedding_misses": time_misses,
            "adaln_input_hits": self._inference_static_adaln_input_hits,
            "adaln_input_misses": self._inference_static_adaln_input_misses,
            "adaln_projection_hits": adaln_hits,
            "adaln_projection_misses": adaln_misses,
        }

    @classmethod
    def allocate(
        cls,
        device: str | torch.device,
        architecture: MiniMaxH3Architecture | None = None,
        *,
        parallel_context: ParallelContext | None = None,
        load_time_w8a8_policy: str | None = None,
    ) -> "MiniMaxH3DiT":
        """Allocate TP-local parameter storage directly on the target device."""

        with torch.device(device):
            return cls(
                architecture,
                parallel_context=parallel_context,
                load_time_w8a8_policy=load_time_w8a8_policy,
            )

    def _validate_topology(self) -> None:
        context = self.parallel_context
        if context.ring_world_size != 1:
            raise NotImplementedError("MiniMax H3 supports Ulysses but not Ring parallelism")
        for name, value in (
            ("num_attention_heads", self.arch.num_attention_heads),
            ("hidden_size", self.arch.hidden_size),
            ("ffn_hidden_size", self.arch.ffn_hidden_size),
            ("time_embed_hidden_size", self.arch.time_embed_hidden_size),
            ("adaln_out_features", self.arch.adaln_out_features),
            ("final_adaln_out_features", self.arch.final_adaln_out_features),
            ("video_row_width", self.arch.video_row_width),
            ("audio_latent_channels", self.arch.audio_latent_channels),
        ):
            if value % context.tp_world_size:
                raise ValueError(f"MiniMax H3 {name}={value} is not divisible by TP")
        # Ulysses pads uneven head shards and removes padding before projection.
        if MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT % context.ulysses_world_size:
            raise ValueError("H3 packed alignment is not divisible by Ulysses")

    @staticmethod
    def _position_ids(value: Any, key: str) -> torch.Tensor:
        ids = value.get("position_ids") if isinstance(value, dict) else getattr(value, "position_ids", None)
        if ids is None:
            raise ValueError(f"{key}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _packed_field(value: Any, key: str, field: str) -> Any:
        result = value.get(field) if isinstance(value, dict) else getattr(value, field, None)
        if result is None:
            raise ValueError(f"{key}.{field} is required")
        return result

    @staticmethod
    def _packed_optional(value: Any, field: str) -> Any:
        return value.get(field) if isinstance(value, dict) else getattr(value, field, None)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Refine prompt rows as a public FSDP alternate-forward entry."""

        return self._refine_prompt_embeds_impl(
            prompt_embeds,
            refiner_cu_seqlens,
            device=device,
        )

    def _refine_prompt_embeds_impl(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        bounds = tuple(
            int(value)
            for value in refiner_cu_seqlens.detach().cpu().view(-1).tolist()
        )
        lengths = tuple(
            stop - start for start, stop in zip(bounds[:-1], bounds[1:])
        )
        text_len = bounds[-1] if bounds else 0
        if (
            len(bounds) < 2
            or bounds[0] != 0
            or any(length < 0 for length in lengths)
            or text_len <= 0
            or text_len > int(prompt_embeds.shape[0])
        ):
            raise ValueError("refiner live text length differs")
        text_rows = prompt_embeds[:text_len].to(device=device, dtype=BF16)
        return self.token_refiner(
            self.condition_proj(text_rows),
            cu_seqlens=refiner_cu_seqlens,
            cu_seqlens_host=bounds,
            max_seqlen=max(lengths),
        )

    def build_rope_cache(
        self,
        img_position_ids: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError("img_position_ids must be [1, S, 3]")
        sequence_length = int(img_position_ids.shape[1])
        world_size = self.parallel_context.ulysses_world_size
        rank = self.parallel_context.ulysses_rank
        if sequence_length % world_size:
            raise ValueError("packed sequence is not divisible by Ulysses")
        local_length = sequence_length // world_size
        row_start = rank * local_length
        frequencies = self.rope(
            img_position_ids[:, row_start : row_start + local_length]
        ).to(device)
        return (
            rope_cos_sin_cache(frequencies, dtype=BF16),
            torch.arange(local_length, device=device, dtype=torch.long),
        )

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embeddings_selected: torch.Tensor,
        unique_timesteps: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        row_start: int,
        row_stop: int,
        device: torch.device,
        refined_prompt_embeds_length: int | torch.Tensor | None,
        local_embedding_layout: dict[str, torch.Tensor | int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if refined_prompt_embeds_length is None:
            text_len = int(refiner_cu_seqlens[-1].item())
        elif torch.is_tensor(refined_prompt_embeds_length):
            text_len = int(refined_prompt_embeds_length.item())
        else:
            text_len = int(refined_prompt_embeds_length)
        if text_len <= 0 or text_len > int(text_embeddings_selected.shape[0]):
            raise ValueError("refiner live text length differs")
        text_pos = text_pos[:text_len]
        if refined_prompt_embeds_length is None:
            # ``refine_prompt_embeds`` is registered as an alternate FSDP root
            # forward for standalone pipeline refinement. Calling that wrapped
            # method from inside the root forward would re-enter root hooks.
            text_embed = self._refine_prompt_embeds_impl(
                text_embeddings_selected, refiner_cu_seqlens, device=device
            )
        else:
            text_embed = text_embeddings_selected[:text_len].to(device=device, dtype=BF16)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError("refined prompt embedding width differs")

        local_seq_len = row_stop - row_start
        trusted_layout = local_embedding_layout is not None
        if trusted_layout:
            used_len = text_len + int(img_pos.numel()) + int(audio_pos.numel())
            local_live_rows = min(max(used_len - row_start, 0), local_seq_len)
            embeddings = torch.empty(
                (local_seq_len, self.hidden_size), device=device, dtype=BF16
            )
            if local_live_rows < local_seq_len:
                embeddings[local_live_rows:].zero_()
            text_source_start = int(local_embedding_layout["text_source_start"])
            text_source_stop = int(local_embedding_layout["text_source_stop"])
            img_global_ids = local_embedding_layout["img_global_ids"]
            img_row_ids = local_embedding_layout["img_row_ids"]
            audio_global_ids = local_embedding_layout["audio_global_ids"]
            audio_row_ids = local_embedding_layout["audio_row_ids"]
            text_rows = text_source_stop - text_source_start
            if text_rows:
                embeddings[:text_rows].copy_(text_embed[text_source_start:text_source_stop])
            write_rows = embeddings.index_copy_
        else:
            embeddings = torch.zeros(
                (local_seq_len, self.hidden_size), device=device, dtype=BF16
            )
            text_source_ids = torch.nonzero(
                (text_pos >= row_start) & (text_pos < row_stop), as_tuple=False
            ).view(-1)
            text_row_ids = text_pos.index_select(0, text_source_ids) - row_start
            img_source_ids = torch.nonzero(
                (img_pos >= row_start) & (img_pos < row_stop), as_tuple=False
            ).view(-1)
            img_global_ids = img_pos.index_select(0, img_source_ids)
            img_row_ids = img_global_ids - row_start
            audio_source_ids = torch.nonzero(
                (audio_pos >= row_start) & (audio_pos < row_stop), as_tuple=False
            ).view(-1)
            audio_global_ids = audio_pos.index_select(0, audio_source_ids)
            audio_row_ids = audio_global_ids - row_start
            if text_row_ids.numel():
                embeddings.index_add_(
                    0,
                    text_row_ids,
                    text_embed.index_select(0, text_source_ids).to(BF16),
                )
            write_rows = embeddings.index_add_

        # FSDP2 manages both patch projections as separate parameter groups.
        # Every rank must therefore invoke them in the same order even when a
        # Ulysses shard contains no rows for one modality; otherwise ranks enter
        # different parameter all-gathers and deadlock.  Empty linear inputs
        # preserve the collective order without adding packed-sequence rows.
        # Empty shards still need the input derivative when differentiating
        # earlier denoise actions or clean KV. A new_empty tensor drops that
        # edge and makes TP2-shaped paired gathers disagree about backward.
        video_rows = x.view(-1, x.shape[-1]).index_select(0, img_global_ids).to(FP32)
        if int(img_pos.numel()) == 0 and not torch.is_grad_enabled():
            video_embeddings = x.new_empty((0, self.hidden_size), dtype=BF16)
        else:
            video_embeddings = self.video_patch_proj(video_rows).to(BF16)
        if img_row_ids.numel():
            write_rows(0, img_row_ids, video_embeddings)

        audio_rows = audio_x.view(-1, audio_x.shape[-1]).index_select(0, audio_global_ids).to(FP32)
        audio_embeddings = self.audio_patch_proj(audio_rows).to(BF16)
        if audio_row_ids.numel():
            write_rows(0, audio_row_ids, audio_embeddings)
        # Empty modality shards must also retain both projections in the
        # autograd graph.  Otherwise FSDP executes a projection's backward
        # collectives only on ranks whose Ulysses slice contains that modality.
        projection_dependency = (
            text_embed.sum()
            + video_embeddings.sum()
            + audio_embeddings.sum()
        ) * 0.0
        return (
            embeddings + projection_dependency.to(dtype=embeddings.dtype),
            self.time_embedder(unique_timesteps),
        )

    def forward(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        unexpected = sorted(set(kwargs) - _FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(f"MiniMaxH3DiT received unexpected kwargs: {unexpected}")
        streaming_cache_only = kwargs.get("streaming_cache_only", False)
        if type(streaming_cache_only) is not bool:
            raise TypeError("streaming_cache_only must be a boolean")
        skip_audio_output_head = kwargs.get("skip_audio_output_head", False)
        if type(skip_audio_output_head) is not bool:
            raise TypeError("skip_audio_output_head must be a boolean")
        x = _required(kwargs, "x")
        audio_x = _required(kwargs, "audio_x")
        img_position_ids = _required(kwargs, "img_position_ids")
        unique_timesteps = _required(kwargs, "unique_timesteps")
        inverse_indices = _required(kwargs, "inverse_indices").view(-1).to(torch.long)
        update_mask = _required(kwargs, "update_mask")
        block_token_tags = kwargs.get("block_token_tags")
        token_tags = kwargs.get("token_tags")
        if block_token_tags is None:
            token_tags = _required(kwargs, "token_tags").view(-1).to(torch.long)
        else:
            block_token_tags = block_token_tags.view(-1).to(torch.long)
            token_tags = None
        prompt_embeds = _required(kwargs, "prompt_embeds")
        img_pos = self._position_ids(_required(kwargs, "img_pos_info"), "img_pos_info")
        audio_pos = self._position_ids(
            _required(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_pos = self._position_ids(
            _required(kwargs, "text_pos_info"), "text_pos_info"
        )
        infer_out_pos = self._position_ids(
            _required(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )
        packed = _required(kwargs, "packed_seq_params")
        cu_seqlens = self._packed_field(
            packed, "packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        host = self._packed_optional(packed, "cu_seqlens_q_host")
        cu_seqlens_host = tuple(
            int(value) for value in (cu_seqlens.tolist() if host is None else host)
        )
        max_seqlen = int(
            self._packed_field(packed, "packed_seq_params", "max_seqlen_q")
        )
        refiner = _required(kwargs, "refiner_packed_seq_params")
        refiner_cu = self._packed_field(
            refiner, "refiner_packed_seq_params", "cu_seqlens_q"
        ).to(torch.int32)
        if x.dim() != 3 or x.shape[0] != 1:
            raise ValueError("x must be [1, S, C]")
        seq_len = int(x.shape[1])
        if token_tags is not None and int(token_tags.shape[0]) != seq_len:
            raise ValueError("token_tags must cover the packed sequence")
        if int(inverse_indices.shape[0]) != seq_len:
            raise ValueError("inverse_indices must cover the packed sequence")
        context = self.parallel_context
        if seq_len % context.ulysses_world_size:
            raise ValueError("packed sequence is not divisible by Ulysses")
        local_seq_len = seq_len // context.ulysses_world_size
        row_start = context.ulysses_rank * local_seq_len
        row_stop = row_start + local_seq_len
        device = x.device
        rope_cache = kwargs.get("rope_cache") or self.build_rope_cache(
            img_position_ids, device=device
        )
        img_pos = img_pos.to(device)
        audio_pos = audio_pos.to(device)
        text_pos = text_pos.to(device)
        hidden, timestep_embedding = self._embed(
            x=x,
            audio_x=audio_x,
            text_embeddings_selected=prompt_embeds,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            refiner_cu_seqlens=refiner_cu.to(device),
            row_start=row_start,
            row_stop=row_stop,
            device=device,
            refined_prompt_embeds_length=kwargs.get("refined_prompt_embeds_length"),
            local_embedding_layout=kwargs.get("local_embedding_layout"),
        )
        if self._inference_static_timestep_cache_enabled:
            adaln_key = id(timestep_embedding)
            adaln_input = self._inference_static_adaln_input_cache.get(adaln_key)
            if adaln_input is None:
                adaln_input = F.silu(timestep_embedding).to(BF16)
                self._inference_static_adaln_input_cache[adaln_key] = adaln_input
                self._inference_static_adaln_input_misses += 1
            else:
                self._inference_static_adaln_input_hits += 1
        else:
            adaln_input = F.silu(timestep_embedding).to(BF16)
        inverse_indices = inverse_indices.to(device)
        block_inverse = inverse_indices[row_start:row_stop]
        if block_token_tags is None:
            assert token_tags is not None
            block_token_tags = token_tags.to(device)[row_start:row_stop].clamp(min=0)
        else:
            block_token_tags = block_token_tags.to(device)
            if int(block_token_tags.shape[0]) != local_seq_len:
                raise ValueError("block_token_tags must cover the local sequence")
        combined = kwargs.get("block_combined_indices")
        if combined is None:
            combined = torch.add(
                block_token_tags,
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
        cu_seqlens = cu_seqlens.to(device)
        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
                ulysses_active=context.ulysses_world_size > 1,
            )
            if isinstance(hidden, tuple):
                # Joint clean-KV capture is an explicit output of each FSDP
                # block, so its backward unshards parameters before replay.
                hidden, _, _ = hidden
        # Full-prefix capture and sigma=0 clean commit consume only the K/V
        # staged by the 50 attention hooks above.  Their callers explicitly set
        # this private flag so the unused final logits are never constructed or
        # gathered; ordinary denoise forwards retain the original path below.
        if streaming_cache_only:
            return None
        if int(infer_out_pos.numel()) == 0:
            if skip_audio_output_head:
                raise ValueError(
                    "skip_audio_output_head requires video output rows"
                )
            audio_logits = self.final_layer.forward_audio_only(
                hidden, adaln_input=adaln_input, inverse_indices=block_inverse
            )
            if context.ulysses_world_size > 1:
                audio_logits = ulysses_all_gather_rows(
                    audio_logits, context=context
                )
            audio_logits = audio_logits.index_select(0, audio_pos.to(device))
            audio_logits = tp_all_gather(audio_logits, context=context)
            video_logits = audio_logits.new_empty((0, self.arch.video_row_width))
            return video_logits, audio_logits
        if skip_audio_output_head:
            video_logits = self.final_layer.forward_video_only(
                hidden, adaln_input=adaln_input, inverse_indices=block_inverse
            )
            audio_logits = None
        else:
            video_logits, audio_logits = self.final_layer(
                hidden, adaln_input=adaln_input, inverse_indices=block_inverse
            )
        if context.ulysses_world_size > 1:
            if audio_logits is None:
                video_logits = ulysses_all_gather_rows(
                    video_logits, context=context
                )
            else:
                video_width = video_logits.shape[-1]
                logits = ulysses_all_gather_rows(
                    torch.cat((video_logits, audio_logits), dim=-1), context=context
                )
                video_logits, audio_logits = logits.split(
                    (video_width, logits.shape[-1] - video_width), dim=-1
                )
        video_logits = video_logits.index_select(0, infer_out_pos.to(device))
        video_logits = tp_all_gather(video_logits, context=context)
        if audio_logits is None:
            audio_logits = torch.zeros(
                (int(audio_pos.numel()), self.arch.audio_latent_channels),
                device=device,
                dtype=video_logits.dtype,
            )
        else:
            audio_logits = audio_logits.index_select(0, audio_pos.to(device))
            audio_logits = tp_all_gather(audio_logits, context=context)
        if not bool(kwargs.get("skip_mask_out_condition", False)):
            update_mask = update_mask.view(-1).to(device)
            if int(update_mask.shape[0]) != int(video_logits.shape[0]):
                raise ValueError("update_mask length differs")
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)
        return video_logits, audio_logits


__all__ = [
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiT",
]
