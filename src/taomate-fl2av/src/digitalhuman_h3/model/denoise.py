# SPDX-License-Identifier: Apache-2.0
"""Native MiniMax H3 cfg-distilled audio-video denoise loop.

The released algorithm is implemented directly; framework-specific distributed
state is replaced by an explicit local
``ParallelContext`` owned by Digitalhuman H3.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import torch

from digitalhuman_h3.distributed import ParallelContext, current_parallel_context

from .architecture import MINIMAX_H3_ADALN_MODALITY_NUM


MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
MINIMAX_H3_VIDEO_ROW_WIDTH = 96
MINIMAX_H3_AUDIO_ROW_WIDTH = 32


@torch.no_grad()
def _minimax_h3_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_t: torch.Tensor,
    sigma_curr: float,
    sigma_ratio: torch.Tensor,
    one_minus_sigma_ratio: torch.Tensor,
    denoised_scratch: torch.Tensor,
) -> None:
    torch.mul(sigma_t, velocity, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_curr == 0.0:
        return
    torch.mul(one_minus_sigma_ratio, denoised_scratch, out=velocity)
    torch.mul(sigma_ratio, state, out=state)
    torch.add(state, velocity, out=state)


@torch.no_grad()
def _minimax_h3_consistency_update_target_rows_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
    generator: torch.Generator,
    denoised_scratch: torch.Tensor,
) -> None:
    """Predict x0, then use fresh noise for the next consistency call."""

    torch.mul(velocity, sigma_curr, out=denoised_scratch)
    torch.add(state, denoised_scratch, out=denoised_scratch)
    if sigma_next == 0.0:
        state.copy_(denoised_scratch)
        return
    fresh_noise = torch.randn(
        state.shape,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    ).to(device=state.device)
    state.copy_(denoised_scratch, non_blocking=False)
    state.lerp_(fresh_noise, sigma_next)


def _build_local_embedding_layout(
    *,
    seq_len: int,
    text_pos: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if seq_len % world_size:
        raise ValueError(
            f"packed seq_len {seq_len} not divisible by Ulysses world size "
            f"{world_size}"
        )
    local_seq_len = seq_len // world_size
    row_start = rank * local_seq_len
    row_stop = row_start + local_seq_len

    def local_ids(pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_ids = torch.nonzero(
            (pos >= row_start) & (pos < row_stop),
            as_tuple=False,
        ).view(-1)
        return source_ids.to(device), pos.index_select(0, source_ids).to(device)

    text_source_start = min(row_start, int(text_pos.shape[0]))
    text_source_stop = min(row_stop, int(text_pos.shape[0]))
    _, img_global_ids = local_ids(img_pos)
    _, audio_global_ids = local_ids(audio_pos)
    return {
        "text_source_start": text_source_start,
        "text_source_stop": text_source_stop,
        "img_global_ids": img_global_ids,
        "img_row_ids": img_global_ids - row_start,
        "audio_global_ids": audio_global_ids,
        "audio_row_ids": audio_global_ids - row_start,
    }


class MiniMaxH3DenoiseBranch:
    """Static positive-branch packed state and step-local forward inputs."""

    def __init__(
        self,
        *,
        packed: dict[str, Any],
        text_embeddings: torch.Tensor,
        token_tags: torch.Tensor,
        device: torch.device,
        parallel_context: ParallelContext | None = None,
    ) -> None:
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(torch.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        self.update_mask = packed["update_mask"].view(-1).to(torch.bool)
        if "audio_update_mask" in packed:
            self.audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            self.audio_update_mask = torch.ones(
                self.audio_pos.shape[0], dtype=torch.bool
            )
        self.video_target_start = int((~self.update_mask).sum())
        self.audio_target_start = int((~self.audio_update_mask).sum())
        self.video_target_slice = slice(self.video_target_start, None)
        self.audio_target_slice = slice(self.audio_target_start, None)
        text_pos = packed["text_pos"].view(-1).to(torch.long)
        text_len = int(text_pos.shape[0])
        if int(text_embeddings.shape[0]) != text_len:
            raise ValueError(
                f"text_embeddings rows {list(text_embeddings.shape)} != "
                f"packed text_len {text_len}"
            )
        if int(token_tags.view(-1).shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length {int(token_tags.view(-1).shape[0])} != "
                f"seq_len {seq_len}"
            )
        cu = packed["cu_seqlens"].to(torch.int32)
        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        # Keep the packed host descriptors beside their device views.  The
        # opt-in HiAR fused workspace can compose a new branch without first
        # synchronizing these immutable tensors back from CUDA.
        self.text_pos = text_pos
        self.img_position_ids_host = (
            packed["img_position_ids"].detach().to(dtype=torch.float32, device="cpu")
        )
        self.img_cond_seq_idx = self.img_pos_dev[~self.update_mask_dev]
        self.img_target_seq_idx = self.img_pos_dev[self.update_mask_dev]
        self.audio_target_seq_idx = self.audio_pos_dev[self.audio_update_mask_dev]
        self.audio_ref_seq_idx = self.audio_pos_dev[~self.audio_update_mask_dev]
        self.cond_row_idx = torch.nonzero(~self.update_mask_dev).view(-1)
        self.audio_ref_row_idx = torch.nonzero(~self.audio_update_mask_dev).view(-1)
        self.n_video_timestep_rows = (
            seq_len
            - int(self.img_cond_seq_idx.numel())
            - int(self.audio_target_seq_idx.numel())
            - int(self.audio_ref_seq_idx.numel())
        )
        self._x_buffer_primed = False
        self._inference_static_runtime_plans_enabled = False
        self._inference_timestep_plan_cache: dict[
            tuple[tuple[float, ...], tuple[float, ...], float, float],
            list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ] = {}
        self._inference_static_runtime_plan_hits = 0
        self._inference_static_runtime_plan_misses = 0
        self._inference_native_attention_metadata_cache: dict[
            tuple[tuple[int, ...] | None, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._inference_native_attention_metadata_hits = 0
        self._inference_native_attention_metadata_misses = 0
        self.x_buffer = torch.zeros(
            1,
            seq_len,
            MINIMAX_H3_VIDEO_ROW_WIDTH,
            dtype=torch.float32,
            device=device,
        )
        self.audio_x_buffer = torch.zeros(
            1,
            seq_len,
            MINIMAX_H3_AUDIO_ROW_WIDTH,
            dtype=torch.float32,
            device=device,
        )
        text_pos_dev = text_pos.to(device)
        context = parallel_context or current_parallel_context()
        self.parallel_context = context
        ulysses_world_size = context.ulysses_world_size
        ulysses_rank = context.ulysses_rank
        if seq_len % ulysses_world_size:
            raise ValueError(
                f"packed seq_len {seq_len} not divisible by Ulysses world size "
                f"{ulysses_world_size}"
            )
        token_tags_host = token_tags.view(-1).to(dtype=torch.long)
        # Fused HiAR branches need the complete packed tag vector before it is
        # sliced into Ulysses-local rows.  Keeping this host tensor does not
        # change the ordinary one-document forward path.
        self.token_tags = token_tags_host
        local_seq_len = seq_len // ulysses_world_size
        local_row_start = ulysses_rank * local_seq_len
        local_row_stop = local_row_start + local_seq_len
        self.local_row_slice = slice(local_row_start, local_row_stop)
        self.block_token_tags = (
            token_tags_host[local_row_start:local_row_stop].clamp(min=0).to(device)
        )
        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": packed["img_position_ids"][None]
            .to(torch.float32)
            .to(device),
            "update_mask": self.update_mask_dev,
            "block_token_tags": self.block_token_tags,
            "skip_mask_out_condition": True,
            "prompt_embeds": text_embeddings.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {
                "position_ids": self.img_target_seq_idx
            },
            "local_embedding_layout": _build_local_embedding_layout(
                seq_len=seq_len,
                text_pos=text_pos,
                img_pos=self.img_pos,
                audio_pos=self.audio_pos,
                world_size=ulysses_world_size,
                rank=ulysses_rank,
                device=device,
            ),
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "cu_seqlens_q_host": tuple(int(value) for value in cu.tolist()),
                "max_seqlen_q": int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor(
                    [0, text_len, text_len], dtype=torch.int32, device=device
                ),
                "cu_seqlens_q_host": (0, text_len, text_len),
                "max_seqlen_q": text_len,
            },
        }

    def enable_inference_static_runtime_plans(self) -> None:
        """Reuse immutable inference descriptors for an unchanged packed layout."""

        self._inference_static_runtime_plans_enabled = True
        self._inference_timestep_plan_cache = {}
        self._inference_static_runtime_plan_hits = 0
        self._inference_static_runtime_plan_misses = 0
        self._inference_native_attention_metadata_cache = {}
        self._inference_native_attention_metadata_hits = 0
        self._inference_native_attention_metadata_misses = 0

    def inference_static_runtime_plan_stats(self) -> dict[str, int]:
        return {
            "timestep_plan_hits": self._inference_static_runtime_plan_hits,
            "timestep_plan_misses": self._inference_static_runtime_plan_misses,
            "attention_metadata_hits": (
                self._inference_native_attention_metadata_hits
            ),
            "attention_metadata_misses": (
                self._inference_native_attention_metadata_misses
            ),
        }

    def forward_kwargs(
        self,
        *,
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        step_timesteps: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        refresh_all_media_rows: bool = False,
    ) -> dict[str, Any]:
        differentiable = torch.is_grad_enabled() and (
            video_rows.requires_grad or audio_rows.requires_grad
        )
        if differentiable:
            # The persistent buffers are an inference optimization.  Writing a
            # differentiable state into them makes the buffer retain a
            # CopySlices graph; a second Stage3 replay would then traverse an
            # already-freed graph.  Build a fresh local state for every train
            # forward and copy every media row so repeated low-resolution
            # transitions remain independent.
            x = self.x_buffer.detach().clone()
            audio_x = self.audio_x_buffer.detach().clone()
            x[0].index_copy_(0, self.img_pos_dev, video_rows)
            audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
        else:
            x = self.x_buffer
            audio_x = self.audio_x_buffer
            if refresh_all_media_rows or not self._x_buffer_primed:
                x[0].index_copy_(0, self.img_pos_dev, video_rows)
                audio_x[0].index_copy_(0, self.audio_pos_dev, audio_rows)
                self._x_buffer_primed = True
            else:
                x[0].index_copy_(
                    0,
                    self.img_target_seq_idx,
                    video_rows[self.video_target_slice],
                )
                audio_x[0].index_copy_(
                    0,
                    self.audio_target_seq_idx,
                    audio_rows[self.audio_target_slice],
                )
        unique_timesteps, inverse_indices, block_combined_indices = step_timesteps
        return {
            **self.static_kwargs,
            "x": x,
            "audio_x": audio_x,
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "block_combined_indices": block_combined_indices,
        }

    def _expand_step_timesteps(
        self,
        *,
        t_video: float,
        t_audio: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor],
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates: list[float] = []
        fill_groups: list[tuple[torch.Tensor, int]] = []
        base_slot = -1
        if self.n_video_timestep_rows > 0:
            base_slot = len(candidates)
            candidates.append(float(t_video))
        for seq_idx, value in (
            (self.img_cond_seq_idx, imgvid_cond_timestep),
            (self.audio_target_seq_idx, t_audio),
            (self.audio_ref_seq_idx, audio_ref_cond_timestep),
        ):
            if seq_idx.numel() > 0:
                fill_groups.append((seq_idx, len(candidates)))
                candidates.append(float(value))
        unique_cpu, slot_to_unique = torch.unique(
            torch.tensor(candidates, dtype=torch.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[base_slot]) if base_slot >= 0 else 0
        pattern = tuple(slot_to_unique.tolist())
        inverse_indices = inverse_indices_by_pattern.get(pattern)
        if inverse_indices is None:
            inverse_indices = torch.full(
                (self.seq_len,), base_index, dtype=torch.long, device=device
            )
            for seq_idx, slot in fill_groups:
                inverse_indices.index_fill_(0, seq_idx, int(slot_to_unique[slot]))
            inverse_indices_by_pattern[pattern] = inverse_indices
        block_combined = block_combined_by_pattern.get(pattern)
        if block_combined is None:
            block_combined = torch.add(
                self.block_token_tags,
                inverse_indices[self.local_row_slice],
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
            block_combined_by_pattern[pattern] = block_combined
        return unique_cpu.to(device), inverse_indices, block_combined

    def prepare_timestep_plan(
        self,
        *,
        video_timesteps: list[float],
        audio_timesteps: list[float],
        imgvid_cond_noise_aug: float,
        audio_ref_cond_noise_aug: float,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(video_timesteps) != len(audio_timesteps):
            raise ValueError("video/audio timestep plans must have equal length")
        cache_key = None
        if self._inference_static_runtime_plans_enabled:
            # Normalize through float32 on CPU exactly as _expand_step_timesteps
            # does.  This avoids reading any CUDA scalar back to Python.
            cache_key = (
                tuple(
                    float(value)
                    for value in torch.tensor(
                        video_timesteps, dtype=torch.float32
                    ).tolist()
                ),
                tuple(
                    float(value)
                    for value in torch.tensor(
                        audio_timesteps, dtype=torch.float32
                    ).tolist()
                ),
                float(
                    torch.tensor(
                        imgvid_cond_noise_aug, dtype=torch.float32
                    ).item()
                ),
                float(
                    torch.tensor(
                        audio_ref_cond_noise_aug, dtype=torch.float32
                    ).item()
                ),
            )
            cached = self._inference_timestep_plan_cache.get(cache_key)
            if cached is not None:
                self._inference_static_runtime_plan_hits += 1
                return cached
        inverse_indices_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        block_combined_by_pattern: dict[tuple[int, ...], torch.Tensor] = {}
        plan = [
            self._expand_step_timesteps(
                t_video=t_video,
                t_audio=t_audio,
                imgvid_cond_timestep=max(t_video, imgvid_cond_noise_aug),
                audio_ref_cond_timestep=max(t_audio, audio_ref_cond_noise_aug),
                inverse_indices_by_pattern=inverse_indices_by_pattern,
                block_combined_by_pattern=block_combined_by_pattern,
            )
            for t_video, t_audio in zip(video_timesteps, audio_timesteps)
        ]
        if cache_key is not None:
            self._inference_timestep_plan_cache[cache_key] = plan
            self._inference_static_runtime_plan_misses += 1
        return plan


def minimax_h3_denoise_loop(
    *,
    model: Any,
    positive: MiniMaxH3DenoiseBranch,
    initial_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    keyframe_cond_rows: torch.Tensor | None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    device: torch.device,
    model_forward: (
        Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    audio_ref_rows: torch.Tensor | None = None,
    imgvid_cond_noise_aug_for_inference: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_for_inference: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    on_initial_state: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    on_velocity: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    on_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    step_profiler: Callable[[int], AbstractContextManager] | None = None,
    consistency_renoise: bool = False,
    renoise_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run positive-only MiniMax H3 Euler-eta0 denoising for video and audio."""

    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("video/audio sigma schedules must have equal length")
    if len(sigmas_video) < 2:
        raise ValueError("sigma schedules need at least 2 entries")
    if consistency_renoise:
        if sigmas_video != sigmas_audio:
            raise ValueError(
                "consistency sampling requires one shared AV sigma schedule"
            )
        if renoise_generator is None:
            raise ValueError(
                "consistency sampling requires a fresh-noise generator"
            )
    n_cond = positive.video_target_start
    if keyframe_cond_rows is None:
        if n_cond != 0:
            raise ValueError(
                f"layout has {n_cond} cond rows but keyframe_cond_rows is None"
            )
    elif int(keyframe_cond_rows.shape[0]) != n_cond:
        raise ValueError(
            f"keyframe_cond_rows {int(keyframe_cond_rows.shape[0])} != "
            f"layout cond rows {n_cond}"
        )
    video_rows = initial_video_rows.to(device=device, dtype=torch.float32, copy=True)
    audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32, copy=True)
    if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
        raise ValueError(
            f"initial video rows {int(video_rows.shape[0])} != positive layout "
            f"rows {int(positive.img_pos.shape[0])}"
        )
    if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
        raise ValueError(
            f"initial audio rows {int(audio_rows.shape[0])} != positive layout "
            f"rows {int(positive.audio_pos.shape[0])}"
        )
    n_audio_ref = positive.audio_target_start
    if audio_ref_rows is None:
        if n_audio_ref != 0:
            raise ValueError(
                f"layout has {n_audio_ref} audio ref rows but audio_ref_rows is None"
            )
        audio_anchor = None
    else:
        if int(audio_ref_rows.shape[0]) != n_audio_ref:
            raise ValueError(
                f"audio_ref_rows {int(audio_ref_rows.shape[0])} != layout "
                f"audio ref rows {n_audio_ref}"
            )
        audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
    cond_anchor = (
        keyframe_cond_rows.to(device=device, dtype=torch.float32)
        if keyframe_cond_rows is not None
        else None
    )
    if cond_anchor is not None:
        video_rows.index_copy_(0, positive.cond_row_idx, cond_anchor)
    if audio_anchor is not None:
        audio_rows.index_copy_(0, positive.audio_ref_row_idx, audio_anchor)
    if on_initial_state is not None:
        on_initial_state(video_rows, audio_rows)

    video_target_slice = positive.video_target_slice
    audio_target_slice = positive.audio_target_slice
    video_timesteps = [1.0 - sigma for sigma in sigmas_video[:-1]]
    audio_timesteps = [1.0 - sigma for sigma in sigmas_audio[:-1]]
    video_step_t = torch.tensor(video_timesteps, dtype=torch.float32, device=device)
    audio_step_t = torch.tensor(audio_timesteps, dtype=torch.float32, device=device)
    timestep_plan = positive.prepare_timestep_plan(
        video_timesteps=video_timesteps,
        audio_timesteps=audio_timesteps,
        imgvid_cond_noise_aug=float(imgvid_cond_noise_aug_for_inference),
        audio_ref_cond_noise_aug=float(audio_cond_noise_aug_for_inference),
    )
    video_sigmas = torch.tensor(sigmas_video, dtype=torch.float32, device=device)
    audio_sigmas = torch.tensor(sigmas_audio, dtype=torch.float32, device=device)
    video_sigma_ratios = video_sigmas[1:] / video_sigmas[:-1]
    audio_sigma_ratios = audio_sigmas[1:] / audio_sigmas[:-1]
    video_sigma_t = 1.0 - video_step_t
    audio_sigma_t = 1.0 - audio_step_t
    video_one_minus_sigma_ratios = 1.0 - video_sigma_ratios
    audio_one_minus_sigma_ratios = 1.0 - audio_sigma_ratios
    video_denoised_scratch = torch.empty_like(video_rows[video_target_slice])
    audio_denoised_scratch = torch.empty_like(audio_rows[audio_target_slice])
    for step in range(len(sigmas_video) - 1):
        step_cm = step_profiler(step) if step_profiler is not None else nullcontext()
        with step_cm:
            sigma_video = sigmas_video[step]
            sigma_audio = sigmas_audio[step]
            forward_kwargs = positive.forward_kwargs(
                video_rows=video_rows,
                audio_rows=audio_rows,
                step_timesteps=timestep_plan[step],
            )
            with torch.no_grad():
                if model_forward is None:
                    velocity_video, velocity_audio = model(**forward_kwargs)
                else:
                    velocity_video, velocity_audio = model_forward(
                        model, forward_kwargs, step
                    )
                if on_velocity is not None:
                    on_velocity(step, velocity_video, velocity_audio)
                velocity_video = velocity_video.float()
                velocity_audio = velocity_audio[audio_target_slice].float()
                if consistency_renoise:
                    assert renoise_generator is not None
                    _minimax_h3_consistency_update_target_rows_(
                        video_rows[video_target_slice],
                        velocity_video,
                        sigma_curr=sigma_video,
                        sigma_next=sigmas_video[step + 1],
                        generator=renoise_generator,
                        denoised_scratch=video_denoised_scratch,
                    )
                    _minimax_h3_consistency_update_target_rows_(
                        audio_rows[audio_target_slice],
                        velocity_audio,
                        sigma_curr=sigma_audio,
                        sigma_next=sigmas_audio[step + 1],
                        generator=renoise_generator,
                        denoised_scratch=audio_denoised_scratch,
                    )
                else:
                    _minimax_h3_update_target_rows_(
                        video_rows[video_target_slice],
                        velocity_video,
                        sigma_t=video_sigma_t[step],
                        sigma_curr=sigma_video,
                        sigma_ratio=video_sigma_ratios[step],
                        one_minus_sigma_ratio=video_one_minus_sigma_ratios[step],
                        denoised_scratch=video_denoised_scratch,
                    )
                    _minimax_h3_update_target_rows_(
                        audio_rows[audio_target_slice],
                        velocity_audio,
                        sigma_t=audio_sigma_t[step],
                        sigma_curr=sigma_audio,
                        sigma_ratio=audio_sigma_ratios[step],
                        one_minus_sigma_ratio=audio_one_minus_sigma_ratios[step],
                        denoised_scratch=audio_denoised_scratch,
                    )
            if on_step is not None:
                on_step(step, video_rows, audio_rows)
    return video_rows, audio_rows


__all__ = [
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_AUDIO_ROW_WIDTH",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MINIMAX_H3_VIDEO_ROW_WIDTH",
    "MiniMaxH3DenoiseBranch",
    "minimax_h3_denoise_loop",
]
