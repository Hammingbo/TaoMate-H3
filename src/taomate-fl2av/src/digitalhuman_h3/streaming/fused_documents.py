from __future__ import annotations

from typing import Any


FUSED_DOCUMENT_ULYSSES_DEGREES = frozenset((1, 2, 4, 8))


def require_fused_document_ulysses_degree(degree: int) -> int:
    if degree not in FUSED_DOCUMENT_ULYSSES_DEGREES:
        raise ValueError(
            "fused H3 documents support Ulysses degree 1, 2, 4, or 8"
        )
    return degree


def split_first_denoise_interval(
    sigma_schedule: list[float],
) -> tuple[list[float], list[float]]:
    """Split one Euler interval while retaining its shared boundary state."""

    if len(sigma_schedule) < 3:
        raise ValueError("fused clean-denoise requires at least two denoise intervals")
    return sigma_schedule[:2], sigma_schedule[1:]


def build_fused_denoise_branch(
    *,
    torch: Any,
    previous: Any,
    current: Any,
    model: Any,
    device: Any,
    static_runtime_plans: bool = False,
    reusable_branch: Any | None = None,
    static_layout_token: Any | None = None,
) -> Any:
    """Pack clean(previous) and noisy(current) as independent documents."""

    from digitalhuman_h3.model.denoise import MiniMaxH3DenoiseBranch

    context = previous.parallel_context
    if context is not current.parallel_context:
        raise RuntimeError("fused H3 branches use different parallel contexts")
    require_fused_document_ulysses_degree(int(context.ulysses_world_size))

    if reusable_branch is not None:
        if not static_runtime_plans:
            raise RuntimeError("fused branch reuse requires static runtime plans")
        if reusable_branch._hiar_static_layout_token != static_layout_token:
            raise RuntimeError("fused branch static layout token differs")
        previous_seq = int(previous.seq_len)
        destination = reusable_branch.static_kwargs["img_position_ids"][0]
        previous_positions = previous.static_kwargs["img_position_ids"][0]
        current_positions = current.static_kwargs["img_position_ids"][0]
        if (
            int(destination.shape[0]) != previous_seq + int(current.seq_len)
            or tuple(destination.shape[1:]) != tuple(previous_positions.shape[1:])
            or tuple(destination.shape[1:]) != tuple(current_positions.shape[1:])
        ):
            raise RuntimeError("fused branch reusable position shape differs")
        destination[:previous_seq].copy_(previous_positions)
        destination[previous_seq:].copy_(current_positions)
        reusable_branch.static_kwargs["rope_cache"] = model.build_rope_cache(
            reusable_branch.static_kwargs["img_position_ids"], device=device
        )
        reusable_branch._x_buffer_primed = False
        reusable_branch._hiar_static_workspace_reuses += 1
        return reusable_branch

    def host_cu(branch: Any) -> tuple[int, ...]:
        packed = branch.static_kwargs["packed_seq_params"]
        return tuple(int(item) for item in packed["cu_seqlens_q_host"])

    previous_cu = host_cu(previous)
    current_cu = host_cu(current)
    if len(previous_cu) != 3 or len(current_cu) != 3:
        raise RuntimeError(
            "native H3 phase branch must contain live and padding documents"
        )
    previous_seq = int(previous.seq_len)
    current_seq = int(current.seq_len)
    combined_cu_host = (
        0,
        previous_cu[1],
        previous_seq,
        previous_seq + current_cu[1],
        previous_seq + current_seq,
    )

    def offset(value: Any, amount: int) -> Any:
        return value.detach().cpu().view(-1).to(torch.long) + int(amount)

    previous_tags = previous.token_tags.detach().cpu()
    current_tags = current.token_tags.detach().cpu()
    previous_text_pos = (
        previous.text_pos
        if static_runtime_plans
        else previous.static_kwargs["text_pos_info"]["position_ids"].detach().cpu()
    )
    current_text_pos = (
        current.text_pos
        if static_runtime_plans
        else current.static_kwargs["text_pos_info"]["position_ids"].detach().cpu()
    )
    previous_position_ids = (
        previous.img_position_ids_host
        if static_runtime_plans
        else previous.static_kwargs["img_position_ids"][0].detach().cpu()
    )
    current_position_ids = (
        current.img_position_ids_host
        if static_runtime_plans
        else current.static_kwargs["img_position_ids"][0].detach().cpu()
    )
    packed = {
        "seq_len": previous_seq + current_seq,
        "img_pos": torch.cat(
            (
                previous.img_pos.detach().cpu(),
                offset(current.img_pos, previous_seq),
            )
        ),
        "audio_pos": torch.cat(
            (
                previous.audio_pos.detach().cpu(),
                offset(current.audio_pos, previous_seq),
            )
        ),
        "text_pos": torch.cat(
            (
                previous_text_pos.view(-1),
                current_text_pos.view(-1).to(torch.long) + previous_seq,
            )
        ),
        "update_mask": torch.cat(
            (
                previous.update_mask.detach().cpu(),
                current.update_mask.detach().cpu(),
            )
        ),
        "audio_update_mask": torch.cat(
            (
                previous.audio_update_mask.detach().cpu(),
                current.audio_update_mask.detach().cpu(),
            )
        ),
        "img_position_ids": torch.cat(
            (
                previous_position_ids,
                current_position_ids,
            ),
            dim=0,
        ),
        "token_tags": torch.cat((previous_tags, current_tags)),
        "cu_seqlens": torch.tensor(combined_cu_host, dtype=torch.int32),
    }
    previous_prompt = previous.static_kwargs["prompt_embeds"]
    current_prompt = current.static_kwargs["prompt_embeds"]
    combined = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.cat((previous_prompt, current_prompt), dim=0),
        token_tags=packed["token_tags"],
        device=device,
        parallel_context=context,
    )
    combined.static_kwargs["refined_prompt_embeds_length"] = int(
        previous_prompt.shape[0] + current_prompt.shape[0]
    )
    combined.static_kwargs["rope_cache"] = model.build_rope_cache(
        combined.static_kwargs["img_position_ids"], device=device
    )
    # The one-document fast layout assumes text is one contiguous prefix.
    combined.static_kwargs.pop("local_embedding_layout", None)
    combined.static_kwargs["packed_seq_params"].update(
        cu_seqlens_q_host=combined_cu_host,
        max_seqlen_q=max(
            stop - start
            for start, stop in zip(combined_cu_host, combined_cu_host[1:])
        ),
    )
    combined._hiar_previous_seq_len = previous_seq
    combined._hiar_previous_video_rows = (
        int(previous.img_pos.numel() - previous.video_target_start)
        if static_runtime_plans
        else int(previous.update_mask.sum().item())
    )
    combined._hiar_previous_audio_rows = int(previous.audio_pos.numel())
    combined._hiar_static_layout_token = static_layout_token
    combined._hiar_static_workspace_reuses = 0
    if static_runtime_plans:
        timestep_owner = torch.full(
            (previous_seq + current_seq,),
            2,
            dtype=torch.long,
        )
        timestep_owner[:previous_seq].fill_(0)
        timestep_owner.index_fill_(0, previous.audio_pos, 1)
        timestep_owner.index_fill_(
            0, current.audio_pos + previous_seq, 3
        )
        previous_condition = previous.img_pos[~previous.update_mask]
        current_condition = current.img_pos[~current.update_mask] + previous_seq
        if previous_condition.numel():
            timestep_owner.index_fill_(0, previous_condition, 4)
        if current_condition.numel():
            timestep_owner.index_fill_(0, current_condition, 5)
        combined._hiar_fused_timestep_owner = timestep_owner.to(device)
        combined._hiar_fused_timestep_plan_cache = {}
        combined._hiar_fused_timestep_plan_hits = 0
        combined._hiar_fused_timestep_plan_misses = 0
        combined.enable_inference_static_runtime_plans()
    return combined


def fused_document_timestep_plan(
    torch: Any,
    *,
    branch: Any,
    previous_video_timestep: float,
    previous_audio_timestep: float,
    current_video_timestep: float,
    current_audio_timestep: float,
    previous_condition_timestep: float,
    current_condition_timestep: float,
) -> tuple[Any, Any, Any]:
    """Assign independent AdaLN timesteps to both packed documents."""

    from digitalhuman_h3.model.architecture import MINIMAX_H3_ADALN_MODALITY_NUM

    device = branch.img_pos_dev.device
    values = [
        previous_video_timestep,
        previous_audio_timestep,
        current_video_timestep,
        current_audio_timestep,
        max(previous_video_timestep, previous_condition_timestep),
        max(current_video_timestep, current_condition_timestep),
    ]
    static_runtime_plans = getattr(
        branch, "_inference_static_runtime_plans_enabled", False
    )
    if static_runtime_plans:
        values_cpu = torch.tensor(values, dtype=torch.float32)
        cache_key = tuple(float(value) for value in values_cpu.tolist())
        cached = branch._hiar_fused_timestep_plan_cache.get(cache_key)
        if cached is not None:
            branch._hiar_fused_timestep_plan_hits += 1
            return cached
    if static_runtime_plans:
        unique_cpu, slots_cpu = torch.unique(
            values_cpu, sorted=True, return_inverse=True
        )
        unique = unique_cpu.to(device)
        slots = slots_cpu.to(device)
        inverse = slots.index_select(0, branch._hiar_fused_timestep_owner)
        combined = torch.add(
            branch.block_token_tags,
            inverse[branch.local_row_slice],
            alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
        )
        result = (unique, inverse, combined)
        branch._hiar_fused_timestep_plan_cache[cache_key] = result
        branch._hiar_fused_timestep_plan_misses += 1
        return result
    unique, slots = torch.unique(
        torch.tensor(values, dtype=torch.float32, device=device),
        sorted=True,
        return_inverse=True,
    )
    previous_seq = int(branch._hiar_previous_seq_len)
    inverse = torch.empty(branch.seq_len, dtype=torch.long, device=device)
    inverse[:previous_seq].fill_(int(slots[0]))
    inverse[previous_seq:].fill_(int(slots[2]))
    previous_audio_positions = branch.audio_pos_dev[
        branch.audio_pos_dev < previous_seq
    ]
    current_audio_positions = branch.audio_pos_dev[
        branch.audio_pos_dev >= previous_seq
    ]
    inverse.index_fill_(0, previous_audio_positions, int(slots[1]))
    inverse.index_fill_(0, current_audio_positions, int(slots[3]))
    previous_condition = branch.img_pos_dev[
        (~branch.update_mask_dev) & (branch.img_pos_dev < previous_seq)
    ]
    current_condition = branch.img_pos_dev[
        (~branch.update_mask_dev) & (branch.img_pos_dev >= previous_seq)
    ]
    if previous_condition.numel():
        inverse.index_fill_(0, previous_condition, int(slots[4]))
    if current_condition.numel():
        inverse.index_fill_(0, current_condition, int(slots[5]))
    combined = torch.add(
        branch.block_token_tags,
        inverse[branch.local_row_slice],
        alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
    )
    return unique, inverse, combined


def fused_document_media_rows(
    torch: Any,
    *,
    branch: Any,
    previous_video_rows: Any,
    current_video_rows: Any,
    previous_audio_rows: Any,
    current_audio_rows: Any,
) -> tuple[Any, Any]:
    """Pack two documents into reusable inference scratch when enabled."""

    if not getattr(branch, "_inference_static_runtime_plans_enabled", False):
        return (
            torch.cat((previous_video_rows, current_video_rows), dim=0),
            torch.cat((previous_audio_rows, current_audio_rows), dim=0),
        )
    video_shape = (
        int(previous_video_rows.shape[0]) + int(current_video_rows.shape[0]),
        int(current_video_rows.shape[1]),
    )
    audio_shape = (
        int(previous_audio_rows.shape[0]) + int(current_audio_rows.shape[0]),
        int(current_audio_rows.shape[1]),
    )
    video = getattr(branch, "_hiar_fused_video_rows_scratch", None)
    audio = getattr(branch, "_hiar_fused_audio_rows_scratch", None)
    if video is None or tuple(video.shape) != video_shape:
        video = torch.empty(
            video_shape,
            dtype=current_video_rows.dtype,
            device=current_video_rows.device,
        )
        branch._hiar_fused_video_rows_scratch = video
    if audio is None or tuple(audio.shape) != audio_shape:
        audio = torch.empty(
            audio_shape,
            dtype=current_audio_rows.dtype,
            device=current_audio_rows.device,
        )
        branch._hiar_fused_audio_rows_scratch = audio
    previous_video_count = int(previous_video_rows.shape[0])
    previous_audio_count = int(previous_audio_rows.shape[0])
    video[:previous_video_count].copy_(previous_video_rows)
    video[previous_video_count:].copy_(current_video_rows)
    audio[:previous_audio_count].copy_(previous_audio_rows)
    audio[previous_audio_count:].copy_(current_audio_rows)
    return video, audio


def current_document_velocities(
    branch: Any,
    velocity_video: Any,
    velocity_audio: Any,
) -> tuple[Any, Any]:
    """Drop the previous clean document's unused output rows."""

    return (
        velocity_video[int(branch._hiar_previous_video_rows) :],
        velocity_audio[int(branch._hiar_previous_audio_rows) :],
    )


__all__ = [
    "FUSED_DOCUMENT_ULYSSES_DEGREES",
    "build_fused_denoise_branch",
    "current_document_velocities",
    "fused_document_media_rows",
    "fused_document_timestep_plan",
    "require_fused_document_ulysses_degree",
    "split_first_denoise_interval",
]
