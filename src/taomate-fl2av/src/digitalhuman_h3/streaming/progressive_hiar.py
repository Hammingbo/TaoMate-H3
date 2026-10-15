from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from .hiar_pp import (
    HIAR_STAGE_COUNT,
    HiARFullPrefixMemoryUpdate,
    HiARProgressAVKVCache,
    hiar_full_prefix_update_seconds,
)


PROGRESSIVE_HIAR_STAGE_STATE_INDICES = (0, 16, 33)


@dataclass(frozen=True)
class _StageHookView:
    hook: Any


def canonical_high_clean_prefix_update(
    *,
    plan: Any,
    phase_index: int,
    clean_video_rows: Any,
    clean_audio_rows: Any,
    frame_rows: int,
    pending_one_group_latents: Any | None,
) -> tuple[HiARFullPrefixMemoryUpdate | None, Any | None]:
    """Build the canonical high-clean AV-prefix update shared by train/infer."""

    import torch

    from .full_prefix import (
        FULL_PREFIX_AUDIO_CHANNELS,
        FULL_PREFIX_VIDEO_LATENT_UNITS,
        FullPrefixAVLatents,
        canonical_full_prefix_audio_rows,
        canonical_full_prefix_av_latents,
    )

    phase = plan.phases[phase_index]
    update_seconds = hiar_full_prefix_update_seconds(plan, phase_index)
    if update_seconds is None:
        pending = (
            FullPrefixAVLatents(
                video=clean_video_rows.detach().float().clone(),
                audio=clean_audio_rows.detach().float().clone(),
            )
            if int(phase.group_count) == 1
            else None
        )
        return None, pending
    if int(phase.group_count) == 2:
        canonical = canonical_full_prefix_av_latents(
            clean_video_rows,
            clean_audio_rows,
            phase=phase,
            frame_rows=frame_rows,
        )
        return (
            HiARFullPrefixMemoryUpdate(
                canonical_latents=canonical,
                block_seconds=float(update_seconds),
            ),
            None,
        )
    if pending_one_group_latents is None:
        raise RuntimeError("one-group HiAR prefix pair lost its first block")
    paired_video = torch.cat(
        (
            pending_one_group_latents.video,
            clean_video_rows.detach().float(),
        ),
        dim=0,
    )
    temporal_units, remainder = divmod(int(paired_video.shape[0]), frame_rows)
    if remainder or temporal_units < FULL_PREFIX_VIDEO_LATENT_UNITS:
        raise RuntimeError("one-group HiAR prefix pair has invalid video rows")
    pending_audio = pending_one_group_latents.audio.reshape(
        FULL_PREFIX_AUDIO_CHANNELS,
        -1,
        int(pending_one_group_latents.audio.shape[-1]),
    )
    current_audio = clean_audio_rows.detach().float().reshape(
        FULL_PREFIX_AUDIO_CHANNELS,
        -1,
        int(clean_audio_rows.shape[-1]),
    )
    paired_audio = torch.cat((pending_audio, current_audio), dim=1)
    canonical = FullPrefixAVLatents(
        video=(
            paired_video.reshape(
                temporal_units,
                frame_rows,
                int(paired_video.shape[-1]),
            )[-FULL_PREFIX_VIDEO_LATENT_UNITS:]
            .reshape(-1, int(paired_video.shape[-1]))
            .detach()
            .float()
            .clone()
        ),
        audio=canonical_full_prefix_audio_rows(
            paired_audio.reshape(-1, int(paired_audio.shape[-1])),
            audio_latent_units_per_channel=int(paired_audio.shape[1]),
        ),
    )
    return (
        HiARFullPrefixMemoryUpdate(
            canonical_latents=canonical,
            block_seconds=float(update_seconds),
        ),
        None,
    )


class ProgressiveHiARStageHistories:
    """Three independent stage-input AV-KV histories for Progressive HiAR."""

    def __init__(
        self,
        *,
        model: Any,
        max_history_blocks: int,
        attention_backend: str = "sdpa",
    ) -> None:
        from .attention_hook import H3StreamingAttentionHook
        from .cache import kv_contract_from_model
        from .capability import require_streaming_capability

        require_streaming_capability(model=model)
        contract = kv_contract_from_model(model)
        segmented = attention_backend in {
            "flash_attn3_fp8_segmented",
            "flash_attn3_fp8_segmented_topk",
        }
        self.max_history_blocks = int(max_history_blocks)
        self.caches = tuple(
            HiARProgressAVKVCache(
                contract=contract,
                max_history_blocks=self.max_history_blocks,
                segmented_fp8_history=segmented,
            )
            for _ in range(HIAR_STAGE_COUNT)
        )
        self.hooks = tuple(
            H3StreamingAttentionHook(
                cache,
                attention_backend=attention_backend,
                history_kv_recent_full_blocks=self.max_history_blocks,
            )
            for cache in self.caches
        )
        self._active_stage: int | None = None

    @property
    def committed_blocks(self) -> tuple[int, int, int]:
        return tuple(int(cache.committed_blocks) for cache in self.caches)

    @property
    def retained_blocks(self) -> tuple[int, int, int]:
        return tuple(int(cache.retained_blocks) for cache in self.caches)

    @property
    def history_tokens(self) -> tuple[int, int, int]:
        return tuple(int(cache.history_tokens) for cache in self.caches)

    @property
    def history_audio_tokens(self) -> tuple[int, int, int]:
        return tuple(int(cache.history_audio_tokens) for cache in self.caches)

    @property
    def history_video_tokens(self) -> tuple[int, int, int]:
        return tuple(int(cache.history_video_tokens) for cache in self.caches)

    def _stage(self, stage_index: int) -> tuple[Any, Any]:
        if not 0 <= int(stage_index) < HIAR_STAGE_COUNT:
            raise ValueError("Progressive HiAR stage index must be 0, 1, or 2")
        return self.caches[int(stage_index)], self.hooks[int(stage_index)]

    @contextmanager
    def install_stage(self, stage_index: int) -> Iterator[_StageHookView]:
        from .capability import set_official_attention_hook

        if self._active_stage is not None:
            raise RuntimeError("Progressive HiAR stage hooks cannot overlap")
        _, hook = self._stage(stage_index)
        self._active_stage = int(stage_index)
        set_official_attention_hook(hook)
        try:
            yield _StageHookView(hook=hook)
        finally:
            hook.deactivate()
            set_official_attention_hook(None)
            self._active_stage = None

    @contextmanager
    def denoise_and_commit_input(
        self,
        *,
        stage_index: int,
        block_index: int,
        full_prefix: Any | None = None,
    ) -> Iterator[None]:
        from .attention_hook import HookMode

        cache, hook = self._stage(stage_index)
        if int(block_index) != cache.committed_blocks:
            raise ValueError(
                "Progressive HiAR blocks must commit in global order"
            )
        cache.begin_progress_commit(int(block_index))
        try:
            with self.install_stage(stage_index):
                hook.activate(
                    mode=HookMode.PROGRESS_COMMIT,
                    block_index=int(block_index),
                    full_prefix=full_prefix,
                )
                yield
        except BaseException:
            if cache.clean_commit_active:
                cache.rollback()
            raise
        else:
            try:
                cache.commit_progress()
            except BaseException:
                if cache.clean_commit_active:
                    cache.rollback()
                raise

    def release(self) -> None:
        from .capability import set_official_attention_hook

        for hook, cache in zip(self.hooks, self.caches):
            hook.deactivate()
            if cache.clean_commit_active:
                cache.rollback()
            cache.clear()
        if self._active_stage is not None:
            set_official_attention_hook(None)
            self._active_stage = None

    def receipt(self) -> dict[str, object]:
        return {
            "schema": "digitalhuman_h3.streaming.progressive_hiar.v2",
            "history_mode": "stage_input_per_transition",
            "full_prefix_modality": "audio_video",
            "stage_state_indices": list(PROGRESSIVE_HIAR_STAGE_STATE_INDICES),
            "stage_committed_blocks": list(self.committed_blocks),
            "stage_history_blocks": list(self.retained_blocks),
            "stage_history_tokens": list(self.history_tokens),
            "max_history_blocks": self.max_history_blocks,
            "terminal_clean_forward": False,
        }


__all__ = [
    "PROGRESSIVE_HIAR_STAGE_STATE_INDICES",
    "ProgressiveHiARStageHistories",
    "canonical_high_clean_prefix_update",
]
