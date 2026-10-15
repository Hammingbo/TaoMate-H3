# Modified for endpoint inference: request-zero keyframe receipt; computation unchanged.
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import wraps
import os
from typing import Any, Callable, Iterator

from .cache import CleanAVKVCache, KVContract, kv_contract_from_model
from .dual_clean import NativeLowCleanCapture, resolution_local_session
from .full_prefix import (
    FULL_PREFIX_AUDIO_VIDEO,
    FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL,
    FULL_PREFIX_MODALITIES,
    FULL_PREFIX_VIDEO_ONLY,
    FULL_PREFIX_VIDEO_LATENT_UNITS,
    FullPrefixAVLatents,
    canonical_full_prefix_av_latents,
    canonical_full_prefix_rows,
    create_full_prefix_memory,
    place_full_prefix_av_latents,
    place_full_prefix_rows,
)
from .geometry import (
    AUDIO_LATENT_RATE,
    StreamPhase,
    StreamPlan,
    VIDEO_FPS,
    canonical_continuation_plan,
    plan_native_stream,
    video_temporal_position,
    video_temporal_positions,
)
from .policy import StreamingInferencePolicy
from .prompt_local_prefix import PromptLocalVideoPrefix, prompt_local_prefix_video_latents
from .session import H3StreamingSession


DENOISE_LOOP_MODULE = "digitalhuman_h3.model.denoise"
PACKED_SEQUENCE_MODULE = "digitalhuman_h3.model.packed_sequence"
VIDEO_POSITION_POLICY = "request_aligned_native_cycle"
CANONICAL_VIDEO_POSITION_POLICY = "global_canonical_codec_stream"
PROMPT_POSITION_POLICY = "current_prompt_precedes_request_media"


def _direct_request_full_prefix(
    *,
    enabled: bool,
    retained: Any | None,
    encode: Any,
) -> tuple[Any | None, Any | None, bool]:
    """Resolve one direct-request prefix and report whether it ran the DiT."""

    if enabled and retained is not None:
        return retained, retained, False
    prefix = encode()
    return prefix, (prefix if enabled and prefix is not None else retained), (
        prefix is not None
    )


@dataclass(frozen=True)
class StreamingPhaseExecution:
    phase_index: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int
    denoise_forwards: int
    clean_forwards: int
    committed_history_tokens: int
    full_prefix_tokens: int = 0
    full_prefix_forwards: int = 0
    prompt_local_video_prefix_tokens: int = 0
    phase_latent_handoff_used: bool = False
    full_prefix_capture_video_latents: int = 0
    full_prefix_capture_audio_latents_per_channel: int = 0
    global_block_index: int | None = None
    full_prefix_visible_source_block_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StreamingExecution:
    native_frame_count: int
    phase_executions: tuple[StreamingPhaseExecution, ...]
    request_task: str = "t2va"
    persistent_kv_modality: str = "audio_video"
    noisy_kv_writes: int = 0
    decode_mode: str = "official_full_clip_noncausal_video_vae"
    video_latent_offset: int = 0
    audio_latent_offset: int = 0
    starting_history_tokens: int = 0
    retained_history_tokens: int = 0
    cross_request_kv_reused: bool = False
    cross_request_kv_boundary_reset: bool = False
    cross_request_kv_boundary_reset_tokens: int = 0
    cross_request_clean_video_latent_handoff: bool = False
    full_prefix_half_life_seconds: float | None = None
    full_prefix_modality: str | None = None
    full_prefix_phases_used: int = 0
    streaming_policy: dict[str, object] | None = None
    request_index: int = 0
    current_prompt_rope_start: float = 0.0
    audio_kv_reset_applied: bool = False
    audio_kv_reset_tokens: int = 0
    starting_history_audio_tokens: int = 0
    starting_history_video_tokens: int = 0
    retained_history_audio_tokens: int = 0
    retained_history_video_tokens: int = 0
    canonical_continuation_geometry: bool = False
    native_frame_offset: int = 0
    published_native_frames: int = 0
    video_transport_prefix_latents: int = 0
    audio_transport_prefix_latents_per_channel: int = 0
    published_video_latents: int = 0
    published_audio_latents_per_channel: int = 0
    current_text_token_count: int = 0
    text_time_start: float = 0.0
    text_time_stop: float = 0.0
    media_time_start: float = 0.0
    request_video_time_shift: float = 0.0
    audio_origin_skew: float = 0.0
    fixed_media_time_origin: int = 0
    reuse_across_requests: bool = False
    position_capture: dict[str, object] | None = None
    progressive_hiar: dict[str, object] | None = None
    progressive_dual_clean: dict[str, object] | None = None
    prompt_local_video_prefix: dict[str, object] | None = None
    audio_teacher: dict[str, object] | None = None
    attention_backend_receipt: dict[str, object] | None = None
    clip_keyframe_deadline: dict[str, int] | None = None
    clip_keyframe_visible_frame_indices: tuple[int, ...] | None = None

    @property
    def phase_count(self) -> int:
        return len(self.phase_executions)

    @property
    def denoise_forwards(self) -> int:
        return sum(item.denoise_forwards for item in self.phase_executions)

    @property
    def clean_forwards(self) -> int:
        return sum(item.clean_forwards for item in self.phase_executions)

    @property
    def full_prefix_forwards(self) -> int:
        return sum(item.full_prefix_forwards for item in self.phase_executions)

    @property
    def history_partition_is_consistent(self) -> bool:
        return (
            self.starting_history_tokens
            == self.starting_history_audio_tokens
            + self.starting_history_video_tokens
            and self.retained_history_tokens
            == self.retained_history_audio_tokens
            + self.retained_history_video_tokens
        )

    def continues_history_from(self, previous: StreamingExecution) -> bool:
        """Match request-boundary history, including a declared audio reset."""

        if self.cross_request_kv_boundary_reset:
            return (
                self.history_partition_is_consistent
                and previous.history_partition_is_consistent
                and self.starting_history_tokens == 0
                and self.starting_history_audio_tokens == 0
                and self.starting_history_video_tokens == 0
                and self.cross_request_kv_boundary_reset_tokens
                == previous.retained_history_tokens
            )
        return (
            self.history_partition_is_consistent
            and previous.history_partition_is_consistent
            and previous.retained_history_tokens
            == self.starting_history_tokens + self.audio_kv_reset_tokens
            and previous.retained_history_audio_tokens
            == self.starting_history_audio_tokens + self.audio_kv_reset_tokens
            and previous.retained_history_video_tokens
            == self.starting_history_video_tokens
        )

    def to_dict(self) -> dict[str, object]:
        full_prefix_modality = self.full_prefix_modality
        if (
            full_prefix_modality is None
            and self.full_prefix_half_life_seconds is not None
        ):
            full_prefix_modality = FULL_PREFIX_AUDIO_VIDEO
        return {
            "native_frame_count": self.native_frame_count,
            "phase_count": self.phase_count,
            "denoise_forwards": self.denoise_forwards,
            "clean_forwards": self.clean_forwards,
            "full_prefix_forwards": self.full_prefix_forwards,
            "request_task": self.request_task,
            "persistent_kv_modality": self.persistent_kv_modality,
            "noisy_kv_writes": self.noisy_kv_writes,
            "decode_mode": self.decode_mode,
            "video_position_policy": (
                CANONICAL_VIDEO_POSITION_POLICY
                if self.canonical_continuation_geometry
                else VIDEO_POSITION_POLICY
            ),
            "prompt_position_policy": PROMPT_POSITION_POLICY,
            "video_latent_offset": self.video_latent_offset,
            "audio_latent_offset": self.audio_latent_offset,
            "starting_history_tokens": self.starting_history_tokens,
            "retained_history_tokens": self.retained_history_tokens,
            "cross_request_kv_reused": self.cross_request_kv_reused,
            "cross_request_kv_boundary_reset": self.cross_request_kv_boundary_reset,
            "cross_request_kv_boundary_reset_tokens": (
                self.cross_request_kv_boundary_reset_tokens
            ),
            "cross_request_clean_video_latent_handoff": (
                self.cross_request_clean_video_latent_handoff
            ),
            "full_prefix_half_life_seconds": self.full_prefix_half_life_seconds,
            "full_prefix_phases_used": self.full_prefix_phases_used,
            "full_prefix_modality": full_prefix_modality,
            "full_prefix_audio_latent_units_per_channel": (
                FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
                if full_prefix_modality == FULL_PREFIX_AUDIO_VIDEO
                else 0
            ),
            "streaming_policy": self.streaming_policy,
            "request_index": self.request_index,
            "current_prompt_rope_start": self.current_prompt_rope_start,
            "audio_kv_reset_applied": self.audio_kv_reset_applied,
            "audio_kv_reset_tokens": self.audio_kv_reset_tokens,
            "starting_history_audio_tokens": self.starting_history_audio_tokens,
            "starting_history_video_tokens": self.starting_history_video_tokens,
            "retained_history_audio_tokens": self.retained_history_audio_tokens,
            "retained_history_video_tokens": self.retained_history_video_tokens,
            "canonical_continuation_geometry": (
                self.canonical_continuation_geometry
            ),
            "native_frame_offset": self.native_frame_offset,
            "published_native_frames": self.published_native_frames,
            "video_transport_prefix_latents": (
                self.video_transport_prefix_latents
            ),
            "audio_transport_prefix_latents_per_channel": (
                self.audio_transport_prefix_latents_per_channel
            ),
            "published_video_latents": self.published_video_latents,
            "published_audio_latents_per_channel": (
                self.published_audio_latents_per_channel
            ),
            "current_text_token_count": self.current_text_token_count,
            "text_time_start": self.text_time_start,
            "text_time_stop": self.text_time_stop,
            "media_time_start": self.media_time_start,
            "request_video_time_shift": self.request_video_time_shift,
            "audio_origin_skew": self.audio_origin_skew,
            "fixed_media_time_origin": self.fixed_media_time_origin,
            "reuse_across_requests": self.reuse_across_requests,
            "position_capture": self.position_capture,
            "progressive_hiar": self.progressive_hiar,
            "progressive_dual_clean": self.progressive_dual_clean,
            "prompt_local_video_prefix": self.prompt_local_video_prefix,
            "audio_teacher": self.audio_teacher,
            "attention_backend_receipt": self.attention_backend_receipt,
            **({"clip_keyframe_deadline": dict(self.clip_keyframe_deadline)}
               if self.clip_keyframe_deadline is not None else {}),
            **({"clip_keyframe_visible_frame_indices": list(self.clip_keyframe_visible_frame_indices)}
               if self.clip_keyframe_visible_frame_indices is not None else {}),
            "phases": [item.to_dict() for item in self.phase_executions],
        }


@dataclass(frozen=True)
class _RequestTemporalAlignment:
    request_index: int
    current_text_token_count: int
    text_time_start: Fraction
    text_time_stop: Fraction
    media_time_start: Fraction
    request_video_time_shift: Fraction
    audio_origin_skew: Fraction


def _request_temporal_alignment(
    *,
    current_text_len: int,
    fixed_media_origin: int,
    total_frame_count: int,
    total_video_latents: int,
    video_latent_offset: int,
    audio_latent_offset: int,
) -> _RequestTemporalAlignment:
    """Right-align the current text to a fixed, video-derived media origin."""

    if current_text_len <= 0:
        raise ValueError("current text must contain at least one token")
    if total_frame_count <= 0 or total_video_latents <= 0:
        raise ValueError("streaming request geometry must be positive")
    if video_latent_offset < 0 or audio_latent_offset < 0:
        raise ValueError("streaming media offsets must be non-negative")
    request_index, remainder = divmod(video_latent_offset, total_video_latents)
    if remainder:
        raise ValueError(
            "video_latent_offset must start at a complete request boundary"
        )
    total_audio_latents = round(
        Fraction(total_frame_count * AUDIO_LATENT_RATE, VIDEO_FPS)
    )
    expected_audio_offset = request_index * total_audio_latents
    if audio_latent_offset != expected_audio_offset:
        raise ValueError(
            "audio_latent_offset does not match the video request boundary: "
            f"{audio_latent_offset} != {expected_audio_offset}"
        )
    request_video_time_shift = request_index * video_temporal_position(
        total_video_latents
    )
    media_time_start = Fraction(fixed_media_origin) + request_video_time_shift
    return _RequestTemporalAlignment(
        request_index=request_index,
        current_text_token_count=current_text_len,
        text_time_start=media_time_start - current_text_len,
        text_time_stop=media_time_start,
        media_time_start=media_time_start,
        request_video_time_shift=request_video_time_shift,
        audio_origin_skew=Fraction(audio_latent_offset) - request_video_time_shift,
    )


def _required_tensor(value: Any, name: str) -> Any:
    import torch

    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _kv_contract_for_model(model: Any) -> KVContract:
    """Resolve the canonical post-TP/post-Ulysses cache contract."""

    return kv_contract_from_model(model)


def _resolve_spatial_latent_shape(
    positive: Any,
    *,
    frame_rows: int,
) -> tuple[int, int]:
    """Recover the exact target latent canvas from official absolute positions."""

    import torch

    static = positive.static_kwargs
    position_ids = _required_tensor(
        static.get("img_position_ids"),
        "positive.static_kwargs['img_position_ids']",
    )
    target_info = static.get("img_pos_for_infer_output_info")
    if not isinstance(target_info, dict):
        raise TypeError("img_pos_for_infer_output_info must be a mapping")
    target_ids = _required_tensor(
        target_info.get("position_ids"),
        "img_pos_for_infer_output_info.position_ids",
    ).view(-1)
    if position_ids.ndim != 3 or int(position_ids.shape[0]) != 1:
        raise ValueError("official H3 img_position_ids must be [1, sequence, 3]")
    with torch.no_grad():
        target_grid = position_ids[0].index_select(
            0, target_ids.to(device=position_ids.device, dtype=torch.long)
        )
        latent_h = int(torch.unique(target_grid[:, 1]).numel()) * 2
        latent_w = int(torch.unique(target_grid[:, 2]).numel()) * 2
    if (latent_h // 2) * (latent_w // 2) != frame_rows:
        raise RuntimeError(
            "official H3 spatial RoPE grid and target packed-row count disagree"
        )
    return latent_h, latent_w


def _current_prompt_rope_start(
    *,
    text_len: int,
    media_time_origin: int,
    video_latent_offset: int,
    total_video_latents: int,
    canonical_continuation_geometry: bool = False,
) -> float:
    """Place the current prompt immediately before its request-local media."""

    if canonical_continuation_geometry:
        return float(
            media_time_origin
            + video_temporal_position(video_latent_offset)
            - text_len
        )
    request_index = video_latent_offset // total_video_latents
    request_time_origin = request_index * video_temporal_position(
        total_video_latents
    )
    return float(media_time_origin + request_time_origin - text_len)


def _streaming_prompt_rope_start(
    *,
    text_len: int,
    media_time_origin: int,
    video_latent_offset: int,
    total_video_latents: int,
    prompt_video_latent_start: int | None,
    canonical_continuation_geometry: bool = False,
) -> float:
    """Place text immediately before its prompt segment in one long stream."""

    if prompt_video_latent_start is None:
        return _current_prompt_rope_start(
            text_len=text_len,
            media_time_origin=media_time_origin,
            video_latent_offset=video_latent_offset,
            total_video_latents=total_video_latents,
            canonical_continuation_geometry=canonical_continuation_geometry,
        )
    if prompt_video_latent_start < 0:
        raise ValueError("prompt video latent start must be non-negative")
    return float(
        media_time_origin
        + video_temporal_position(prompt_video_latent_start)
        - text_len
    )


def _first_frame_condition_position(
    *,
    canonical_continuation_geometry: bool,
    request_index: int,
    video_latent_offset: int,
    total_video_latents: int,
) -> Fraction:
    """Place a continuation anchor on the prior clean video latent."""

    if canonical_continuation_geometry and video_latent_offset > 0:
        return video_temporal_position(video_latent_offset - 1)
    if request_index == 0:
        return video_temporal_position(0)
    return (
        (request_index - 1) * video_temporal_position(total_video_latents)
        + video_temporal_position(total_video_latents - 1)
    )


def _uses_bounded_rope_exact_layout(
    *,
    phase: StreamPhase,
    total_frame_count: int,
    total_video_latents: int,
    canonical_continuation_geometry: bool,
) -> bool:
    return (
        not canonical_continuation_geometry
        and total_frame_count == 243
        and total_video_latents == 72
        and phase.frame_start == 0
        and phase.frame_stop == total_frame_count
        and phase.video_latent_start == 0
        and phase.video_latent_stop == total_video_latents
        and phase.audio_latent_start == 0
        and phase.audio_latent_stop == 405
    )


def _compact_packed_grid_position_capture(
    *,
    grid: Any,
    text_rows: list[int],
    video_rows: list[int],
    audio_rows: list[list[int]],
    phase: StreamPhase,
) -> dict[str, object]:
    """Summarize values read from an already-written packed position grid."""

    def temporal(row: int) -> float:
        try:
            value = grid[row, 0]
        except TypeError:
            value = grid[row][0]
        item = getattr(value, "item", None)
        return float(item() if callable(item) else value)

    text = [temporal(row) for row in text_rows]
    video = [temporal(row) for row in video_rows]
    audio = [[temporal(row) for row in channel] for channel in audio_rows]
    return {
        "source": "actual_packed_grid",
        "text": {
            "count": len(text),
            "first": text[0],
            "last": text[-1],
            "stop_exclusive": text[-1] + 1.0,
            "unit_step": all(
                abs(current - previous - 1.0) < 1e-4
                for previous, current in zip(text, text[1:])
            ),
        },
        "video": {
            "unique_count": len({round(value, 5) for value in video}),
            "first": video[0],
            "last": video[-1],
        },
        "audio": {
            "channel_count": len(audio),
            "per_channel_count": len(audio[0]),
            "first": audio[0][0],
            "last": audio[0][-1],
            "channels_equal": all(
                abs(left - right) < 1e-4
                for left, right in zip(audio[0], audio[1])
            ),
        },
        "phase_boundaries": [
            {
                "phase_index": phase.index,
                "frame_start": phase.frame_start,
                "frame_stop": phase.frame_stop,
                "video_latent_start": phase.video_latent_start,
                "video_latent_stop": phase.video_latent_stop,
                "audio_latent_start": phase.audio_latent_start,
                "audio_latent_stop": phase.audio_latent_stop,
            }
        ],
    }


def build_streaming_phase_packed_layout(
    *,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    text_token_tags: Any,
    keyframe_frame_indices: tuple[int, ...],
    total_frame_count: int,
    total_video_latents: int,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
    prompt_video_latent_start: int | None = None,
    phase_tail_condition: bool = False,
    canonical_continuation_geometry: bool = False,
    clip_keyframe_end_video_latent: int | None = None,
) -> dict[str, Any]:
    """Build one future-free target phase on the native H3 video cadence."""

    import importlib
    import torch

    packed_module = importlib.import_module(PACKED_SEQUENCE_MODULE)
    build = getattr(packed_module, "minimax_h3_packed_sequence", None)
    if not callable(build):
        raise RuntimeError("pinned H3 packed-sequence builder is unavailable")
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    text_len = int(prompt_embeds.shape[0])
    packed = build(
        text_len=text_len,
        latent_t=phase.video_latent_count,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=phase.audio_latent_count,
        audio_channel=2,
        include_keyframe_cond=bool(keyframe_frame_indices),
        keyframe_frame_indices=keyframe_frame_indices or None,
        frame_count=total_frame_count if keyframe_frame_indices else None,
    )
    text_pos = _required_tensor(packed.get("text_pos"), "text_pos").view(-1)
    tags = _required_tensor(packed.get("token_tags"), "token_tags")
    if list(text_token_tags.shape) != [text_len]:
        raise ValueError("official H3 text token tags do not match prompt embeddings")
    tags[text_pos] = text_token_tags.to(device=tags.device, dtype=tags.dtype)

    grid = _required_tensor(packed.get("img_position_ids"), "img_position_ids")
    img_pos = _required_tensor(packed.get("img_pos"), "img_pos").view(-1)
    update_mask = _required_tensor(
        packed.get("update_mask"), "update_mask"
    ).view(-1)
    target_pos = img_pos[update_mask]
    condition_pos = img_pos[~update_mask]
    frame_rows = (latent_h // 2) * (latent_w // 2)
    if int(target_pos.numel()) != phase.video_latent_count * frame_rows:
        raise RuntimeError("phase video packed rows do not match its latent geometry")
    exact_alignment = _uses_bounded_rope_exact_layout(
        phase=phase,
        total_frame_count=total_frame_count,
        total_video_latents=total_video_latents,
        canonical_continuation_geometry=canonical_continuation_geometry,
    )
    temporal_alignment = (
        _request_temporal_alignment(
            current_text_len=text_len,
            fixed_media_origin=media_time_origin,
            total_frame_count=total_frame_count,
            total_video_latents=total_video_latents,
            video_latent_offset=video_latent_offset,
            audio_latent_offset=audio_latent_offset,
        )
        if exact_alignment
        else None
    )
    if temporal_alignment is None:
        request_index, request_local_start = divmod(
            video_latent_offset + phase.video_latent_start,
            total_video_latents,
        )
        request_time_origin = request_index * video_temporal_position(
            total_video_latents
        )
        prompt_time_origin = _streaming_prompt_rope_start(
            text_len=text_len,
            media_time_origin=media_time_origin,
            video_latent_offset=video_latent_offset,
            total_video_latents=total_video_latents,
            prompt_video_latent_start=prompt_video_latent_start,
            canonical_continuation_geometry=canonical_continuation_geometry,
        )
    else:
        request_index = temporal_alignment.request_index
        request_local_start = phase.video_latent_start
        request_time_origin = temporal_alignment.request_video_time_shift
        prompt_time_origin = float(temporal_alignment.text_time_start)
    grid[text_pos, 0] = prompt_time_origin + torch.arange(
        text_len,
        dtype=grid.dtype,
        device=grid.device,
    )
    if canonical_continuation_geometry:
        video_positions = video_temporal_positions(
            video_latent_offset + phase.video_latent_start,
            phase.video_latent_count,
        )
    else:
        video_positions = tuple(
            request_time_origin + position
            for position in video_temporal_positions(
                request_local_start,
                phase.video_latent_count,
            )
        )
    video_times = torch.tensor(
        [float(media_time_origin + position) for position in video_positions],
        dtype=grid.dtype,
    )
    grid[target_pos, 0] = video_times.repeat_interleave(frame_rows)
    if keyframe_frame_indices:
        if int(condition_pos.numel()) != len(keyframe_frame_indices) * frame_rows:
            raise RuntimeError("FL2VA phase has invalid condition rows")
        for condition_index, frame_index in enumerate(keyframe_frame_indices):
            rows = condition_pos[
                condition_index * frame_rows : (condition_index + 1) * frame_rows
            ]
            if clip_keyframe_end_video_latent is not None:
                # Both anchors belong to the whole clip, not the current
                # prompt/request window. Native RoPE exposes time-to-target.
                condition_position = (
                    Fraction(0)
                    if frame_index == 0
                    else video_temporal_position(clip_keyframe_end_video_latent)
                    - video_temporal_position(1)
                )
            elif phase_tail_condition:
                condition_position = (
                    video_temporal_position(
                        video_latent_offset + phase.video_latent_start - 1
                    )
                    if canonical_continuation_geometry
                    else request_time_origin
                    + video_temporal_position(phase.video_latent_start - 1)
                )
            elif frame_index == 0:
                condition_position = _first_frame_condition_position(
                    canonical_continuation_geometry=(
                        canonical_continuation_geometry
                    ),
                    request_index=request_index,
                    video_latent_offset=video_latent_offset,
                    total_video_latents=total_video_latents,
                )
            else:
                condition_position = (
                    request_time_origin
                    + video_temporal_positions(total_video_latents, 1)[0]
                    - video_temporal_position(1)
                )
            condition_time = float(media_time_origin + condition_position)
            grid[rows, 0] = condition_time

    audio_pos = _required_tensor(packed.get("audio_pos"), "audio_pos").view(
        2, phase.audio_latent_count
    )
    audio_times = media_time_origin + torch.arange(
        audio_latent_offset + phase.audio_latent_start,
        audio_latent_offset + phase.audio_latent_stop,
        dtype=grid.dtype,
    )
    grid[audio_pos[0], 0] = audio_times
    grid[audio_pos[1], 0] = audio_times
    if exact_alignment:
        packed["_digitalhuman_h3_position_capture"] = (
            _compact_packed_grid_position_capture(
                grid=grid,
                text_rows=[int(value) for value in text_pos.tolist()],
                video_rows=[
                    int(value)
                    for value in target_pos.view(
                        phase.video_latent_count, frame_rows
                    )[:, 0].tolist()
                ],
                audio_rows=[
                    [int(value) for value in channel]
                    for channel in audio_pos.tolist()
                ],
                phase=phase,
            )
        )
    return packed


def _build_phase_branch(
    *,
    loop_module: Any,
    positive: Any,
    phase: StreamPhase,
    latent_h: int,
    latent_w: int,
    model: Any,
    device: Any,
    text_token_tags: Any,
    keyframe_frame_indices: tuple[int, ...],
    total_frame_count: int,
    total_video_latents: int,
    media_time_origin: int,
    video_latent_offset: int,
    audio_latent_offset: int,
    prompt_video_latent_start: int | None = None,
    phase_tail_condition: bool = False,
    canonical_continuation_geometry: bool = False,
    reusable_branch: Any | None = None,
    static_runtime_plans: bool = False,
    static_layout_token: Any | None = None,
    clip_keyframe_end_video_latent: int | None = None,
) -> Any:
    import torch

    branch_type = getattr(loop_module, "MiniMaxH3DenoiseBranch", None)
    if branch_type is None:
        raise RuntimeError("pinned H3 MiniMaxH3DenoiseBranch is unavailable")
    packed = build_streaming_phase_packed_layout(
        positive=positive,
        phase=phase,
        latent_h=latent_h,
        latent_w=latent_w,
        text_token_tags=text_token_tags,
        keyframe_frame_indices=keyframe_frame_indices,
        total_frame_count=total_frame_count,
        total_video_latents=total_video_latents,
        media_time_origin=media_time_origin,
        video_latent_offset=video_latent_offset,
        audio_latent_offset=audio_latent_offset,
        prompt_video_latent_start=prompt_video_latent_start,
        phase_tail_condition=phase_tail_condition,
        canonical_continuation_geometry=canonical_continuation_geometry,
        clip_keyframe_end_video_latent=clip_keyframe_end_video_latent,
    )
    position_capture = packed.pop("_digitalhuman_h3_position_capture", None)
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    if reusable_branch is None:
        branch = branch_type(
            packed=packed,
            text_embeddings=prompt_embeds,
            token_tags=packed["token_tags"],
            device=device,
            parallel_context=model.parallel_context,
        )
        branch._digitalhuman_h3_static_layout_token = static_layout_token
        branch._digitalhuman_h3_static_workspace_reuses = 0
        if static_runtime_plans:
            branch.enable_inference_static_runtime_plans()
    else:
        branch = reusable_branch
        if not static_runtime_plans:
            raise RuntimeError("phase branch reuse requires static runtime plans")
        if (
            branch._digitalhuman_h3_static_layout_token
            != static_layout_token
        ):
            raise RuntimeError("phase branch static layout token differs")
        position_ids = packed["img_position_ids"][None].to(
            device=device,
            dtype=torch.float32,
        )
        destination = branch.static_kwargs["img_position_ids"]
        if tuple(destination.shape) != tuple(position_ids.shape):
            raise RuntimeError("phase branch reusable position shape differs")
        destination.copy_(position_ids)
        branch.img_position_ids_host = packed["img_position_ids"].detach().to(
            dtype=torch.float32,
            device="cpu",
        )
        # A freshly constructed branch refreshes all media rows on its first
        # forward.  Preserve that exact behavior after changing the absolute
        # RoPE positions while retaining the already allocated buffers.
        branch._x_buffer_primed = False
        branch._digitalhuman_h3_static_workspace_reuses += 1

    refined_length = positive.static_kwargs.get("refined_prompt_embeds_length")
    if refined_length is not None:
        branch.static_kwargs["refined_prompt_embeds_length"] = int(refined_length)
    build_rope = getattr(model, "build_rope_cache", None)
    if not callable(build_rope):
        raise RuntimeError("pinned H3 model does not expose build_rope_cache")
    with torch.no_grad():
        branch.static_kwargs["rope_cache"] = build_rope(
            branch.static_kwargs["img_position_ids"],
            device=device,
        )
    branch._digitalhuman_h3_position_capture = position_capture
    return branch


def _resolve_global_text_token_tags(positive: Any) -> Any:
    """Recover official text/vision AdaLN tags before rebuilding phase layouts."""

    import torch

    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    text_len = int(prompt_embeds.shape[0])
    local_tags = _required_tensor(
        positive.static_kwargs.get("block_token_tags"),
        "positive.static_kwargs['block_token_tags']",
    ).view(-1)
    seq_len = int(getattr(positive, "seq_len", 0))
    if seq_len <= 0:
        raise ValueError("official H3 branch sequence length is invalid")
    if int(local_tags.numel()) == seq_len:
        global_tags = local_tags
    else:
        from digitalhuman_h3.distributed import ulysses_all_gather_rows

        global_tags = ulysses_all_gather_rows(
            local_tags.view(-1, 1), context=positive.parallel_context
        ).view(-1)
    if int(global_tags.numel()) != seq_len or text_len > seq_len:
        raise RuntimeError("official H3 global token-tag reconstruction differs")
    tags = global_tags[:text_len].detach().to(device="cpu", dtype=torch.long)
    if bool(((tags != 0) & (tags != 1)).any().item()):
        raise ValueError("H3 prompt rows contain unsupported streaming token tags")
    return tags


def _run_base_audio_teacher(
    *,
    enabled: bool,
    original_loop: Any,
    loop_module: Any,
    model: Any,
    positive: Any,
    initial_audio_rows: Any,
    total_audio_latents: int,
    latent_h: int,
    latent_w: int,
    device: Any,
) -> dict[str, Any] | None:
    """Plan one whole request's audio once for all of its streaming chunks."""

    if not enabled:
        return None
    import importlib
    import time

    import torch

    packed_module = importlib.import_module(PACKED_SEQUENCE_MODULE)
    build = getattr(packed_module, "minimax_h3_audio_only_packed_sequence", None)
    branch_type = getattr(loop_module, "MiniMaxH3DenoiseBranch", None)
    if not callable(build) or branch_type is None:
        raise RuntimeError("Base audio teacher support is unavailable")
    raw_text = _required_tensor(
        getattr(positive, "_digitalhuman_h3_raw_text_embeddings", None),
        "positive._digitalhuman_h3_raw_text_embeddings",
    )
    raw_tags = _required_tensor(
        getattr(positive, "_digitalhuman_h3_raw_text_tags", None),
        "positive._digitalhuman_h3_raw_text_tags",
    ).view(-1)
    text_len = int(raw_text.shape[0])
    if list(raw_tags.shape) != [text_len]:
        raise RuntimeError("Base audio teacher text tags changed shape")
    packed = build(
        text_len=text_len,
        audio_t=total_audio_latents,
        latent_h=latent_h,
        latent_w=latent_w,
    )
    packed["token_tags"][packed["text_pos"].view(-1)] = raw_tags.to(
        device="cpu", dtype=packed["token_tags"].dtype
    )
    teacher = branch_type(
        packed=packed,
        text_embeddings=raw_text,
        token_tags=packed["token_tags"],
        device=device,
        parallel_context=model.parallel_context,
    )

    from digitalhuman_h3.denoise_schedule import select_time_shift_sigmas
    from digitalhuman_h3.inference.lora_checkpoint import h3_base_model_inference

    sigmas_video = select_time_shift_sigmas(num_steps=10, shift_scale=12.0)
    sigmas_audio = select_time_shift_sigmas(num_steps=10, shift_scale=3.0)
    milestones: dict[int, Any] = {}

    def capture(step: int, _video: Any, audio: Any) -> None:
        milestone_index = {2: 0, 5: 1, 8: 2}.get(step)
        if milestone_index is not None:
            milestones[milestone_index] = audio.detach().clone()

    empty_video = torch.empty(
        (0, int(model.arch.video_row_width)), dtype=torch.float32, device="cpu"
    )
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with h3_base_model_inference(model), torch.no_grad():
        refined = model.refine_prompt_embeds(
            teacher.static_kwargs["prompt_embeds"],
            teacher.static_kwargs["refiner_packed_seq_params"]["cu_seqlens_q"],
            device=device,
        )
        teacher.static_kwargs["prompt_embeds"] = refined
        teacher.static_kwargs["refined_prompt_embeds_length"] = int(
            refined.shape[0]
        )
        teacher.static_kwargs["rope_cache"] = model.build_rope_cache(
            teacher.static_kwargs["img_position_ids"], device=device
        )
        original_loop(
            model=model,
            positive=teacher,
            initial_video_rows=empty_video,
            initial_audio_rows=initial_audio_rows,
            keyframe_cond_rows=None,
            audio_ref_rows=None,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=device,
            on_step=capture,
        )
    torch.cuda.synchronize(device)
    if set(milestones) != {0, 1, 2}:
        raise RuntimeError("Base audio teacher milestone capture is incomplete")
    return {
        "milestones": milestones,
        "receipt": {
            "schema": "digitalhuman_h3.streaming.base_audio_teacher.v1",
            "mode": "base10_milestones",
            "model": "base_h3_with_inference_lora_bypassed",
            "packed_video_rows": 0,
            "packed_text_rows": text_len,
            "packed_audio_rows": 2 * total_audio_latents,
            "packed_live_rows": int(packed["cu_seqlens"][1].item()),
            "packed_padded_rows": int(packed["seq_len"]),
            "audio_latents_per_channel": total_audio_latents,
            "schedule": "base_10_state_prefix",
            "full_state_count": 10,
            "executed_forwards": 9,
            "captured_base_state_numbers": [3, 6, 9],
            "stage3_target_state_indices": [16, 33, 49],
            "audio_sigmas_at_captured_states": [
                float(sigmas_audio[3]),
                float(sigmas_audio[6]),
                float(sigmas_audio[9]),
            ],
            "wall_seconds": time.perf_counter() - started,
            "reused_current_request_initial_audio_noise": True,
            "reused_by_every_streaming_chunk": True,
            "video_projection_executed": False,
            "video_output_head_executed": False,
        },
    }


def _resolve_base_audio_teacher(
    *,
    external_artifact: Any | None,
    request_index: int,
    torch_module: Any,
    device: Any,
    run_internal: Callable[[], dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Load an external BF16 teacher or run the existing in-model teacher."""

    if external_artifact is not None:
        return external_artifact.load_request(
            torch_module,
            request_index=request_index,
            device=device,
        )
    return run_internal()


def _teacher_phase_audio_rows(
    rows: Any,
    *,
    phase: StreamPhase,
    total_audio_latents: int,
) -> Any:
    return (
        rows.view(2, total_audio_latents, -1)[
            :, phase.audio_latent_start : phase.audio_latent_stop
        ]
        .contiguous()
        .view(-1, int(rows.shape[-1]))
    )


def _active_base_audio_teacher(
    teacher: dict[str, Any] | None,
    *,
    official_audio_latents: int,
    active_audio_latents: int,
    transport_prefix_audio_latents: int,
) -> dict[str, Any] | None:
    """Expose an exact active suffix while retaining the official teacher receipt."""

    if teacher is None:
        return None
    if getattr(teacher, "streaming_milestones", False):
        return teacher.active(
            official_audio_latents=official_audio_latents,
            active_audio_latents=active_audio_latents,
            transport_prefix_audio_latents=transport_prefix_audio_latents,
        )
    if (
        official_audio_latents <= 0
        or active_audio_latents <= 0
        or transport_prefix_audio_latents < 0
        or active_audio_latents + transport_prefix_audio_latents
        != official_audio_latents
    ):
        raise ValueError("Base audio teacher active geometry differs from its request")
    milestones = teacher.get("milestones")
    receipt = teacher.get("receipt")
    if not isinstance(milestones, dict) or not isinstance(receipt, dict):
        raise RuntimeError("Base audio teacher payload is invalid")
    if receipt.get("active_milestones_are_exact_canonical_target") is True:
        if receipt.get("audio_latents_per_channel") != active_audio_latents:
            raise RuntimeError(
                "rollover Base audio teacher differs from the canonical target"
            )
        for rows in milestones.values():
            if list(rows.shape) != [2 * active_audio_latents, 32]:
                raise RuntimeError(
                    "rollover Base audio teacher milestone changed target geometry"
                )
        active_receipt = dict(receipt)
        active_receipt.update(
            {
                "official_audio_latents_per_channel": official_audio_latents,
                "active_audio_latents_per_channel": active_audio_latents,
                "transport_prefix_audio_latents_per_channel": (
                    transport_prefix_audio_latents
                ),
                "active_audio_slice_per_channel": [0, active_audio_latents],
            }
        )
        return {"milestones": milestones, "receipt": active_receipt}
    if receipt.get("audio_latents_per_channel") != official_audio_latents:
        raise RuntimeError("Base audio teacher no longer has official request geometry")

    active_milestones: dict[int, Any] = {}
    for milestone_index, rows in milestones.items():
        if list(rows.shape) != [2 * official_audio_latents, 32]:
            raise RuntimeError("Base audio teacher milestone changed official geometry")
        active_milestones[int(milestone_index)] = (
            rows.view(2, official_audio_latents, 32)[
                :, transport_prefix_audio_latents:
            ]
            .contiguous()
            .view(2 * active_audio_latents, 32)
        )

    active_receipt = dict(receipt)
    active_receipt.update(
        {
            "official_audio_latents_per_channel": official_audio_latents,
            "active_audio_latents_per_channel": active_audio_latents,
            "transport_prefix_audio_latents_per_channel": (
                transport_prefix_audio_latents
            ),
            "active_audio_slice_per_channel": [
                transport_prefix_audio_latents,
                official_audio_latents,
            ],
            "active_milestones_are_exact_official_suffix": True,
        }
    )
    return {"milestones": active_milestones, "receipt": active_receipt}


def _base_audio_teacher_step_callback(
    teacher: dict[str, Any] | None,
    *,
    phase: StreamPhase,
    total_audio_latents: int,
    audio_target_slice: slice,
    step_offset: int = 0,
    outer: Callable[[int, Any, Any], None] | None = None,
) -> Callable[[int, Any, Any], None] | None:
    """Inject the request-wide Base10 milestone used by one Stage3 loop."""

    if teacher is None:
        return outer
    milestones = teacher["milestones"]

    def on_step(step_index: int, video_rows: Any, audio_rows: Any) -> None:
        milestone = milestones.get(step_offset + step_index)
        if milestone is not None:
            teacher_rows = _teacher_phase_audio_rows(
                milestone,
                phase=phase,
                total_audio_latents=total_audio_latents,
            )
            audio_rows[audio_target_slice].copy_(teacher_rows)
        if outer is not None:
            outer(step_index, video_rows, audio_rows)

    return on_step


@contextmanager
def _native_attention_metadata(
    branch: Any,
    *,
    live_documents_hint: tuple[int, ...] | None = None,
    mlp_video_stride: int = 1,
) -> Iterator[None]:
    """Bind full-sequence modality tags and target-media commit rows."""

    import torch

    from digitalhuman_h3.distributed import ulysses_all_gather_rows
    from digitalhuman_h3.model.layers import native_stream_attention_metadata

    metadata_key = (live_documents_hint, int(mlp_video_stride))
    metadata_cache = (
        branch._inference_native_attention_metadata_cache
        if getattr(branch, "_inference_static_runtime_plans_enabled", False)
        else None
    )
    cached = None if metadata_cache is None else metadata_cache.get(metadata_key)
    if cached is None:
        tags = _required_tensor(
            branch.static_kwargs.get("block_token_tags"), "block_token_tags"
        ).view(-1)
        if int(tags.numel()) != int(branch.seq_len):
            tags = ulysses_all_gather_rows(
                tags.view(-1, 1), context=branch.parallel_context
            ).view(-1)
        if int(tags.numel()) != int(branch.seq_len):
            raise RuntimeError(
                "native H3 streaming token tags do not cover the sequence"
            )
        mask = torch.zeros(
            int(branch.seq_len), dtype=torch.bool, device=tags.device
        )
        mask.index_fill_(0, branch.img_target_seq_idx.to(tags.device), True)
        mask.index_fill_(0, branch.audio_target_seq_idx.to(tags.device), True)
        if metadata_cache is not None:
            metadata_cache[metadata_key] = (tags, mask)
            branch._inference_native_attention_metadata_misses += 1
    else:
        tags, mask = cached
        branch._inference_native_attention_metadata_hits += 1
    with native_stream_attention_metadata(
        tags,
        mask,
        live_documents_hint=live_documents_hint,
        mlp_video_stride=mlp_video_stride,
    ):
        yield


@dataclass(frozen=True)
class _RequestVideoState:
    task: str
    target_initial_rows: Any
    condition_initial_rows: Any | None
    keyframe_condition_rows: Any | None
    keyframe_frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class _ClipKeyframeState:
    condition_initial_rows: Any
    keyframe_condition_rows: Any
    keyframe_frame_indices: tuple[int, ...] = (0, -1)


def _resolve_request_video_state(
    *,
    positive: Any,
    initial_video: Any,
    keyframe_cond_rows: Any,
    frame_rows: int,
    total_video_latents: int,
) -> _RequestVideoState:
    """Admit T2VA, one first-frame anchor, or first-plus-last anchors."""

    import torch

    condition_rows = int(positive.video_target_start)
    target_rows = total_video_latents * frame_rows
    if list(initial_video.shape) != [condition_rows + target_rows, 96]:
        raise ValueError(
            "request video rows do not match condition plus native stream geometry"
        )
    if condition_rows == 0:
        if keyframe_cond_rows is not None:
            raise ValueError("T2VA streaming request unexpectedly carries keyframe rows")
        return _RequestVideoState(
            task="t2va",
            target_initial_rows=initial_video,
            condition_initial_rows=None,
            keyframe_condition_rows=None,
            keyframe_frame_indices=(),
        )
    if condition_rows not in {frame_rows, 2 * frame_rows} or not torch.is_tensor(keyframe_cond_rows):
        raise NotImplementedError(
            "streaming admits one first-frame or two first/last FL2VA keyframes"
        )
    if list(keyframe_cond_rows.shape) != [condition_rows, 96]:
        raise ValueError("FL2VA keyframe rows do not match the target latent canvas")

    position_ids = _required_tensor(
        positive.static_kwargs.get("img_position_ids"),
        "positive.static_kwargs['img_position_ids']",
    )
    img_pos = _required_tensor(getattr(positive, "img_pos", None), "positive.img_pos")
    prompt_embeds = _required_tensor(
        positive.static_kwargs.get("prompt_embeds"),
        "positive.static_kwargs['prompt_embeds']",
    )
    condition_positions = position_ids[0].index_select(
        0,
        img_pos[:condition_rows].to(device=position_ids.device, dtype=torch.long),
    )
    expected_time = float(prompt_embeds.shape[0])
    if not bool((condition_positions[:frame_rows, 0] == expected_time).all().item()):
        raise NotImplementedError(
            "streaming FL2VA requires the first condition at frame zero"
        )
    if condition_rows == 2 * frame_rows:
        expected_last_time = float(
            prompt_embeds.shape[0]
            + video_temporal_positions(total_video_latents, 1)[0]
            - 5 / 3
        )
        last_times = condition_positions[frame_rows:, 0]
        if not torch.allclose(
            last_times,
            torch.full_like(last_times, expected_last_time),
            rtol=0.0,
            atol=1e-5,
        ):
            raise NotImplementedError(
                "II2AV streaming requires the second condition at the last frame"
            )
    frame_indices = (0,) if condition_rows == frame_rows else (0, -1)
    return _RequestVideoState(
        task=("fl2va_first_frame" if len(frame_indices) == 1 else "fl2va_first_last"),
        target_initial_rows=initial_video[condition_rows:],
        condition_initial_rows=initial_video[:condition_rows],
        keyframe_condition_rows=keyframe_cond_rows,
        keyframe_frame_indices=frame_indices,
    )


def _phase_initial_rows(
    *,
    initial_video_rows: Any,
    initial_audio_rows: Any,
    phase: StreamPhase,
    frame_rows: int,
    total_audio_latents: int,
) -> tuple[Any, Any]:
    video = initial_video_rows[
        phase.video_latent_start * frame_rows : phase.video_latent_stop * frame_rows
    ].contiguous()
    audio = initial_audio_rows.view(2, total_audio_latents, -1)[
        :, phase.audio_latent_start : phase.audio_latent_stop
    ].contiguous()
    return video, audio.view(-1, int(audio.shape[-1]))


def _join_phase_audio(rows: list[Any], phases: tuple[StreamPhase, ...]) -> Any:
    import torch

    by_channel: list[list[Any]] = [[], []]
    for value, phase in zip(rows, phases):
        current = value.view(2, phase.audio_latent_count, -1)
        by_channel[0].append(current[0])
        by_channel[1].append(current[1])
    return torch.cat(
        (torch.cat(by_channel[0], dim=0), torch.cat(by_channel[1], dim=0)),
        dim=0,
    )


class H3StreamingRuntimeOverlay:
    """Phase scheduler around the pinned H3 denoise loop.

    The official request validation, encoders, sigma schedules, output
    publication, and full-clip decoders remain unchanged.  This overlay only
    replaces the full target denoise loop with future-free native AV phases.
    By default FL2VA conditions appear only in phase zero. The opt-in
    clip_absolute_persistent policy retains explicit II2VA first/last references
    in every phase, at fixed whole-clip positions. These references are separate
    from generated clean AV-KV. Long-video continuation still uses T2VA transport
    and does not inject a generated terminal latent a second time.
    """

    def __init__(
        self,
        *,
        plan: StreamPlan,
        server_args: Any,
        reuse_across_requests: bool = False,
        full_prefix_half_life_seconds: float | None = None,
        full_prefix_modality: str = FULL_PREFIX_AUDIO_VIDEO,
        policy: StreamingInferencePolicy | None = None,
        attention_backend: str = "sdpa",
        sol_attn_tau: float = 1.0,
        sol_attn_dense_layers: int = 2,
        sol_attn_dense_denoise_steps: int = 1,
        sol_attn_route_current_video: bool = False,
        direct_sage_kv_workspace: bool = False,
        sol_attn_static_summary_cache: bool = False,
        external_base_audio_teacher: Any | None = None,
    ) -> None:
        if not isinstance(plan, StreamPlan):
            raise TypeError("plan must be a StreamPlan")
        if not plan.phases:
            raise ValueError("streaming plan has no phases")
        if tuple(phase.index for phase in plan.phases) != tuple(
            range(len(plan.phases))
        ):
            raise ValueError("streaming phase indices are not contiguous")
        if plan.phases[0].frame_start != 0:
            raise ValueError("streaming plan does not start at frame zero")
        if plan.phases[-1].frame_stop != plan.native_frame_count:
            raise ValueError("streaming phases do not cover the native clip")
        for previous, current in zip(plan.phases, plan.phases[1:]):
            if (
                previous.frame_stop != current.frame_start
                or previous.video_latent_stop != current.video_latent_start
                or previous.audio_latent_stop != current.audio_latent_start
            ):
                raise ValueError("streaming phase boundaries are discontinuous")
        self.plan = plan
        self.server_args = server_args
        self.reuse_across_requests = bool(reuse_across_requests)
        self.policy = policy or StreamingInferencePolicy()
        if attention_backend not in {
            "sdpa",
            "flash_attn3",
            "flash_attn3_fp8_segmented",
            "sage_attn",
            "sol_attn",
        }:
            raise ValueError("unsupported Stage3 streaming attention backend")
        self.attention_backend = attention_backend
        self.sol_attn_tau = float(sol_attn_tau)
        self.sol_attn_dense_layers = int(sol_attn_dense_layers)
        self.sol_attn_dense_denoise_steps = int(sol_attn_dense_denoise_steps)
        self.sol_attn_route_current_video = bool(sol_attn_route_current_video)
        self.direct_sage_kv_workspace = bool(direct_sage_kv_workspace)
        if self.direct_sage_kv_workspace and attention_backend != "sage_attn":
            raise ValueError(
                "Direct dense K/V workspace requires SageAttention"
            )
        self.sol_attn_static_summary_cache = bool(
            sol_attn_static_summary_cache
        )
        self.external_base_audio_teacher = external_base_audio_teacher
        if external_base_audio_teacher is not None:
            if not self.policy.base_audio_teacher:
                raise ValueError(
                    "external Base audio teacher requires --base-audio-teacher"
                )
            if not self.policy.canonical_continuation_geometry:
                raise ValueError(
                    "external Base audio teacher requires direct canonical "
                    "continuation geometry"
                )
            if (
                self.policy.hybrid_prefix_prompt_mode != "current"
                or self.policy.hybrid_prefix_audio_prompt_mode != "current"
            ):
                raise ValueError(
                    "external Base audio teacher supports current-only prompts"
                )
        if full_prefix_modality not in FULL_PREFIX_MODALITIES:
            raise ValueError(
                "full-prefix modality must be video_only or audio_video"
            )
        self.full_prefix_modality = full_prefix_modality
        if (
            self.policy.canonical_continuation_geometry
            and not self.reuse_across_requests
        ):
            raise ValueError(
                "canonical continuation geometry requires cross-request KV reuse"
            )
        self.last_execution: StreamingExecution | None = None
        self.executions: list[StreamingExecution] = []
        self._running = False
        self._cache: CleanAVKVCache | None = None
        self._low_cache: CleanAVKVCache | None = None
        self._progressive_hiar_histories: Any | None = None
        self._progressive_predicted_history: Any | None = None
        self._direct_prompt_local_prefix: PromptLocalVideoPrefix | None = None
        self._progressive_hiar_pending_group_latents: Any | None = None
        self._media_time_origin: int | None = None
        self._request_index = 0
        self._native_frame_offset = 0
        self._video_latent_offset = 0
        self._audio_latent_offset = 0
        self._clean_video_condition_rows: Any | None = None
        self._clip_keyframe_deadline: dict[str, int] | None = None
        self._clip_keyframes: _ClipKeyframeState | None = None
        self._clip_tail_start_request = 0
        self._canonical_transport_video_rows: Any | None = None
        self._canonical_transport_audio_rows: Any | None = None
        self._prefix_renorm_video_anchor: tuple[Any, Any] | None = None
        self._full_prefix_memory = (
            None
            if (
                full_prefix_half_life_seconds is None
                or not self.policy.full_prefix_enabled
                or self.policy.progressive_history_kv_mode == "stage_input"
            )
            else create_full_prefix_memory(
                modality=full_prefix_modality,
                half_life_seconds=float(full_prefix_half_life_seconds)
            )
        )
        if self.policy.progressive_history_kv_mode == "stage_input":
            if not self.reuse_across_requests:
                raise ValueError(
                    "Progressive HiAR stage-input KV requires cross-request reuse"
                )
            if full_prefix_half_life_seconds is None:
                raise ValueError(
                    "Progressive HiAR stage-input KV requires full-prefix half-life"
                )
            if full_prefix_modality != FULL_PREFIX_AUDIO_VIDEO:
                raise ValueError(
                    "Progressive HiAR stage-input KV currently requires the "
                    "audio_video full-prefix modality"
                )
            from .hiar_pp import HiARDelayedFullPrefixMemory

            self._progressive_hiar_prefix_memory = HiARDelayedFullPrefixMemory(
                half_life_seconds=float(full_prefix_half_life_seconds),
                lag_blocks=self.policy.full_prefix_lag_blocks,
            )
        else:
            self._progressive_hiar_prefix_memory = None

    @property
    def clean_video_condition_rows(self) -> Any | None:
        """Previous request's terminal clean video latent on CPU."""

        return self._clean_video_condition_rows

    def bind_clip_keyframe_deadline(
        self, *, end_frame: int, end_video_latent: int
    ) -> None:
        """Bind the exclusive global endpoint before the first II2VA request."""

        from .h3_temporal_geometry import native_clip_geometry

        if self._running or self._request_index:
            raise RuntimeError("clip keyframe deadline must be bound before generation")
        geometry = native_clip_geometry(end_frame)
        if geometry.video_latent_units != end_video_latent:
            raise ValueError("clip keyframe deadline differs from native video geometry")
        self._clip_keyframe_deadline = {
            "end_frame": end_frame,
            "end_video_latent": end_video_latent,
        }

    def bind_clip_tail_start_request(self, request_index: int) -> None:
        """Delay tail reference tokens, without changing target noise or the clip clock."""

        if self._running or self._request_index:
            raise RuntimeError("tail visibility must be bound before generation")
        if request_index < 0 or self.policy.keyframe_condition_policy != "clip_absolute_persistent":
            raise ValueError("delayed tail visibility requires persistent clip keyframes and a nonnegative request")
        self._clip_tail_start_request = request_index

    def _resolve_clip_keyframes(
        self, request_video: _RequestVideoState
    ) -> _ClipKeyframeState | None:
        """Retain explicit two-image references independently of generated KV."""

        if self.policy.keyframe_condition_policy != "clip_absolute_persistent":
            return None
        if not self.reuse_across_requests:
            self._clip_keyframes = None
        if self._clip_keyframes is None:
            if request_video.keyframe_frame_indices != (0, -1):
                return None
            if self._clip_keyframe_deadline is None:
                if self.reuse_across_requests:
                    raise ValueError("persistent II2VA needs a whole-clip keyframe deadline")
                self._clip_keyframe_deadline = {
                    "end_frame": self.plan.native_frame_count,
                    "end_video_latent": self.plan.phases[-1].video_latent_stop,
                }
            self._clip_keyframes = _ClipKeyframeState(
                condition_initial_rows=request_video.condition_initial_rows.detach().clone(),
                keyframe_condition_rows=request_video.keyframe_condition_rows.detach().clone(),
            )
        if self._request_index < self._clip_tail_start_request:
            # Keep pre-encoding/RNG consumption identical to the all-visible run.
            # Only first-image rows enter phase packing, denoise and clean commit.
            frame_rows = self._clip_keyframes.keyframe_condition_rows.shape[0] // 2
            return _ClipKeyframeState(
                condition_initial_rows=self._clip_keyframes.condition_initial_rows[:frame_rows],
                keyframe_condition_rows=self._clip_keyframes.keyframe_condition_rows[:frame_rows],
                keyframe_frame_indices=(0,),
            )
        return self._clip_keyframes

    def release_retained_state(self) -> None:
        """Release cross-request CUDA state after the final generated chunk."""

        if self._running:
            raise RuntimeError("cannot release streaming state during generation")
        if self._cache is not None:
            self._cache.clear()
            self._cache = None
        if self._low_cache is not None:
            self._low_cache.clear()
            self._low_cache = None
        if self._progressive_hiar_histories is not None:
            self._progressive_hiar_histories.release()
            self._progressive_hiar_histories = None
        self._progressive_hiar_pending_group_latents = None
        if self._progressive_predicted_history is not None:
            self._progressive_predicted_history.release()
            self._progressive_predicted_history = None
        if self._direct_prompt_local_prefix is not None:
            self._direct_prompt_local_prefix.release()
            self._direct_prompt_local_prefix = None
        self._clean_video_condition_rows = None
        self._clip_keyframes = None
        self._clip_keyframe_deadline = None
        self._canonical_transport_video_rows = None
        self._canonical_transport_audio_rows = None
        self._prefix_renorm_video_anchor = None
        if self._full_prefix_memory is not None:
            self._full_prefix_memory.reset()
        if self._progressive_hiar_prefix_memory is not None:
            self._progressive_hiar_prefix_memory.reset()

    def _begin_direct_prompt_local_prefix(self) -> PromptLocalVideoPrefix | None:
        if not self.reuse_across_requests or self.policy.full_prefix_enabled:
            return None
        temporal_latents = prompt_local_prefix_video_latents()
        if not temporal_latents:
            return None
        if self._direct_prompt_local_prefix is None:
            self._direct_prompt_local_prefix = PromptLocalVideoPrefix(temporal_latents)
        self._direct_prompt_local_prefix.begin_request()
        return self._direct_prompt_local_prefix

    def _renorm_clean_video_rows(self, rows: Any) -> Any:
        """Match generated blocks to persistent block-zero statistics."""

        if not self.policy.prefix_renorm:
            return rows
        current = rows.detach().float()
        mean = current.mean(dim=0, keepdim=True)
        std = current.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        if self._prefix_renorm_video_anchor is None:
            self._prefix_renorm_video_anchor = (mean, std)
            return rows
        anchor_mean, anchor_std = self._prefix_renorm_video_anchor
        normalized = (current - mean).div(std).mul(anchor_std).add(anchor_mean)
        return current.lerp(
            normalized, float(self.policy.prefix_renorm_alpha)
        ).to(dtype=rows.dtype)

    def _apply_cross_chunk_cache_policy(self, cache: CleanAVKVCache) -> None:
        if self.policy.cross_chunk_kv_mode == "full":
            return
        recent = self.policy.recent_prefix_blocks
        if self.policy.use_sink_block:
            cache.retain_sink_and_recent_commits(recent)
        elif recent > 0:
            cache.retain_recent_commits(recent)

    def _apply_request_boundary_audio_policy(
        self,
        cache: CleanAVKVCache,
        *,
        total_video_latents: int,
    ) -> tuple[int, bool, int]:
        """Apply periodic audio-KV reset once per completed request window."""

        request_index = (
            self._request_index
            if self.policy.canonical_continuation_geometry
            else self._video_latent_offset // total_video_latents
        )
        reset_applied = (
            self.reuse_across_requests
            and request_index > 0
            and request_index
            % self.policy.cross_chunk_audio_kv_reset_windows
            == 0
        )
        removed_tokens = cache.drop_audio_history() if reset_applied else 0
        return request_index, reset_applied, removed_tokens

    def _encode_full_prefix(
        self,
        *,
        session: H3StreamingSession,
        model: Any,
        loop_module: Any,
        positive: Any,
        text_token_tags: Any,
        latent_h: int,
        latent_w: int,
        media_time_origin: int,
        canonical_latents: Any | None = None,
    ) -> Any | None:
        """Encode the selected detached latent EMA as ephemeral media K/V."""

        memory = self._full_prefix_memory
        if canonical_latents is None:
            canonical_latents = None if memory is None else memory.view()
        if canonical_latents is None:
            return None
        import torch

        if self.full_prefix_modality == FULL_PREFIX_VIDEO_ONLY:
            if isinstance(canonical_latents, FullPrefixAVLatents):
                raise RuntimeError(
                    "video-only full-prefix memory received audio-video latents"
                )
            prefix_device = canonical_latents.device
        else:
            if not isinstance(canonical_latents, FullPrefixAVLatents):
                raise RuntimeError(
                    "audio-video full-prefix memory requires AV latents"
                )
            prefix_device = canonical_latents.video.device

        # Stage3 trains the detached EMA prefix in the canonical first-block
        # layout.  Continuation target geometry must not move or resize that
        # fixed sink representation.
        phase = self.plan.phases[0]
        frame_rows = (latent_h // 2) * (latent_w // 2)
        branch = _build_phase_branch(
            loop_module=loop_module,
            positive=positive,
            phase=phase,
            latent_h=latent_h,
            latent_w=latent_w,
            model=model,
            device=prefix_device,
            text_token_tags=text_token_tags,
            keyframe_frame_indices=(),
            total_frame_count=self.plan.native_frame_count,
            total_video_latents=self.plan.phases[-1].video_latent_stop,
            media_time_origin=media_time_origin,
            video_latent_offset=0,
            audio_latent_offset=0,
        )
        if self.full_prefix_modality == FULL_PREFIX_VIDEO_ONLY:
            video = place_full_prefix_rows(
                canonical_latents,
                phase=phase,
                frame_rows=frame_rows,
            ).to(device=prefix_device, dtype=torch.float32)
            audio = torch.zeros(
                (2 * phase.audio_latent_count, 32),
                device=prefix_device,
                dtype=torch.float32,
            )
        else:
            placed = place_full_prefix_av_latents(
                canonical_latents,
                phase=phase,
                frame_rows=frame_rows,
            )
            video = placed.video.to(device=prefix_device, dtype=torch.float32)
            audio = placed.audio.to(device=prefix_device, dtype=torch.float32)
        timestep = branch.prepare_timestep_plan(
            video_timesteps=[1.0],
            audio_timesteps=[1.0],
            imgvid_cond_noise_aug=0.999,
            audio_ref_cond_noise_aug=1.0,
        )[0]
        forward_kwargs = branch.forward_kwargs(
            video_rows=video,
            audio_rows=audio,
            step_timesteps=timestep,
        )
        forward_kwargs["streaming_cache_only"] = True
        session.hook.begin_full_prefix_capture(
            video_rows=FULL_PREFIX_VIDEO_LATENT_UNITS * frame_rows,
            audio_rows_per_channel=(
                FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
                if self.full_prefix_modality == FULL_PREFIX_AUDIO_VIDEO
                else None
            ),
        )
        try:
            from digitalhuman_h3.distributed import (
                suspend_fp8_activation_calibration,
            )

            with (
                suspend_fp8_activation_calibration(),
                _native_attention_metadata(branch),
                torch.no_grad(),
            ):
                model(**forward_kwargs)
            return session.hook.finish_full_prefix_capture()
        except BaseException:
            session.hook.deactivate()
            raise

    def _update_full_prefix(
        self,
        clean_phase_video_rows: Any,
        clean_phase_audio_rows: Any | None = None,
        *,
        phase: StreamPhase,
        frame_rows: int,
    ) -> bool:
        """Update the EMA only from a complete ten-unit video block."""

        memory = self._full_prefix_memory
        if memory is None:
            return
        # The EMA has a fixed ten-latent canonical shape.  The five-latent
        # request tail stays available through clean KV but cannot replace it.
        if phase.video_latent_count < FULL_PREFIX_VIDEO_LATENT_UNITS:
            return
        if self.full_prefix_modality == FULL_PREFIX_AUDIO_VIDEO:
            if (
                phase.audio_latent_count
                < FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL - 1
            ):
                return
            canonical = canonical_full_prefix_av_latents(
                clean_phase_video_rows,
                clean_phase_audio_rows,
                phase=phase,
                frame_rows=frame_rows,
            )
        else:
            canonical = canonical_full_prefix_rows(
                clean_phase_video_rows,
                phase=phase,
                frame_rows=frame_rows,
            )
        memory.update(
            canonical,
            block_seconds=float(phase.duration),
        )
        return True

    def _run_progressive(
        self,
        original_loop: Any,
        loop_module: Any,
        result_type: Any,
        **kwargs: Any,
    ) -> Any:
        """Run all three resolution transitions inside one cache transaction."""

        import torch

        if self.policy.keyframe_condition_policy == "clip_absolute_persistent":
            raise ValueError("persistent clip keyframes require single-resolution streaming")
        if self.direct_sage_kv_workspace:
            raise ValueError(
                "Direct dense K/V workspace is unavailable for progressive resolution"
            )

        if (
            self.policy.canonical_continuation_geometry
            and self.policy.progressive_history_kv_mode != "clean_commit"
        ):
            raise RuntimeError(
                "progressive canonical continuation requires clean-commit history KV"
            )

        from digitalhuman_h3.progressive_resolution import (
            lift_video_state_rows,
            resize_video_state_rows,
        )

        if self._running:
            raise RuntimeError("one H3 streaming overlay cannot run concurrent requests")
        self._running = True
        try:
            model = kwargs["model"]
            low_positive = kwargs["low_positive"]
            high_positive = kwargs["high_positive"]
            initial_low_video = _required_tensor(
                kwargs["initial_low_video_rows"], "initial_low_video_rows"
            )
            initial_audio = _required_tensor(
                kwargs["initial_audio_rows"], "initial_audio_rows"
            )
            plan = kwargs["plan"]
            switch_position = int(kwargs["switch_position"])
            request_index = self._request_index
            canonical_geometry = self.policy.canonical_continuation_geometry
            canonical_continuation = canonical_geometry and request_index > 0
            # Keep request zero's full sink geometry. Only split the subsequent
            # targets more finely; native frame/audio offsets do not change.
            continuation_groups = os.environ.get("H3_PROGRESSIVE_CONTINUATION_GROUPS")
            continuation_base_plan = (
                plan_native_stream(self.plan.native_frame_count, groups_per_phase=int(continuation_groups))
                if canonical_continuation and continuation_groups else self.plan
            )
            active_plan = (
                canonical_continuation_plan(
                    continuation_base_plan,
                    request_index=request_index,
                )
                if canonical_continuation
                else self.plan
            )
            official_video_latents = self.plan.phases[-1].video_latent_stop
            official_audio_latents = self.plan.phases[-1].audio_latent_stop
            total_video_latents = active_plan.phases[-1].video_latent_stop
            total_audio_latents = active_plan.phases[-1].audio_latent_stop
            video_transport_prefix_latents = (
                official_video_latents - total_video_latents
            )
            audio_transport_prefix_latents = (
                official_audio_latents - total_audio_latents
            )
            low_frame_rows = int(plan.low.spatial_tokens_per_video_latent)
            high_frame_rows = int(plan.high.spatial_tokens_per_video_latent)
            low_latent_h, low_latent_w = _resolve_spatial_latent_shape(
                low_positive, frame_rows=low_frame_rows
            )
            high_latent_h, high_latent_w = _resolve_spatial_latent_shape(
                high_positive, frame_rows=high_frame_rows
            )
            if (low_latent_h, low_latent_w) != (
                int(plan.low.latent_height),
                int(plan.low.latent_width),
            ) or (high_latent_h, high_latent_w) != (
                int(plan.high.latent_height),
                int(plan.high.latent_width),
            ):
                raise RuntimeError(
                    "progressive streaming branch canvas differs from resolution plan"
                )

            low_request_video = _resolve_request_video_state(
                positive=low_positive,
                initial_video=initial_low_video,
                keyframe_cond_rows=kwargs.get("low_keyframe_cond_rows"),
                frame_rows=low_frame_rows,
                total_video_latents=official_video_latents,
            )
            high_template = torch.zeros(
                int(high_positive.img_pos.numel()),
                96,
                dtype=torch.float32,
                device="cpu",
            )
            high_request_video = _resolve_request_video_state(
                positive=high_positive,
                initial_video=high_template,
                keyframe_cond_rows=kwargs.get("high_keyframe_cond_rows"),
                frame_rows=high_frame_rows,
                total_video_latents=official_video_latents,
            )
            if (
                low_request_video.task != high_request_video.task
                or low_request_video.keyframe_frame_indices
                != high_request_video.keyframe_frame_indices
            ):
                raise RuntimeError(
                    "progressive low/high streaming conditioning contracts differ"
                )
            if list(initial_audio.shape) != [2 * official_audio_latents, 32]:
                raise ValueError(
                    "request audio rows do not match cumulative 40 Hz phase boundaries"
                )
            if initial_low_video.device.type != "cpu":
                raise ValueError("official H3 initial noise must enter denoise from CPU")

            official_initial_audio = initial_audio
            transport_video_rows = None
            transport_audio_rows = None
            if canonical_continuation:
                transport_video_rows = self._canonical_transport_video_rows
                transport_audio_rows = self._canonical_transport_audio_rows
                if (
                    transport_video_rows is None
                    or list(transport_video_rows.shape)
                    != [video_transport_prefix_latents * high_frame_rows, 96]
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior progressive "
                        "video transport prefix"
                    )
                if transport_audio_rows is None:
                    raise RuntimeError(
                        "canonical continuation has no prior audio transport prefix"
                    )
                previous_audio_t = int(transport_audio_rows.shape[0]) // 2
                if list(transport_audio_rows.shape) != [2 * previous_audio_t, 32] or (
                    previous_audio_t < audio_transport_prefix_latents
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior audio transport prefix"
                    )
                low_request_video = _RequestVideoState(
                    task=low_request_video.task,
                    target_initial_rows=low_request_video.target_initial_rows[
                        video_transport_prefix_latents * low_frame_rows :
                    ].contiguous(),
                    condition_initial_rows=low_request_video.condition_initial_rows,
                    keyframe_condition_rows=low_request_video.keyframe_condition_rows,
                    keyframe_frame_indices=low_request_video.keyframe_frame_indices,
                )
                initial_audio = (
                    initial_audio.view(2, official_audio_latents, -1)[
                        :, audio_transport_prefix_latents:
                    ]
                    .contiguous()
                    .view(-1, int(initial_audio.shape[-1]))
                )

            sigmas_video = list(kwargs["sigmas_video"])
            sigmas_audio = list(kwargs["sigmas_audio"])
            if (
                len(sigmas_video) != len(sigmas_audio)
                or not 0 < switch_position < len(sigmas_video) - 1
            ):
                raise ValueError("progressive streaming sigma schedule is invalid")
            low_sigmas_video = sigmas_video[: switch_position + 1]
            low_sigmas_audio = sigmas_audio[: switch_position + 1]
            high_sigmas_video = sigmas_video[switch_position:]
            high_sigmas_audio = sigmas_audio[switch_position:]

            phases = active_plan.phases
            callbacks = (
                kwargs.get("on_low_step"),
                kwargs.get("on_resolution_lift"),
                kwargs.get("on_high_step"),
            )
            if len(phases) > 1 and any(callback is not None for callback in callbacks):
                raise NotImplementedError(
                    "progressive trajectory observation runs on the native full clip, "
                    "not the temporal streaming overlay"
                )

            low_text_token_tags = _resolve_global_text_token_tags(low_positive)
            high_text_token_tags = _resolve_global_text_token_tags(high_positive)
            if not torch.equal(low_text_token_tags, high_text_token_tags):
                raise RuntimeError("progressive low/high text token tags differ")
            text_len = int(
                _required_tensor(
                    high_positive.static_kwargs.get("prompt_embeds"),
                    "high_positive.static_kwargs['prompt_embeds']",
                ).shape[0]
            )
            media_time_origin = (
                text_len
                if self._media_time_origin is None or not self.reuse_across_requests
                else self._media_time_origin
            )
            temporal_alignment = (
                _request_temporal_alignment(
                    current_text_len=text_len,
                    fixed_media_origin=media_time_origin,
                    total_frame_count=active_plan.native_frame_count,
                    total_video_latents=total_video_latents,
                    video_latent_offset=self._video_latent_offset,
                    audio_latent_offset=self._audio_latent_offset,
                )
                if not canonical_geometry
                and active_plan.native_frame_count == 243
                and total_video_latents == 72
                and len(self.plan.phases) == 1
                else None
            )
            cuda_device = torch.device("cuda")
            base_model_forward = kwargs.get("model_forward")
            pp = None
            if os.environ.get("H3_PROGRESSIVE_PP_STAGE") is not None:
                if os.environ.get('H3_PIPELINE_PREWARM') == '1' and request_index == 0:
                    from .pipeline_warmup import warm_stage
                    warm_stage_index = int(os.environ['H3_PROGRESSIVE_PP_STAGE'])
                    warm_high = warm_stage_index == 2
                    warm_positive = high_positive if warm_high else low_positive
                    warm_h, warm_w = (high_latent_h, high_latent_w) if warm_high else (low_latent_h, low_latent_w)
                    warm_rows = high_frame_rows if warm_high else low_frame_rows
                    warm_branch = _build_phase_branch(
                        loop_module=loop_module, positive=warm_positive, phase=phases[0],
                        latent_h=warm_h, latent_w=warm_w, model=model, device=cuda_device,
                        text_token_tags=high_text_token_tags if warm_high else low_text_token_tags,
                        keyframe_frame_indices=(), total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents, media_time_origin=media_time_origin,
                        video_latent_offset=0, audio_latent_offset=0,
                        phase_tail_condition=False, canonical_continuation_geometry=canonical_geometry,
                    )
                    warm_video, warm_audio = _phase_initial_rows(
                        initial_video_rows=(high_template if warm_high else low_request_video.target_initial_rows),
                        initial_audio_rows=initial_audio, phase=phases[0], frame_rows=warm_rows,
                        total_audio_latents=total_audio_latents,
                    )
                    warm_stage(
                        owner=self, model=model, branch=warm_branch, phase=phases[0],
                        active_plan=active_plan, video=warm_video, audio=warm_audio,
                        video_sigmas=sigmas_video[warm_stage_index:warm_stage_index+2],
                        audio_sigmas=sigmas_audio[warm_stage_index:warm_stage_index+2],
                        original_loop=original_loop, loop_module=loop_module,
                        model_forward=base_model_forward,
                        directory=os.environ['H3_PIPELINE_WARMUP_DIR'],
                    )
                from .progressive_pipeline import ProgressivePipelineStage
                pp = getattr(self, "_progressive_pipeline", None)
                if pp is None:
                    pp = self._progressive_pipeline = ProgressivePipelineStage(
                        cross_request_visual_handoff=self.policy.cross_request_visual_handoff,
                    )
                if (
                    self.policy.full_prefix_enabled
                    or self.external_base_audio_teacher is None
                    or os.environ.get("H3_PROGRESSIVE_STAGE_LOCAL_PREDICTED_KV") != "1"
                ):
                    raise ValueError("Progressive PP uses native teacher, stage-local low KV and no full-prefix")
                original_model_forward = base_model_forward
                def traced_model_forward(active_model, call_kwargs, step_index):
                    clean_forward = bool(call_kwargs.get("streaming_cache_only")) or step_index == len(sigmas_video) - 1
                    label = "clean" if clean_forward else f"dit_step_{step_index}"
                    fused_low_clean = bool(
                        not clean_forward and predicted_history is not None
                        and step_index in predicted_history.deferred
                    )
                    with pp.compute(label, request=int(request_index), phase=int(phase.index),
                                    fused_low_clean=fused_low_clean):
                        if original_model_forward is None:
                            return active_model(**call_kwargs)
                        return original_model_forward(active_model, call_kwargs, step_index)
                base_model_forward = traced_model_forward
            progressive_hiar = (
                self.policy.progressive_history_kv_mode == "stage_input"
            )
            dual_clean = self.policy.progressive_history_kv_mode == "dual_clean"
            low_cache = None
            low_starting_history_tokens = 0
            stage_predicted_kv = os.environ.get("H3_PROGRESSIVE_STAGE_LOCAL_PREDICTED_KV") == "1"
            predicted_history = None
            if stage_predicted_kv:
                if self.policy.progressive_history_kv_mode != "clean_commit" or self.policy.full_prefix_enabled:
                    raise ValueError("stage-local predicted KV requires clean-commit without full-prefix")
                from .progressive_predicted_history import ProgressivePredictedHistory
                if self._progressive_predicted_history is None:
                    self._progressive_predicted_history = ProgressivePredictedHistory(
                        model, owned_stage=None if pp is None else pp.stage,
                    )
                predicted_history = self._progressive_predicted_history
                predicted_history.begin_request(owner=self, plan=active_plan, model=model, total_video_latents=total_video_latents)
            low_early_send = predicted_history is not None and predicted_history.early_send
            boundary_reset_tokens = 0
            boundary_reset_applied = False
            if progressive_hiar:
                from .progressive_hiar import ProgressiveHiARStageHistories

                if self._progressive_hiar_histories is None:
                    self._progressive_hiar_histories = (
                        ProgressiveHiARStageHistories(
                            model=model,
                            max_history_blocks=(
                                self.policy.progressive_history_kv_max_blocks
                            ),
                            attention_backend=self.attention_backend,
                        )
                    )
                hiar_histories = self._progressive_hiar_histories
                cache = None
                audio_kv_reset_applied = False
                audio_kv_reset_tokens = 0
                starting_history_tokens = sum(hiar_histories.history_tokens)
                starting_history_audio_tokens = sum(
                    hiar_histories.history_audio_tokens
                )
                starting_history_video_tokens = sum(
                    hiar_histories.history_video_tokens
                )
                self._progressive_hiar_pending_group_latents = None
            else:
                hiar_histories = None
                cache = self._cache
                if cache is None:
                    cache = CleanAVKVCache(
                        kv_contract_from_model(model),
                        segmented_fp8_history=(
                            self.attention_backend
                            in {
                                "flash_attn3_fp8_segmented",
                            }
                        ),
                    )
                    if self.reuse_across_requests:
                        self._cache = cache
                pre_boundary_request_index = (
                    request_index
                    if canonical_geometry
                    else self._video_latent_offset // total_video_latents
                )
                if (
                    self.policy.cross_request_kv_boundary_reset
                    and pre_boundary_request_index > 0
                ):
                    boundary_reset_tokens = (
                        cache.discard_retained_history_preserve_serial()
                    )
                    boundary_reset_applied = True
                (
                    request_index,
                    audio_kv_reset_applied,
                    audio_kv_reset_tokens,
                ) = self._apply_request_boundary_audio_policy(
                    cache,
                    total_video_latents=total_video_latents,
                )
                starting_history_tokens = cache.history_tokens
                starting_history_audio_tokens = cache.history_audio_tokens
                starting_history_video_tokens = cache.history_video_tokens
                if dual_clean:
                    low_cache = self._low_cache
                    if low_cache is None:
                        low_cache = CleanAVKVCache(
                            kv_contract_from_model(model),
                            segmented_fp8_history=cache.segmented_fp8_history,
                        )
                        if self.reuse_across_requests:
                            self._low_cache = low_cache
                    if boundary_reset_applied:
                        boundary_reset_tokens += (
                            low_cache.discard_retained_history_preserve_serial()
                        )
                    _, _, low_audio_reset_tokens = self._apply_request_boundary_audio_policy(
                        low_cache, total_video_latents=total_video_latents
                    )
                    audio_kv_reset_tokens += low_audio_reset_tokens
                    low_starting_history_tokens = low_cache.history_tokens
                    starting_history_tokens += low_cache.history_tokens
                    starting_history_audio_tokens += low_cache.history_audio_tokens
                    starting_history_video_tokens += low_cache.history_video_tokens
            reused_kv = starting_history_tokens > 0
            current_prompt_rope_start = _current_prompt_rope_start(
                text_len=text_len,
                media_time_origin=media_time_origin,
                video_latent_offset=self._video_latent_offset,
                total_video_latents=total_video_latents,
                canonical_continuation_geometry=canonical_geometry,
            )
            direct_high_condition_rows = None
            direct_low_condition_rows = None
            reused_clean_video = False
            if (
                self.reuse_across_requests
                and self._clean_video_condition_rows is not None
                and high_request_video.task == "fl2va_first_frame"
            ):
                direct_high_condition_rows = self._clean_video_condition_rows
                if list(direct_high_condition_rows.shape) != [high_frame_rows, 96]:
                    raise RuntimeError(
                        "cross-request clean video latent changed high-resolution geometry"
                    )
                direct_low_condition_rows = resize_video_state_rows(
                    direct_high_condition_rows,
                    temporal_length=1,
                    source_canvas=plan.high,
                    target_canvas=plan.low,
                )
                reused_clean_video = True

            def resolve_request_teacher():
                return _resolve_base_audio_teacher(
                    external_artifact=self.external_base_audio_teacher,
                    request_index=request_index,
                    torch_module=torch,
                    device=cuda_device,
                    run_internal=lambda: _run_base_audio_teacher(
                        enabled=self.policy.base_audio_teacher,
                        original_loop=original_loop,
                        loop_module=loop_module,
                        model=model,
                        positive=high_positive,
                        initial_audio_rows=official_initial_audio,
                        total_audio_latents=official_audio_latents,
                        latent_h=high_latent_h,
                        latent_w=high_latent_w,
                        device=cuda_device,
                    ),
                )
            defer_teacher = (getattr(self.external_base_audio_teacher, 'live', False)
                             and pp is not None and pp.stage == 0)
            independent_stage0 = defer_teacher and os.environ.get('H3_PROGRESSIVE_STAGE0_FREE_RUN') == '1'
            audio_teacher = None if defer_teacher else resolve_request_teacher()
            audio_teacher = _active_base_audio_teacher(
                audio_teacher,
                official_audio_latents=official_audio_latents,
                active_audio_latents=total_audio_latents,
                transport_prefix_audio_latents=audio_transport_prefix_latents,
            )
            if defer_teacher:
                from digitalhuman_h3.inference.live_progressive_teacher import DeferredTeacher
                audio_teacher = DeferredTeacher(lambda: _active_base_audio_teacher(
                    resolve_request_teacher(), official_audio_latents=official_audio_latents,
                    active_audio_latents=total_audio_latents,
                    transport_prefix_audio_latents=audio_transport_prefix_latents,
                ))

            high_video_outputs: list[Any] = []
            high_audio_outputs: list[Any] = []
            lifted_outputs: list[Any] = []
            phase_records: list[StreamingPhaseExecution] = []
            position_capture: dict[str, object] | None = None
            previous_high_tail = None
            previous_low_tail = None
            hiar_visible_prefix_source: int | None = None
            def finish_low_pipeline_phase(phase):
                nonlocal previous_high_tail, previous_low_tail
                needs_tail = phase.index + 1 < len(phases) and self.policy.uses_phase_latent_handoff(phase.index + 1)
                clean_video, clean_audio = pp.finish_low_phase(
                    phase=phase, request=request_index, last_phase=len(phases)-1,
                    needs_next_tail=needs_tail, frame_rows=high_frame_rows,
                    teacher_audio=None if independent_stage0 else _teacher_phase_audio_rows(
                        audio_teacher["milestones"][2], phase=phase, total_audio_latents=total_audio_latents,
                    ),
                )
                if needs_tail:
                    previous_high_tail = clean_video[-high_frame_rows:].detach().cpu().contiguous()
                    previous_low_tail = resize_video_state_rows(
                        previous_high_tail, temporal_length=1, source_canvas=plan.high, target_canvas=plan.low,
                    ).contiguous()
                high_video_outputs.append(clean_video)
                high_audio_outputs.append(clean_audio)
                lifted_outputs.append(clean_video)
                phase_records.append(StreamingPhaseExecution(
                    phase_index=phase.index, frame_start=phase.frame_start, frame_stop=phase.frame_stop,
                    video_latent_start=phase.video_latent_start, video_latent_stop=phase.video_latent_stop,
                    audio_latent_start=phase.audio_latent_start, audio_latent_stop=phase.audio_latent_stop,
                    denoise_forwards=1,
                    clean_forwards=predicted_history.separate_clean_count(phase.index, len(phases)-1),
                    committed_history_tokens=predicted_history.caches[pp.stage].history_tokens,
                    phase_latent_handoff_used=self.policy.uses_phase_latent_handoff(phase.index),
                ))
            session_context = (
                nullcontext(None)
                if progressive_hiar
                else H3StreamingSession(
                    active_plan,
                    model=model,
                    server_args=self.server_args,
                    cache=cache,
                    attention_backend=os.environ.get("H3_PROGRESSIVE_HIGH_ATTENTION_BACKEND", self.attention_backend),
                    sol_attn_tau=self.sol_attn_tau,
                    sol_attn_dense_layers=self.sol_attn_dense_layers,
                    sol_attn_dense_denoise_steps=(
                        self.sol_attn_dense_denoise_steps
                    ),
                    sol_attn_route_current_video=(
                        self.sol_attn_route_current_video
                    ),
                    sol_attn_static_summary_cache=(
                        self.sol_attn_static_summary_cache
                    ),
                    fused_history_use_sink=self.policy.use_sink_block,
                    fused_history_recent_blocks=self.policy.cross_chunk_kv_recent_blocks,
                )
            )
            low_session = (
                H3StreamingSession(
                    active_plan,
                    model=model,
                    server_args=self.server_args,
                    cache=low_cache,
                    attention_backend=self.attention_backend,
                    sol_attn_tau=self.sol_attn_tau,
                    sol_attn_dense_layers=self.sol_attn_dense_layers,
                    sol_attn_dense_denoise_steps=self.sol_attn_dense_denoise_steps,
                    sol_attn_route_current_video=self.sol_attn_route_current_video,
                    sol_attn_static_summary_cache=self.sol_attn_static_summary_cache,
                    fused_history_use_sink=self.policy.use_sink_block,
                    fused_history_recent_blocks=self.policy.cross_chunk_kv_recent_blocks,
                )
                if dual_clean else None
            )
            with session_context as session:
                for phase in phases:
                    skip_unused_low_prep = (
                        pp is not None and pp.stage == 2
                        and os.environ.get('H3_HIGH_SKIP_UNUSED_LOW_PREP') == '1'
                    )
                    use_phase_latent_tail = self.policy.uses_phase_latent_handoff(
                        phase.index
                    )
                    phase_keyframe_indices = (
                        (-1,)
                        if use_phase_latent_tail
                        else (
                            high_request_video.keyframe_frame_indices
                            if high_request_video.task != "t2va"
                            and phase.index == 0
                            else ()
                        )
                    )
                    if progressive_hiar:
                        committed = hiar_histories.committed_blocks
                        if len(set(committed)) != 1:
                            raise RuntimeError(
                                "Progressive HiAR stage histories lost block alignment"
                            )
                        global_block_index = committed[0]
                        prefix_memory = self._progressive_hiar_prefix_memory
                        if prefix_memory is None:
                            raise RuntimeError(
                                "Progressive HiAR delayed full-prefix is absent"
                            )
                        prefix_memory.advance(global_block_index)
                        hiar_visible_prefix_source = (
                            prefix_memory.visible_source_block_index
                        )
                        canonical_prefix_latents = prefix_memory.view()
                        if canonical_prefix_latents is None:
                            full_prefix = None
                        else:
                            with hiar_histories.install_stage(2) as prefix_session:
                                full_prefix = self._encode_full_prefix(
                                    session=prefix_session,
                                    model=model,
                                    loop_module=loop_module,
                                    positive=high_positive,
                                    text_token_tags=high_text_token_tags,
                                    latent_h=high_latent_h,
                                    latent_w=high_latent_w,
                                    media_time_origin=media_time_origin,
                                    canonical_latents=canonical_prefix_latents,
                                )
                    else:
                        global_block_index = phase.index
                        full_prefix = self._encode_full_prefix(
                            session=session,
                            model=model,
                            loop_module=loop_module,
                            positive=high_positive,
                            text_token_tags=high_text_token_tags,
                            latent_h=high_latent_h,
                            latent_w=high_latent_w,
                            media_time_origin=media_time_origin,
                        )
                    low_branch = None if skip_unused_low_prep else _build_phase_branch(
                        loop_module=loop_module,
                        positive=low_positive,
                        phase=phase,
                        latent_h=low_latent_h,
                        latent_w=low_latent_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=low_text_token_tags,
                        keyframe_frame_indices=phase_keyframe_indices,
                        total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents,
                        media_time_origin=media_time_origin,
                        video_latent_offset=self._video_latent_offset,
                        audio_latent_offset=self._audio_latent_offset,
                        phase_tail_condition=use_phase_latent_tail,
                        canonical_continuation_geometry=canonical_geometry,
                    )
                    low_video, phase_audio = _phase_initial_rows(
                        initial_video_rows=low_request_video.target_initial_rows,
                        initial_audio_rows=initial_audio,
                        phase=phase,
                        frame_rows=low_frame_rows,
                        total_audio_latents=total_audio_latents,
                    )
                    low_keyframe_rows = None
                    if phase_keyframe_indices and not skip_unused_low_prep:
                        low_condition_initial = (
                            previous_low_tail
                            if use_phase_latent_tail
                            else low_request_video.condition_initial_rows
                        )
                        if low_condition_initial is None:
                            raise RuntimeError(
                                "progressive phase latent-tail condition is absent"
                            )
                        low_condition_initial = low_condition_initial.to(
                            device=low_video.device,
                            dtype=low_video.dtype,
                        )
                        low_video = torch.cat(
                            (low_condition_initial, low_video), dim=0
                        )
                        low_keyframe_rows = (
                            low_condition_initial
                            if use_phase_latent_tail
                            else (
                                direct_low_condition_rows
                                if direct_low_condition_rows is not None
                                else low_request_video.keyframe_condition_rows
                            )
                        )

                    def low_noisy_forward(
                        active_model: Any,
                        call_kwargs: dict[str, Any],
                        step_index: int,
                    ) -> tuple[Any, Any]:
                        if pp is not None:
                            step_index += pp.stage
                        if progressive_hiar and step_index not in {0, 1}:
                            raise RuntimeError(
                                "Progressive HiAR low loop must contain x0 and x16"
                            )
                        if predicted_history is not None:
                            return predicted_history.forward(
                                step_index, phase.index, owner=self, branch=low_branch,
                                model=active_model, call_kwargs=call_kwargs,
                                model_forward=base_model_forward, metadata=_native_attention_metadata,
                                restore_hook=session.hook, full_prefix=full_prefix,
                                video_sigmas=low_sigmas_video, audio_sigmas=low_sigmas_audio,
                                condition_timestep=1.0 if reused_clean_video or use_phase_latent_tail else 0.999,
                            )
                        history_context = (
                            hiar_histories.denoise_and_commit_input(
                                stage_index=step_index,
                                block_index=global_block_index,
                                full_prefix=full_prefix,
                            )
                            if progressive_hiar
                            else (low_session if dual_clean else session).noisy_step(
                                phase.index,
                                full_prefix=full_prefix,
                                denoise_step_index=step_index,
                            )
                        )
                        with (
                            resolution_local_session(low_session, restore=session)
                            if dual_clean else nullcontext(),
                            history_context,
                            _native_attention_metadata(low_branch),
                        ):
                            if base_model_forward is None:
                                return active_model(**call_kwargs)
                            return base_model_forward(
                                active_model, call_kwargs, step_index
                            )

                    low_on_step = None if skip_unused_low_prep else _base_audio_teacher_step_callback(
                        audio_teacher,
                        phase=phase,
                        total_audio_latents=total_audio_latents,
                        audio_target_slice=low_branch.audio_target_slice,
                        step_offset=0 if pp is None else pp.stage,
                        outer=kwargs.get("on_low_step"),
                    )
                    # LOW0 can send video before its teacher-clean audio history is ready.
                    if independent_stage0:
                        low_on_step = None

                    low_clean_capture = (
                        NativeLowCleanCapture(
                            target_slice=low_branch.video_target_slice,
                            sigmas=low_sigmas_video,
                        )
                        if dual_clean else None
                    )
                    low_capture_callbacks: dict[str, Any] = {}
                    if dual_clean:
                        low_capture_callbacks = {
                            "on_initial_state": low_clean_capture.on_initial_state,
                            "on_velocity": low_clean_capture.on_velocity,
                        }
                    elif predicted_history is not None and not skip_unused_low_prep:
                        clean_audio_callback = _base_audio_teacher_step_callback(
                            audio_teacher, phase=phase, total_audio_latents=total_audio_latents,
                            audio_target_slice=low_branch.audio_target_slice,
                        )
                        low_capture_callbacks = {
                            "on_initial_state": predicted_history.initial,
                            "on_velocity": lambda step, vv, av: predicted_history.capture(
                                step, vv, av, branch=low_branch,
                                video_sigmas=low_sigmas_video, audio_sigmas=low_sigmas_audio,
                                clean_audio_callback=clean_audio_callback,
                                step_offset=0 if pp is None else pp.stage,
                            ),
                        }
                    if pp is not None and pp.stage == 2:
                        generated_low_audio = _teacher_phase_audio_rows(
                            audio_teacher["milestones"][1], phase=phase,
                            total_audio_latents=total_audio_latents,
                        )
                        lifted_target = pp.receive(
                            "input", request_index, phase.index,
                            (phase.video_latent_count * high_frame_rows, 96),
                        )
                    else:
                        if pp is not None and pp.stage == 1:
                            phase_audio = _teacher_phase_audio_rows(
                                audio_teacher["milestones"][0], phase=phase,
                                total_audio_latents=total_audio_latents,
                            )
                            low_video = pp.receive("input", request_index, phase.index, tuple(low_video.shape))
                        owned_slice = slice(None) if pp is None else slice(pp.stage, pp.stage + 2)
                        generated_low_video, generated_low_audio = original_loop(
                            model=model,
                            model_forward=low_noisy_forward,
                            positive=low_branch,
                            initial_video_rows=low_video,
                            initial_audio_rows=phase_audio,
                            keyframe_cond_rows=low_keyframe_rows,
                            audio_ref_rows=None,
                            sigmas_video=low_sigmas_video[owned_slice],
                            sigmas_audio=low_sigmas_audio[owned_slice],
                            device=cuda_device,
                            imgvid_cond_noise_aug_for_inference=(
                                1.0
                                if reused_clean_video or use_phase_latent_tail
                                else 0.999
                            ),
                            audio_cond_noise_aug_for_inference=1.0,
                            on_step=low_on_step,
                            **low_capture_callbacks,
                        )
                        if predicted_history is not None:
                            low_history_commit = dict(
                                owner=self, phase=phase.index, branch=low_branch, model=model,
                                restore_hook=session.hook, full_prefix=full_prefix,
                                condition_timestep=1.0 if reused_clean_video or use_phase_latent_tail else 0.999,
                                model_forward=base_model_forward, metadata=_native_attention_metadata,
                            )
                            if not low_early_send:
                                predicted_history.commit(**low_history_commit)
                        if pp is not None and pp.stage == 0:
                            pp.send("next", generated_low_video, request_index, phase.index)
                            if low_early_send:
                                predicted_history.commit(**low_history_commit)
                            finish_low_pipeline_phase(phase)
                            continue
                        lift_function = lift_video_state_rows
                        lift_extra = {}
                        if os.environ.get("H3_PROGRESSIVE_REUSE_LIFT_NOISE") == "1":
                            from .progressive_noise_reuse import lift_with_request_noise
                            lift_function = lift_with_request_noise
                            high_noise = high_positive._digitalhuman_h3_coupled_initial_video_target_rows
                            high_start = video_transport_prefix_latents + phase.video_latent_start
                            lift_extra = {
                                "low_noise_rows": low_request_video.target_initial_rows[
                                    phase.video_latent_start*low_frame_rows:phase.video_latent_stop*low_frame_rows
                                ],
                                "high_noise_rows": high_noise[
                                    high_start*high_frame_rows:(high_start+phase.video_latent_count)*high_frame_rows
                                ],
                            }
                        lifted_target, lift_receipt = lift_function(
                            generated_low_video[low_branch.video_target_slice],
                            temporal_length=phase.video_latent_count,
                            temporal_offset=(
                                video_transport_prefix_latents
                                + phase.video_latent_start
                            ),
                            source_canvas=plan.low,
                            target_canvas=plan.high,
                            video_sigma=float(sigmas_video[switch_position]),
                            noise_seed=int(kwargs["lift_noise_seed"]),
                            **lift_extra,
                        )
                    if pp is not None and pp.stage == 1:
                        pp.send("next", lifted_target, request_index, phase.index)
                        if low_early_send:
                            predicted_history.commit(**low_history_commit)
                        finish_low_pipeline_phase(phase)
                        continue
                    lifted_outputs.append(lifted_target)
                    if pp is None and kwargs.get("on_resolution_lift") is not None:
                        kwargs["on_resolution_lift"](
                            lifted_target, generated_low_audio, lift_receipt
                        )

                    high_branch = _build_phase_branch(
                        loop_module=loop_module,
                        positive=high_positive,
                        phase=phase,
                        latent_h=high_latent_h,
                        latent_w=high_latent_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=high_text_token_tags,
                        keyframe_frame_indices=phase_keyframe_indices,
                        total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents,
                        media_time_origin=media_time_origin,
                        video_latent_offset=self._video_latent_offset,
                        audio_latent_offset=self._audio_latent_offset,
                        phase_tail_condition=use_phase_latent_tail,
                        canonical_continuation_geometry=canonical_geometry,
                    )
                    branch_capture = getattr(
                        high_branch, "_digitalhuman_h3_position_capture", None
                    )
                    if branch_capture is not None:
                        if position_capture is not None:
                            raise RuntimeError(
                                "bounded RoPE exact progressive request produced "
                                "multiple high-resolution captures"
                            )
                        position_capture = dict(branch_capture)
                    high_video = lifted_target
                    high_keyframe_rows = None
                    if phase_keyframe_indices:
                        high_condition_initial = (
                            previous_high_tail
                            if use_phase_latent_tail
                            else high_request_video.condition_initial_rows
                        )
                        if high_condition_initial is None:
                            raise RuntimeError(
                                "progressive high-resolution latent tail is absent"
                            )
                        high_condition_initial = high_condition_initial.to(
                            device=high_video.device,
                            dtype=high_video.dtype,
                        )
                        high_video = torch.cat(
                            (high_condition_initial, high_video),
                            dim=0,
                        )
                        high_keyframe_rows = (
                            high_condition_initial
                            if use_phase_latent_tail
                            else (
                                direct_high_condition_rows
                                if direct_high_condition_rows is not None
                                else high_request_video.keyframe_condition_rows
                            )
                        )

                    def high_noisy_forward(
                        active_model: Any,
                        call_kwargs: dict[str, Any],
                        step_index: int,
                    ) -> tuple[Any, Any]:
                        if progressive_hiar and step_index != 0:
                            raise RuntimeError(
                                "Progressive HiAR high loop must contain lifted x33"
                            )
                        history_context = (
                            hiar_histories.denoise_and_commit_input(
                                stage_index=2,
                                block_index=global_block_index,
                                full_prefix=full_prefix,
                            )
                            if progressive_hiar
                            else session.noisy_step(
                                phase.index,
                                full_prefix=full_prefix,
                                denoise_step_index=switch_position + step_index,
                            )
                        )
                        with (
                            history_context,
                            _native_attention_metadata(high_branch),
                        ):
                            if base_model_forward is None:
                                return active_model(**call_kwargs)
                            return base_model_forward(
                                active_model,
                                call_kwargs,
                                switch_position + step_index,
                            )

                    high_on_step = _base_audio_teacher_step_callback(
                        audio_teacher,
                        phase=phase,
                        total_audio_latents=total_audio_latents,
                        audio_target_slice=high_branch.audio_target_slice,
                        step_offset=switch_position,
                        outer=kwargs.get("on_high_step"),
                    )

                    generated_high_video, generated_high_audio = original_loop(
                        model=model,
                        model_forward=high_noisy_forward,
                        positive=high_branch,
                        initial_video_rows=high_video,
                        initial_audio_rows=generated_low_audio,
                        keyframe_cond_rows=high_keyframe_rows,
                        audio_ref_rows=None,
                        sigmas_video=high_sigmas_video,
                        sigmas_audio=high_sigmas_audio,
                        device=cuda_device,
                        imgvid_cond_noise_aug_for_inference=(
                            1.0
                            if reused_clean_video or use_phase_latent_tail
                            else 0.999
                        ),
                        audio_cond_noise_aug_for_inference=1.0,
                        on_step=high_on_step,
                    )

                    clean_phase_video = self._renorm_clean_video_rows(
                        generated_high_video[high_branch.video_target_slice],
                    )
                    generated_high_video[high_branch.video_target_slice].copy_(
                        clean_phase_video
                    )
                    if pp is not None:
                        pp.feedback(
                            clean_phase_video, request=request_index, phase=phase.index,
                            last_phase=len(phases)-1, frame_rows=high_frame_rows,
                            needs_next_tail=phase.index+1 < len(phases) and self.policy.uses_phase_latent_handoff(phase.index+1),
                        )
                        pp.publish_phase(clean_phase_video, generated_high_audio, request_index, phase.index)
                    if progressive_hiar:
                        from .progressive_hiar import (
                            canonical_high_clean_prefix_update,
                        )

                        if hiar_histories.committed_blocks != (
                            global_block_index + 1,
                            global_block_index + 1,
                            global_block_index + 1,
                        ):
                            raise RuntimeError(
                                "Progressive HiAR did not commit all three stage inputs"
                            )
                        prefix_update, pending_group_latents = (
                            canonical_high_clean_prefix_update(
                                plan=self.plan,
                                phase_index=phase.index,
                                clean_video_rows=clean_phase_video,
                                clean_audio_rows=generated_high_audio,
                                frame_rows=high_frame_rows,
                                pending_one_group_latents=(
                                    self._progressive_hiar_pending_group_latents
                                ),
                            )
                        )
                        self._progressive_hiar_pending_group_latents = (
                            pending_group_latents
                        )
                        if prefix_update is not None:
                            self._progressive_hiar_prefix_memory.enqueue(
                                global_block_index,
                                prefix_update.canonical_latents,
                                block_seconds=prefix_update.block_seconds,
                            )
                    else:
                        history_before_commit = session.cache.history_tokens
                        current_commit_tokens = (
                            phase.video_latent_count * high_frame_rows
                            + 2 * phase.audio_latent_count
                        )
                        expected_history_tokens = (
                            history_before_commit + current_commit_tokens
                        )
                        clean_timestep = high_branch.prepare_timestep_plan(
                            video_timesteps=[1.0],
                            audio_timesteps=[1.0],
                            imgvid_cond_noise_aug=(
                                1.0
                                if reused_clean_video or use_phase_latent_tail
                                else 0.999
                            ),
                            audio_ref_cond_noise_aug=1.0,
                        )[0]
                        clean_kwargs = high_branch.forward_kwargs(
                            video_rows=generated_high_video,
                            audio_rows=generated_high_audio,
                            step_timesteps=clean_timestep,
                        )
                        with (
                            session.clean_commit(
                                phase.index, full_prefix=full_prefix
                            ),
                            _native_attention_metadata(high_branch),
                            torch.no_grad(),
                        ):
                            if base_model_forward is None:
                                model(**clean_kwargs)
                            else:
                                base_model_forward(
                                    model, clean_kwargs, len(sigmas_video) - 1
                                )
                        if session.cache.history_tokens != expected_history_tokens:
                            raise RuntimeError(
                                "progressive clean H3 KV history differs from the "
                                "high-resolution phase boundary"
                            )
                        if dual_clean:
                            low_history_before = low_cache.history_tokens
                            low_clean_video = generated_low_video.clone()
                            low_clean_video[low_branch.video_target_slice].copy_(
                                low_clean_capture.clean_video_rows
                            )
                            low_clean_timestep = low_branch.prepare_timestep_plan(
                                video_timesteps=[1.0],
                                audio_timesteps=[1.0],
                                imgvid_cond_noise_aug=(1.0 if reused_clean_video else 0.999),
                                audio_ref_cond_noise_aug=1.0,
                            )[0]
                            low_clean_kwargs = low_branch.forward_kwargs(
                                video_rows=low_clean_video,
                                audio_rows=generated_high_audio,
                                step_timesteps=low_clean_timestep,
                            )
                            with (
                                resolution_local_session(low_session, restore=session),
                                low_session.clean_commit(phase.index),
                                _native_attention_metadata(low_branch),
                                torch.no_grad(),
                            ):
                                if base_model_forward is None:
                                    model(**low_clean_kwargs)
                                else:
                                    base_model_forward(
                                        model, low_clean_kwargs, len(sigmas_video) - 1
                                    )
                            expected_low_tokens = low_history_before + (
                                phase.video_latent_count * low_frame_rows
                                + 2 * phase.audio_latent_count
                            )
                            if low_cache.history_tokens != expected_low_tokens:
                                raise RuntimeError("native LOW clean KV differs from the phase boundary")
                            self._apply_cross_chunk_cache_policy(low_cache)
                        self._apply_cross_chunk_cache_policy(session.cache)
                        self._update_full_prefix(
                            clean_phase_video,
                            generated_high_audio,
                            phase=phase,
                            frame_rows=high_frame_rows,
                        )
                    if (
                        phase.index + 1 < len(phases)
                        and self.policy.uses_phase_latent_handoff(phase.index + 1)
                    ):
                        if skip_unused_low_prep:
                            # Only this high worker consumes the high tail; the
                            # low workers already own their stage-local tails.
                            previous_high_tail = clean_phase_video[-high_frame_rows:].detach().clone()
                        else:
                            previous_high_tail = (
                                clean_phase_video[-high_frame_rows:]
                                .detach()
                                .to(device="cpu", dtype=torch.float32)
                                .contiguous()
                            )
                            previous_low_tail = resize_video_state_rows(
                                previous_high_tail,
                                temporal_length=1,
                                source_canvas=plan.high,
                                target_canvas=plan.low,
                            ).contiguous()
                    high_video_outputs.append(clean_phase_video)
                    high_audio_outputs.append(generated_high_audio)
                    phase_records.append(
                        StreamingPhaseExecution(
                            phase_index=phase.index,
                            frame_start=phase.frame_start,
                            frame_stop=phase.frame_stop,
                            video_latent_start=phase.video_latent_start,
                            video_latent_stop=phase.video_latent_stop,
                            audio_latent_start=phase.audio_latent_start,
                            audio_latent_stop=phase.audio_latent_stop,
                            denoise_forwards=1 if pp is not None else (
                                (len(low_sigmas_video) - 1)
                                + len(high_sigmas_video) - 1
                            ),
                            clean_forwards=(2 if dual_clean else int(not progressive_hiar)) + (
                                predicted_history.separate_clean_count(phase.index, len(phases)-1)
                                if predicted_history is not None else 0
                            ),
                            committed_history_tokens=(
                                sum(hiar_histories.history_tokens)
                                if progressive_hiar
                                else session.cache.history_tokens + (
                                    low_cache.history_tokens if dual_clean else 0
                                )
                            ),
                            full_prefix_tokens=(
                                0 if full_prefix is None else full_prefix.token_count
                            ),
                            full_prefix_forwards=int(full_prefix is not None),
                            phase_latent_handoff_used=use_phase_latent_tail,
                            full_prefix_capture_video_latents=(
                                0
                                if full_prefix is None
                                else self.plan.phases[0].video_latent_count
                            ),
                            full_prefix_capture_audio_latents_per_channel=(
                                0
                                if (
                                    full_prefix is None
                                    or self.full_prefix_modality
                                    == FULL_PREFIX_VIDEO_ONLY
                                )
                                else FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
                            ),
                            global_block_index=(
                                global_block_index
                                if progressive_hiar
                                else None
                            ),
                            full_prefix_visible_source_block_index=(
                                hiar_visible_prefix_source
                                if progressive_hiar
                                else None
                            ),
                        )
                    )
                    if full_prefix is not None:
                        full_prefix.release()
                    full_prefix = None

            if independent_stage0:
                high_audio_outputs = [_teacher_phase_audio_rows(
                    audio_teacher['milestones'][2], phase=p, total_audio_latents=total_audio_latents,
                ) for p in phases]
            joined_video = torch.cat(high_video_outputs, dim=0)
            joined_audio = _join_phase_audio(high_audio_outputs, phases)
            joined_lifted = torch.cat(lifted_outputs, dim=0)
            if list(joined_video.shape) != [
                total_video_latents * high_frame_rows,
                96,
            ] or list(joined_audio.shape) != [2 * total_audio_latents, 32]:
                raise RuntimeError(
                    "reassembled progressive streaming output changed high geometry"
                )
            audio_teacher_receipt = None
            if audio_teacher is not None:
                final_teacher = audio_teacher["milestones"][2]
                if not torch.equal(joined_audio, final_teacher):
                    raise RuntimeError(
                        "published progressive audio differs from Base teacher "
                        "clean latent"
                    )
                audio_teacher_receipt = dict(audio_teacher["receipt"])
                audio_teacher_receipt["published_clean_audio_exact_match"] = True
            returned_video = joined_video
            returned_audio = joined_audio
            if canonical_continuation:
                returned_video = torch.cat(
                    (
                        transport_video_rows.to(
                            device=joined_video.device,
                            dtype=joined_video.dtype,
                        ),
                        joined_video,
                    ),
                    dim=0,
                )
                previous_audio = transport_audio_rows.view(
                    2, -1, int(joined_audio.shape[-1])
                )[:, -audio_transport_prefix_latents:]
                returned_audio = torch.cat(
                    (
                        previous_audio.to(
                            device=joined_audio.device,
                            dtype=joined_audio.dtype,
                        ),
                        joined_audio.view(
                            2, total_audio_latents, int(joined_audio.shape[-1])
                        ),
                    ),
                    dim=1,
                ).contiguous().view(-1, int(joined_audio.shape[-1]))
            if self.reuse_across_requests:
                self._clean_video_condition_rows = (
                    joined_video[-high_frame_rows:]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                if canonical_geometry:
                    self._canonical_transport_video_rows = (
                        joined_video[-2 * high_frame_rows :]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    audio_by_channel = joined_audio.view(
                        2, total_audio_latents, int(joined_audio.shape[-1])
                    )
                    self._canonical_transport_audio_rows = (
                        audio_by_channel[:, -min(9, total_audio_latents) :]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                        .view(-1, int(joined_audio.shape[-1]))
                    )
                if progressive_hiar:
                    retained_history_tokens = sum(
                        hiar_histories.history_tokens
                    )
                    retained_history_audio_tokens = sum(
                        hiar_histories.history_audio_tokens
                    )
                    retained_history_video_tokens = sum(
                        hiar_histories.history_video_tokens
                    )
                else:
                    retained_history_tokens = cache.history_tokens
                    retained_history_audio_tokens = cache.history_audio_tokens
                    retained_history_video_tokens = cache.history_video_tokens
                    if dual_clean:
                        retained_history_tokens += low_cache.history_tokens
                        retained_history_audio_tokens += low_cache.history_audio_tokens
                        retained_history_video_tokens += low_cache.history_video_tokens
                self._media_time_origin = media_time_origin
            else:
                retained_history_tokens = 0
                retained_history_audio_tokens = 0
                retained_history_video_tokens = 0
            progressive_hiar_receipt = None
            if stage_predicted_kv:
                progressive_hiar_receipt = {
                    "history_mode": "stage_local_predicted_clean" + ("_pipeline" if pp is not None else "_serial_reference"),
                    "stage_history_canvas": ["low", "low", "high"],
                    "low_stage_history_sources": ["predicted_clean", "predicted_clean"],
                    "low_stage_audio_history_source": "base10_clean" if audio_teacher is not None else "predicted_clean",
                    "high_history_source": "native_final_clean_with_prefix_renorm",
                    "low_committed_blocks": [item.committed_blocks for item in predicted_history.caches],
                    "extra_low_clean_forwards_per_chunk": None if predicted_history.fused else len(predicted_history.owned_stages),
                    "separate_low_clean_forwards_total": predicted_history.separate_clean_forwards,
                    "fused_low_commit_denoise_forwards_total": predicted_history.fused_forwards,
                    "parallel_pipeline": pp is not None,
                    "low_tail_condition_source": (
                        "disabled" if not self.policy.cross_request_visual_handoff
                        and self.policy.phase_latent_handoff_mode == "off"
                        else "native_final_high"
                    ),
                    "low_tail_requires_high_feedback": (
                        (self.policy.cross_request_visual_handoff
                             or self.policy.phase_latent_handoff_mode != "off")
                    ),
                    "owned_stage": None if pp is None else pp.stage,
                    "owned_low_stages": list(predicted_history.owned_stages),
                    "fused_commit_denoise": predicted_history.fused,
                    "fused_static_runtime_plans": predicted_history.fused and os.environ.get("H3_PROGRESSIVE_FUSED_STATIC_PLANS") == "1",
                }
            progressive_dual_clean_receipt = (
                {
                    "history_mode": "dual_clean",
                    "low_source": "last_low_native_x0_prediction",
                    "low_source_sigma": float(low_sigmas_video[-2]),
                    "low_source_step": len(low_sigmas_video) - 2,
                    "high_source": "final_high_clean",
                    "audio_source": "same_final_clean_audio",
                    "low_prediction_resized": False,
                    "low_prediction_renormalized": False,
                    "low_starting_history_tokens": low_starting_history_tokens,
                    "low_retained_history_tokens": low_cache.history_tokens,
                    "high_retained_history_tokens": cache.history_tokens,
                    "low_committed_blocks": low_cache.committed_blocks,
                    "high_committed_blocks": cache.committed_blocks,
                    "low_clean_forwards": len(phases),
                    "high_clean_forwards": len(phases),
                }
                if dual_clean else None
            )
            if progressive_hiar:
                progressive_hiar_receipt = hiar_histories.receipt()
                progressive_hiar_receipt.update(
                    {
                        "full_prefix_lag_blocks": (
                            self.policy.full_prefix_lag_blocks
                        ),
                        "full_prefix_visible_source_block_index": (
                            hiar_visible_prefix_source
                        ),
                        "prefix_renorm": False,
                        "block_group_count": int(
                            self.plan.phases[0].group_count
                        ),
                        "stage_input_canvas": ["low", "low", "high"],
                        "stage2_input": "lifted",
                        "audio_stage_input_aligned": True,
                        "full_prefix_modality": "audio_video",
                        "full_prefix_audio_latent_units_per_channel": (
                            FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
                        ),
                        "terminal_clean_forward_count": 0,
                    }
                )
            execution = StreamingExecution(
                native_frame_count=active_plan.native_frame_count,
                attention_backend_receipt=(
                    session.hook.attention_backend_receipt()
                    if session is not None and self.attention_backend == "sage_attn"
                    else None
                ),
                phase_executions=tuple(phase_records),
                request_task=high_request_video.task,
                noisy_kv_writes=3 * len(phase_records) if progressive_hiar else 0,
                video_latent_offset=self._video_latent_offset,
                audio_latent_offset=self._audio_latent_offset,
                starting_history_tokens=starting_history_tokens,
                retained_history_tokens=retained_history_tokens,
                cross_request_kv_reused=reused_kv,
                cross_request_kv_boundary_reset=boundary_reset_applied,
                cross_request_kv_boundary_reset_tokens=boundary_reset_tokens,
                cross_request_clean_video_latent_handoff=reused_clean_video,
                full_prefix_half_life_seconds=(
                    self._progressive_hiar_prefix_memory.half_life_seconds
                    if progressive_hiar
                    else (
                        None
                        if self._full_prefix_memory is None
                        else self._full_prefix_memory.half_life_seconds
                    )
                ),
                full_prefix_modality=(
                    None
                    if (
                        self._progressive_hiar_prefix_memory is None
                        and self._full_prefix_memory is None
                    )
                    else self.full_prefix_modality
                ),
                full_prefix_phases_used=sum(
                    int(item.full_prefix_tokens > 0) for item in phase_records
                ),
                streaming_policy=self.policy.to_dict(),
                request_index=request_index,
                current_prompt_rope_start=current_prompt_rope_start,
                audio_kv_reset_applied=audio_kv_reset_applied,
                audio_kv_reset_tokens=audio_kv_reset_tokens,
                starting_history_audio_tokens=starting_history_audio_tokens,
                starting_history_video_tokens=starting_history_video_tokens,
                retained_history_audio_tokens=retained_history_audio_tokens,
                retained_history_video_tokens=retained_history_video_tokens,
                canonical_continuation_geometry=canonical_geometry,
                native_frame_offset=self._native_frame_offset,
                published_native_frames=active_plan.native_frame_count,
                video_transport_prefix_latents=(
                    video_transport_prefix_latents
                ),
                audio_transport_prefix_latents_per_channel=(
                    audio_transport_prefix_latents
                ),
                published_video_latents=total_video_latents,
                published_audio_latents_per_channel=total_audio_latents,
                current_text_token_count=(
                    0
                    if temporal_alignment is None
                    else temporal_alignment.current_text_token_count
                ),
                text_time_start=(
                    current_prompt_rope_start
                    if temporal_alignment is None
                    else float(temporal_alignment.text_time_start)
                ),
                text_time_stop=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.text_time_stop)
                ),
                media_time_start=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.media_time_start)
                ),
                request_video_time_shift=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.request_video_time_shift)
                ),
                audio_origin_skew=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.audio_origin_skew)
                ),
                fixed_media_time_origin=media_time_origin,
                reuse_across_requests=self.reuse_across_requests,
                position_capture=position_capture,
                progressive_hiar=progressive_hiar_receipt,
                progressive_dual_clean=progressive_dual_clean_receipt,
                audio_teacher=audio_teacher_receipt,
            )
            self.last_execution = execution
            self.executions.append(execution)
            if self.reuse_across_requests:
                self._video_latent_offset += total_video_latents
                self._audio_latent_offset += total_audio_latents
                self._native_frame_offset += active_plan.native_frame_count
                self._request_index += 1
            if high_request_video.condition_initial_rows is not None:
                returned_condition_rows = (
                    direct_high_condition_rows
                    if direct_high_condition_rows is not None
                    else high_request_video.keyframe_condition_rows
                )
                returned_video = torch.cat(
                    (
                        returned_condition_rows.to(
                            device=returned_video.device,
                            dtype=returned_video.dtype,
                        ),
                        returned_video,
                    ),
                    dim=0,
                )
            if list(returned_video.shape) != list(high_template.shape):
                raise RuntimeError(
                    "progressive streaming video output no longer matches raw request ABI"
                )
            if list(returned_audio.shape) != [2 * official_audio_latents, 32]:
                raise RuntimeError(
                    "progressive streaming audio output no longer matches raw request ABI"
                )
            return result_type(
                video_rows=returned_video,
                audio_rows=returned_audio,
                lifted_video_target_rows=joined_lifted,
                lift_receipt={
                    "schema": "digitalhuman_h3.progressive_resolution_lift.streaming.v1",
                    "phase_count": len(phases),
                    "video_sigma": float(sigmas_video[switch_position]),
                    "noise_seed": int(kwargs["lift_noise_seed"]),
                    "temporal_offsets": [
                        video_transport_prefix_latents + phase.video_latent_start
                        for phase in phases
                    ],
                },
                high_target_source="continued_from_lifted_low_state",
                low_model_forwards=switch_position,
                high_model_forwards=(
                    len(sigmas_video) - switch_position - 1
                ),
            )
        finally:
            self._running = False

    def _cache_for_model(self, model: Any) -> CleanAVKVCache:
        contract = _kv_contract_for_model(model)
        if not self.reuse_across_requests:
            return CleanAVKVCache(contract)
        if self._cache is None:
            self._cache = CleanAVKVCache(contract)
        elif self._cache.contract != contract:
            raise ValueError(
                "streaming KV cache topology changed across reused requests"
            )
        return self._cache

    def _run(self, original_loop: Any, loop_module: Any, **kwargs: Any) -> Any:
        import torch

        if self._running:
            raise RuntimeError("one H3 streaming overlay cannot run concurrent requests")
        if self.policy.phase_latent_handoff_mode != "off":
            raise RuntimeError(
                "phase latent handoff is defined only for progressive-resolution "
                "streaming; use mode=off for a single-resolution request"
            )
        if self.policy.progressive_history_kv_mode != "clean_commit":
            raise RuntimeError(
                "resolution-local progressive KV requires progressive resolution"
            )
        self._running = True
        try:
            model = kwargs["model"]
            positive = kwargs["positive"]
            initial_video = _required_tensor(
                kwargs["initial_video_rows"], "initial_video_rows"
            )
            initial_audio = _required_tensor(
                kwargs["initial_audio_rows"], "initial_audio_rows"
            )
            if kwargs.get("audio_ref_rows") is not None:
                raise NotImplementedError(
                    "streaming Ref2VA audio/reference conditions are not authorized"
                )
            if int(positive.audio_target_start) != 0:
                raise RuntimeError("streaming production entry rejects audio reference rows")

            request_index = self._request_index
            canonical_geometry = self.policy.canonical_continuation_geometry
            canonical_continuation = canonical_geometry and request_index > 0
            active_plan = (
                canonical_continuation_plan(
                    self.plan,
                    request_index=request_index,
                )
                if canonical_continuation
                else self.plan
            )
            official_video_latents = self.plan.phases[-1].video_latent_stop
            official_audio_latents = self.plan.phases[-1].audio_latent_stop
            total_video_latents = active_plan.phases[-1].video_latent_stop
            total_audio_latents = active_plan.phases[-1].audio_latent_stop
            video_transport_prefix_latents = (
                official_video_latents - total_video_latents
            )
            audio_transport_prefix_latents = (
                official_audio_latents - total_audio_latents
            )
            target_video_rows = int(initial_video.shape[0]) - int(
                positive.video_target_start
            )
            if (
                target_video_rows <= 0
                or target_video_rows % official_video_latents
            ):
                raise ValueError("request target video rows do not match the stream plan")
            frame_rows = target_video_rows // official_video_latents
            if list(initial_audio.shape) != [2 * official_audio_latents, 32]:
                raise ValueError(
                    "request audio rows do not match cumulative 40 Hz phase boundaries"
                )
            latent_h, latent_w = _resolve_spatial_latent_shape(
                positive,
                frame_rows=frame_rows,
            )
            request_video = _resolve_request_video_state(
                positive=positive,
                initial_video=initial_video,
                keyframe_cond_rows=kwargs.get("keyframe_cond_rows"),
                frame_rows=frame_rows,
                total_video_latents=official_video_latents,
            )
            official_initial_audio = initial_audio
            transport_video_rows = None
            transport_audio_rows = None
            if canonical_continuation:
                if request_video.task not in {"t2va", "fl2va_first_frame"}:
                    raise RuntimeError(
                        "canonical continuation requests admit T2VA or one "
                        "terminal clean-latent anchor"
                    )
                transport_video_rows = self._canonical_transport_video_rows
                transport_audio_rows = self._canonical_transport_audio_rows
                if (
                    transport_video_rows is None
                    or list(transport_video_rows.shape)
                    != [video_transport_prefix_latents * frame_rows, 96]
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior video transport prefix"
                    )
                if transport_audio_rows is None:
                    raise RuntimeError(
                        "canonical continuation has no prior audio transport prefix"
                    )
                previous_audio_t = int(transport_audio_rows.shape[0]) // 2
                if list(transport_audio_rows.shape) != [2 * previous_audio_t, 32] or (
                    previous_audio_t < audio_transport_prefix_latents
                ):
                    raise RuntimeError(
                        "canonical continuation has no exact prior audio transport prefix"
                    )
                request_video = _RequestVideoState(
                    task=request_video.task,
                    target_initial_rows=request_video.target_initial_rows[
                        video_transport_prefix_latents * frame_rows :
                    ].contiguous(),
                    condition_initial_rows=request_video.condition_initial_rows,
                    keyframe_condition_rows=request_video.keyframe_condition_rows,
                    keyframe_frame_indices=request_video.keyframe_frame_indices,
                )
                initial_audio_by_channel = initial_audio.view(
                    2, official_audio_latents, -1
                )
                initial_audio = initial_audio_by_channel[
                    :, audio_transport_prefix_latents:
                ].contiguous().view(-1, int(initial_audio.shape[-1]))
            clip_keyframes = self._resolve_clip_keyframes(request_video)
            phase_condition_state = clip_keyframes or request_video
            clip_keyframe_end = (
                self._clip_keyframe_deadline["end_video_latent"]
                if clip_keyframes is not None
                else None
            )
            text_token_tags = _resolve_global_text_token_tags(positive)
            text_len = int(
                _required_tensor(
                    positive.static_kwargs.get("prompt_embeds"),
                    "positive.static_kwargs['prompt_embeds']",
                ).shape[0]
            )
            media_time_origin = (
                text_len
                if self._media_time_origin is None or not self.reuse_across_requests
                else self._media_time_origin
            )
            temporal_alignment = (
                _request_temporal_alignment(
                    current_text_len=text_len,
                    fixed_media_origin=media_time_origin,
                    total_frame_count=active_plan.native_frame_count,
                    total_video_latents=total_video_latents,
                    video_latent_offset=self._video_latent_offset,
                    audio_latent_offset=self._audio_latent_offset,
                )
                if not canonical_geometry
                and active_plan.native_frame_count == 243
                and total_video_latents == 72
                and len(active_plan.phases) == 1
                else None
            )
            device = initial_video.device
            if device.type != "cpu":
                raise ValueError("official H3 initial noise must enter denoise from CPU")
            cuda_device = torch.device("cuda")

            sigmas_video = list(kwargs["sigmas_video"])
            sigmas_audio = list(kwargs["sigmas_audio"])
            if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
                raise ValueError("H3 video/audio sigma schedules must match")
            base_model_forward = kwargs.get("model_forward")
            phase_video_outputs: list[Any] = []
            phase_audio_outputs: list[Any] = []
            phase_records: list[StreamingPhaseExecution] = []
            position_capture: dict[str, object] | None = None
            phases = active_plan.phases
            direct_prompt_prefix = self._begin_direct_prompt_local_prefix()
            cache = self._cache
            if cache is None:
                cache = CleanAVKVCache(
                    kv_contract_from_model(model),
                    segmented_fp8_history=(
                        self.attention_backend
                        in {
                            "flash_attn3_fp8_segmented",
                        }
                    ),
                )
                if self.reuse_across_requests:
                    self._cache = cache
            boundary_reset_tokens = 0
            boundary_reset_applied = False
            pre_boundary_request_index = (
                self._video_latent_offset // total_video_latents
            )
            if (
                self.policy.cross_request_kv_boundary_reset
                and pre_boundary_request_index > 0
            ):
                boundary_reset_tokens = (
                    cache.discard_retained_history_preserve_serial()
                )
                boundary_reset_applied = True
            (
                request_index,
                audio_kv_reset_applied,
                audio_kv_reset_tokens,
            ) = self._apply_request_boundary_audio_policy(
                cache,
                total_video_latents=total_video_latents,
            )
            starting_history_tokens = cache.history_tokens
            starting_history_audio_tokens = cache.history_audio_tokens
            starting_history_video_tokens = cache.history_video_tokens
            reused_kv = starting_history_tokens > 0
            current_prompt_rope_start = _current_prompt_rope_start(
                text_len=text_len,
                media_time_origin=media_time_origin,
                video_latent_offset=self._video_latent_offset,
                total_video_latents=total_video_latents,
                canonical_continuation_geometry=canonical_geometry,
            )
            if (
                temporal_alignment is not None
                and temporal_alignment.request_index != request_index
            ):
                raise RuntimeError(
                    "bounded RoPE request index differs from the cache boundary"
                )
            direct_condition_rows = None
            reused_clean_video = False
            if (
                self.reuse_across_requests
                and self._clean_video_condition_rows is not None
                and request_video.task == "fl2va_first_frame"
            ):
                direct_condition_rows = self._clean_video_condition_rows
                if list(direct_condition_rows.shape) != [frame_rows, 96]:
                    raise RuntimeError(
                        "cross-request clean video latent changed spatial geometry"
                    )
                reused_clean_video = True

            audio_teacher = _resolve_base_audio_teacher(
                external_artifact=self.external_base_audio_teacher,
                request_index=request_index,
                torch_module=torch,
                device=cuda_device,
                run_internal=lambda: _run_base_audio_teacher(
                    enabled=self.policy.base_audio_teacher,
                    original_loop=original_loop,
                    loop_module=loop_module,
                    model=model,
                    positive=positive,
                    initial_audio_rows=official_initial_audio,
                    total_audio_latents=official_audio_latents,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    device=cuda_device,
                ),
            )
            audio_teacher = _active_base_audio_teacher(
                audio_teacher,
                official_audio_latents=official_audio_latents,
                active_audio_latents=total_audio_latents,
                transport_prefix_audio_latents=audio_transport_prefix_latents,
            )

            # The encoded prefix already binds the current request prompt and
            # fixed first-block layout.  The optional direct-only fast path
            # holds that immutable K/V while the slow latent EMA advances
            # across the remaining internal chunks of this five-second request.
            request_full_prefix = None
            with H3StreamingSession(
                active_plan,
                model=model,
                server_args=self.server_args,
                cache=cache,
                attention_backend=self.attention_backend,
                sol_attn_tau=self.sol_attn_tau,
                sol_attn_dense_layers=self.sol_attn_dense_layers,
                sol_attn_dense_denoise_steps=self.sol_attn_dense_denoise_steps,
                sol_attn_route_current_video=self.sol_attn_route_current_video,
                # The progressive loop owns a different stage-history topology;
                # this workspace is deliberately scoped to the Direct loop.
                direct_sage_kv_workspace=self.direct_sage_kv_workspace,
                sol_attn_static_summary_cache=(
                    self.sol_attn_static_summary_cache
                ),
            ) as session:
                for phase in phases:
                    phase_keyframe_indices = (
                        phase_condition_state.keyframe_frame_indices
                        if clip_keyframes is not None or phase.index == 0
                        else ()
                    )
                    if direct_prompt_prefix is not None:
                        full_prefix = direct_prompt_prefix.active
                        full_prefix_encoded = False
                    else:
                        full_prefix, request_full_prefix, full_prefix_encoded = (
                            _direct_request_full_prefix(
                                enabled=self.policy.direct_request_full_prefix_reuse,
                                retained=request_full_prefix,
                                encode=lambda: self._encode_full_prefix(
                                    session=session,
                                    model=model,
                                    loop_module=loop_module,
                                    positive=positive,
                                    text_token_tags=text_token_tags,
                                    latent_h=latent_h,
                                    latent_w=latent_w,
                                    media_time_origin=media_time_origin,
                                ),
                            )
                        )
                    branch = _build_phase_branch(
                        loop_module=loop_module,
                        positive=positive,
                        phase=phase,
                        latent_h=latent_h,
                        latent_w=latent_w,
                        model=model,
                        device=cuda_device,
                        text_token_tags=text_token_tags,
                        keyframe_frame_indices=phase_keyframe_indices,
                        total_frame_count=active_plan.native_frame_count,
                        total_video_latents=total_video_latents,
                        media_time_origin=media_time_origin,
                        video_latent_offset=self._video_latent_offset,
                        audio_latent_offset=self._audio_latent_offset,
                        canonical_continuation_geometry=canonical_geometry,
                        clip_keyframe_end_video_latent=clip_keyframe_end,
                    )
                    branch_capture = getattr(
                        branch, "_digitalhuman_h3_position_capture", None
                    )
                    if branch_capture is not None:
                        if position_capture is not None:
                            raise RuntimeError(
                                "bounded RoPE exact request produced multiple captures"
                            )
                        position_capture = dict(branch_capture)
                    phase_video, phase_audio = _phase_initial_rows(
                        initial_video_rows=request_video.target_initial_rows,
                        initial_audio_rows=initial_audio,
                        phase=phase,
                        frame_rows=frame_rows,
                        total_audio_latents=total_audio_latents,
                    )
                    phase_keyframe_rows = None
                    if phase_keyframe_indices:
                        phase_video = torch.cat(
                            (phase_condition_state.condition_initial_rows, phase_video), dim=0
                        )
                        phase_keyframe_rows = (
                            direct_condition_rows
                            if direct_condition_rows is not None
                            else phase_condition_state.keyframe_condition_rows
                        )

                    def noisy_forward(
                        active_model: Any,
                        call_kwargs: dict[str, Any],
                        step_index: int,
                    ) -> tuple[Any, Any]:
                        if (
                            audio_teacher is not None
                            and base_model_forward is None
                        ):
                            call_kwargs["skip_audio_output_head"] = True
                        with (
                            session.noisy_step(
                                phase.index,
                                full_prefix=full_prefix,
                                denoise_step_index=step_index,
                            ),
                            _native_attention_metadata(branch),
                        ):
                            if base_model_forward is None:
                                return active_model(**call_kwargs)
                            return base_model_forward(
                                active_model,
                                call_kwargs,
                                step_index,
                            )

                    outer_on_step = (
                        kwargs.get("on_step")
                        if phase.index == len(phases) - 1
                        else None
                    )
                    phase_on_step = _base_audio_teacher_step_callback(
                        audio_teacher,
                        phase=phase,
                        total_audio_latents=total_audio_latents,
                        audio_target_slice=branch.audio_target_slice,
                        outer=outer_on_step,
                    )
                    base_step_profiler = kwargs.get("step_profiler")
                    phase_step_profiler = None
                    if base_step_profiler is not None:

                        def phase_step_profiler(step_index: int) -> Any:
                            global_step = (
                                phase.index * (len(sigmas_video) - 1) + step_index
                            )
                            return base_step_profiler(global_step)

                    generated_video, generated_audio = original_loop(
                        model=model,
                        model_forward=noisy_forward,
                        positive=branch,
                        initial_video_rows=phase_video,
                        initial_audio_rows=phase_audio,
                        keyframe_cond_rows=phase_keyframe_rows,
                        audio_ref_rows=None,
                        sigmas_video=sigmas_video,
                        sigmas_audio=sigmas_audio,
                        device=cuda_device,
                        imgvid_cond_noise_aug_for_inference=(
                            1.0
                            if reused_clean_video
                            else kwargs.get(
                                "imgvid_cond_noise_aug_for_inference", 0.999
                            )
                        ),
                        audio_cond_noise_aug_for_inference=kwargs.get(
                            "audio_cond_noise_aug_for_inference", 1.0
                        ),
                        on_step=phase_on_step,
                        step_profiler=phase_step_profiler,
                        consistency_renoise=bool(
                            kwargs.get("consistency_renoise", False)
                        ),
                        renoise_generator=kwargs.get("renoise_generator"),
                    )

                    clean_phase_video = self._renorm_clean_video_rows(
                        generated_video[branch.video_target_slice]
                    )
                    generated_video[branch.video_target_slice].copy_(clean_phase_video)
                    history_before_commit = session.cache.history_tokens

                    clean_timestep = branch.prepare_timestep_plan(
                        video_timesteps=[1.0],
                        audio_timesteps=[1.0],
                        imgvid_cond_noise_aug=float(
                            1.0
                            if reused_clean_video
                            else kwargs.get(
                                "imgvid_cond_noise_aug_for_inference", 0.999
                            )
                        ),
                        audio_ref_cond_noise_aug=float(
                            kwargs.get("audio_cond_noise_aug_for_inference", 1.0)
                        ),
                    )[0]
                    clean_kwargs = branch.forward_kwargs(
                        video_rows=generated_video,
                        audio_rows=generated_audio,
                        step_timesteps=clean_timestep,
                    )
                    with (
                        session.clean_commit(
                            phase.index, full_prefix=full_prefix
                        ),
                        _native_attention_metadata(branch),
                        torch.no_grad(),
                    ):
                        if base_model_forward is None:
                            clean_kwargs["streaming_cache_only"] = True
                            model(**clean_kwargs)
                        else:
                            base_model_forward(
                                model,
                                clean_kwargs,
                                len(sigmas_video) - 1,
                            )

                    expected_history_tokens = history_before_commit + (
                        phase.video_latent_count * frame_rows
                        + 2 * phase.audio_latent_count
                    )
                    if session.cache.history_tokens != expected_history_tokens:
                        raise RuntimeError(
                            "clean H3 KV history does not match the committed "
                            f"AV phase boundary: {session.cache.history_tokens} != "
                            f"{expected_history_tokens}"
                        )

                    if direct_prompt_prefix is not None:
                        direct_prompt_prefix.capture_c0(
                            session.cache, phase.index, frame_rows,
                        )
                    self._apply_cross_chunk_cache_policy(session.cache)
                    self._update_full_prefix(
                        clean_phase_video,
                        generated_audio,
                        phase=phase,
                        frame_rows=frame_rows,
                    )
                    phase_video_outputs.append(clean_phase_video)
                    phase_audio_outputs.append(generated_audio)
                    phase_records.append(
                        StreamingPhaseExecution(
                            phase_index=phase.index,
                            frame_start=phase.frame_start,
                            frame_stop=phase.frame_stop,
                            video_latent_start=phase.video_latent_start,
                            video_latent_stop=phase.video_latent_stop,
                            audio_latent_start=phase.audio_latent_start,
                            audio_latent_stop=phase.audio_latent_stop,
                            denoise_forwards=len(sigmas_video) - 1,
                            clean_forwards=1,
                            committed_history_tokens=session.cache.history_tokens,
                            full_prefix_tokens=(
                                0 if full_prefix is None or direct_prompt_prefix is not None
                                else full_prefix.token_count
                            ),
                            full_prefix_forwards=int(full_prefix_encoded),
                            prompt_local_video_prefix_tokens=(
                                full_prefix.token_count
                                if full_prefix is not None and direct_prompt_prefix is not None else 0
                            ),
                            full_prefix_capture_video_latents=(
                                0
                                if full_prefix is None or direct_prompt_prefix is not None
                                else self.plan.phases[0].video_latent_count
                            ),
                            full_prefix_capture_audio_latents_per_channel=(
                                0
                                if (
                                    full_prefix is None
                                    or direct_prompt_prefix is not None
                                    or self.full_prefix_modality
                                    == FULL_PREFIX_VIDEO_ONLY
                                )
                                else FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
                            ),
                        )
                    )
                    if (
                        full_prefix is not None
                        and direct_prompt_prefix is None
                        and not self.policy.direct_request_full_prefix_reuse
                    ):
                        full_prefix.release()
                    full_prefix = None

            if request_full_prefix is not None:
                request_full_prefix.release()
                request_full_prefix = None
            attention_backend_receipt = session.hook.attention_backend_receipt()

            joined_video = torch.cat(phase_video_outputs, dim=0)
            joined_audio = _join_phase_audio(phase_audio_outputs, phases)
            if joined_video.shape != request_video.target_initial_rows.shape:
                raise RuntimeError("reassembled streaming video rows changed request shape")
            if joined_audio.shape != initial_audio.shape:
                raise RuntimeError("reassembled streaming audio rows changed request shape")
            audio_teacher_receipt = None
            if audio_teacher is not None:
                final_teacher = audio_teacher["milestones"][2]
                if not torch.equal(joined_audio, final_teacher):
                    raise RuntimeError(
                        "published streaming audio differs from Base teacher clean latent"
                    )
                audio_teacher_receipt = dict(audio_teacher["receipt"])
                audio_teacher_receipt["published_clean_audio_exact_match"] = True
            returned_video = joined_video
            returned_audio = joined_audio
            if canonical_continuation:
                returned_video = torch.cat(
                    (
                        transport_video_rows.to(
                            device=joined_video.device,
                            dtype=joined_video.dtype,
                        ),
                        joined_video,
                    ),
                    dim=0,
                )
                previous_audio = transport_audio_rows.view(
                    2, -1, int(joined_audio.shape[-1])
                )[:, -audio_transport_prefix_latents:]
                returned_audio = torch.cat(
                    (
                        previous_audio.to(
                            device=joined_audio.device,
                            dtype=joined_audio.dtype,
                        ),
                        joined_audio.view(
                            2, total_audio_latents, int(joined_audio.shape[-1])
                        ),
                    ),
                    dim=1,
                ).contiguous().view(-1, int(joined_audio.shape[-1]))
            if self.reuse_across_requests:
                self._clean_video_condition_rows = (
                    joined_video[-frame_rows:]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .contiguous()
                )
                if canonical_geometry:
                    self._canonical_transport_video_rows = (
                        joined_video[-2 * frame_rows :]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                    )
                    audio_by_channel = joined_audio.view(
                        2, total_audio_latents, int(joined_audio.shape[-1])
                    )
                    self._canonical_transport_audio_rows = (
                        audio_by_channel[:, -min(9, total_audio_latents) :]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .contiguous()
                        .view(-1, int(joined_audio.shape[-1]))
                    )
                retained_history_tokens = cache.history_tokens
                retained_history_audio_tokens = cache.history_audio_tokens
                retained_history_video_tokens = cache.history_video_tokens
                self._media_time_origin = media_time_origin
            else:
                retained_history_tokens = 0
                retained_history_audio_tokens = 0
                retained_history_video_tokens = 0
            execution = StreamingExecution(
                native_frame_count=active_plan.native_frame_count,
                phase_executions=tuple(phase_records),
                request_task=request_video.task,
                prompt_local_video_prefix=(
                    None if direct_prompt_prefix is None else direct_prompt_prefix.report()
                ),
                video_latent_offset=self._video_latent_offset,
                audio_latent_offset=self._audio_latent_offset,
                starting_history_tokens=starting_history_tokens,
                retained_history_tokens=retained_history_tokens,
                cross_request_kv_reused=reused_kv,
                cross_request_kv_boundary_reset=boundary_reset_applied,
                cross_request_kv_boundary_reset_tokens=boundary_reset_tokens,
                cross_request_clean_video_latent_handoff=reused_clean_video,
                full_prefix_half_life_seconds=(
                    None
                    if self._full_prefix_memory is None
                    else self._full_prefix_memory.half_life_seconds
                ),
                full_prefix_modality=(
                    None
                    if self._full_prefix_memory is None
                    else self.full_prefix_modality
                ),
                full_prefix_phases_used=sum(
                    int(item.full_prefix_tokens > 0) for item in phase_records
                ),
                streaming_policy=self.policy.to_dict(),
                request_index=request_index,
                current_prompt_rope_start=current_prompt_rope_start,
                audio_kv_reset_applied=audio_kv_reset_applied,
                audio_kv_reset_tokens=audio_kv_reset_tokens,
                starting_history_audio_tokens=starting_history_audio_tokens,
                starting_history_video_tokens=starting_history_video_tokens,
                retained_history_audio_tokens=retained_history_audio_tokens,
                retained_history_video_tokens=retained_history_video_tokens,
                canonical_continuation_geometry=canonical_geometry,
                native_frame_offset=self._native_frame_offset,
                published_native_frames=active_plan.native_frame_count,
                video_transport_prefix_latents=(
                    video_transport_prefix_latents
                ),
                audio_transport_prefix_latents_per_channel=(
                    audio_transport_prefix_latents
                ),
                published_video_latents=total_video_latents,
                published_audio_latents_per_channel=total_audio_latents,
                current_text_token_count=(
                    0
                    if temporal_alignment is None
                    else temporal_alignment.current_text_token_count
                ),
                text_time_start=(
                    current_prompt_rope_start
                    if temporal_alignment is None
                    else float(temporal_alignment.text_time_start)
                ),
                text_time_stop=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.text_time_stop)
                ),
                media_time_start=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.media_time_start)
                ),
                request_video_time_shift=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.request_video_time_shift)
                ),
                audio_origin_skew=(
                    0.0
                    if temporal_alignment is None
                    else float(temporal_alignment.audio_origin_skew)
                ),
                fixed_media_time_origin=media_time_origin,
                reuse_across_requests=self.reuse_across_requests,
                position_capture=position_capture,
                audio_teacher=audio_teacher_receipt,
                attention_backend_receipt=attention_backend_receipt,
                clip_keyframe_deadline=(
                    dict(self._clip_keyframe_deadline)
                    if clip_keyframes is not None else None
                ),
                clip_keyframe_visible_frame_indices=(
                    clip_keyframes.keyframe_frame_indices
                    if clip_keyframes is not None else None
                ),
            )
            self.last_execution = execution
            self.executions.append(execution)
            if self.reuse_across_requests:
                self._video_latent_offset += total_video_latents
                self._audio_latent_offset += total_audio_latents
                self._native_frame_offset += active_plan.native_frame_count
                self._request_index += 1
            if request_video.condition_initial_rows is not None:
                returned_condition_rows = (
                    direct_condition_rows
                    if direct_condition_rows is not None
                    else request_video.keyframe_condition_rows
                )
                returned_video = torch.cat(
                    (
                        returned_condition_rows.to(
                            device=returned_video.device,
                            dtype=returned_video.dtype,
                        ),
                        returned_video,
                    ),
                    dim=0,
                )
            if returned_video.shape != initial_video.shape:
                raise RuntimeError("streaming output no longer matches official request rows")
            if list(returned_audio.shape) != [2 * official_audio_latents, 32]:
                raise RuntimeError("streaming audio output no longer matches official request rows")
            return returned_video, returned_audio
        finally:
            self._running = False

    @contextmanager
    def install(self) -> Iterator[H3StreamingRuntimeOverlay]:
        import importlib

        pipeline_module = importlib.import_module("digitalhuman_h3.model.pipeline")
        loop_module = importlib.import_module(DENOISE_LOOP_MODULE)
        original = getattr(pipeline_module, "minimax_h3_denoise_loop", None)
        original_progressive = getattr(
            pipeline_module, "minimax_h3_progressive_denoise_loop", None
        )
        if not callable(original):
            raise RuntimeError("pinned H3 denoise loop is unavailable")
        if not callable(original_progressive):
            raise RuntimeError("pinned H3 progressive denoise loop is unavailable")
        if getattr(original, "_digitalhuman_h3_streaming_overlay", False):
            raise RuntimeError("a Digitalhuman H3 streaming overlay is already installed")

        @wraps(original)
        def wrapped(**kwargs: Any) -> Any:
            return self._run(original, loop_module, **kwargs)

        wrapped._digitalhuman_h3_streaming_overlay = True

        @wraps(original_progressive)
        def progressive_wrapped(**kwargs: Any) -> Any:
            from digitalhuman_h3.model.progressive_denoise import (
                H3ProgressiveDenoiseResult,
            )

            return self._run_progressive(
                original,
                loop_module,
                H3ProgressiveDenoiseResult,
                **kwargs,
            )

        progressive_wrapped._digitalhuman_h3_streaming_overlay = True
        pipeline_module.minimax_h3_denoise_loop = wrapped
        pipeline_module.minimax_h3_progressive_denoise_loop = progressive_wrapped
        try:
            yield self
        finally:
            if (
                pipeline_module.minimax_h3_progressive_denoise_loop
                is progressive_wrapped
            ):
                pipeline_module.minimax_h3_progressive_denoise_loop = (
                    original_progressive
                )
            if pipeline_module.minimax_h3_denoise_loop is wrapped:
                pipeline_module.minimax_h3_denoise_loop = original


@contextmanager
def install_streaming_runtime(
    *,
    plan: StreamPlan,
    server_args: Any,
    reuse_across_requests: bool = False,
    full_prefix_half_life_seconds: float | None = None,
    full_prefix_modality: str = FULL_PREFIX_AUDIO_VIDEO,
    policy: StreamingInferencePolicy | None = None,
    attention_backend: str = "sdpa",
    sol_attn_tau: float = 1.0,
    sol_attn_dense_layers: int = 2,
    sol_attn_dense_denoise_steps: int = 1,
    sol_attn_route_current_video: bool = False,
    direct_sage_kv_workspace: bool = False,
    sol_attn_static_summary_cache: bool = False,
    external_base_audio_teacher: Any | None = None,
) -> Iterator[H3StreamingRuntimeOverlay]:
    overlay = H3StreamingRuntimeOverlay(
        plan=plan,
        server_args=server_args,
        reuse_across_requests=reuse_across_requests,
        full_prefix_half_life_seconds=full_prefix_half_life_seconds,
        full_prefix_modality=full_prefix_modality,
        policy=policy,
        attention_backend=attention_backend,
        sol_attn_tau=sol_attn_tau,
        sol_attn_dense_layers=sol_attn_dense_layers,
        sol_attn_dense_denoise_steps=sol_attn_dense_denoise_steps,
        sol_attn_route_current_video=sol_attn_route_current_video,
        direct_sage_kv_workspace=direct_sage_kv_workspace,
        sol_attn_static_summary_cache=sol_attn_static_summary_cache,
        external_base_audio_teacher=external_base_audio_teacher,
    )
    with overlay.install():
        yield overlay


__all__ = [
    "H3StreamingRuntimeOverlay",
    "CANONICAL_VIDEO_POSITION_POLICY",
    "PROMPT_POSITION_POLICY",
    "StreamingExecution",
    "StreamingPhaseExecution",
    "VIDEO_POSITION_POLICY",
    "build_streaming_phase_packed_layout",
    "install_streaming_runtime",
]
