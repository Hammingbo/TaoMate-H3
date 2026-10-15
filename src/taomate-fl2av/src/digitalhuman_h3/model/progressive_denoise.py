# SPDX-License-Identifier: Apache-2.0
"""One H3 progressive-resolution denoise request.

The public function is intentionally one call boundary.  Streaming inference
patches this boundary so the two low-resolution transitions and the final
high-resolution transition execute inside the same temporal-block cache
transaction.  Non-streaming ODE extraction executes the identical math on the
complete native clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..progressive_resolution import (
    ProgressiveResolutionPlan,
    lift_video_state_rows,
)
from .denoise import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop


@dataclass(frozen=True)
class H3ProgressiveDenoiseResult:
    video_rows: torch.Tensor
    audio_rows: torch.Tensor
    lifted_video_target_rows: torch.Tensor
    lift_receipt: dict[str, Any]
    high_target_source: str
    low_model_forwards: int
    high_model_forwards: int


def _initial_high_video_rows(
    branch: MiniMaxH3DenoiseBranch,
    target_rows: torch.Tensor,
    *,
    keyframe_rows: torch.Tensor | None,
) -> torch.Tensor:
    if keyframe_rows is None:
        return target_rows
    rows = torch.zeros(
        int(branch.img_pos.numel()),
        96,
        dtype=torch.float32,
        device=target_rows.device,
    )
    rows[branch.update_mask] = target_rows
    return rows


def minimax_h3_progressive_denoise_loop(
    *,
    model: Any,
    low_positive: MiniMaxH3DenoiseBranch,
    high_positive: MiniMaxH3DenoiseBranch,
    initial_low_video_rows: torch.Tensor,
    initial_audio_rows: torch.Tensor,
    low_keyframe_cond_rows: torch.Tensor | None,
    high_keyframe_cond_rows: torch.Tensor | None,
    sigmas_video: list[float],
    sigmas_audio: list[float],
    switch_position: int,
    plan: ProgressiveResolutionPlan,
    device: torch.device,
    lift_noise_seed: int,
    native_high_initial_video_target_rows: torch.Tensor | None = None,
    native_high_initial_audio_rows: torch.Tensor | None = None,
    on_low_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    on_resolution_lift: (
        Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], None] | None
    ) = None,
    on_high_step: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
    model_forward: (
        Callable[[Any, dict[str, Any], int], tuple[torch.Tensor, torch.Tensor]]
        | None
    ) = None,
) -> H3ProgressiveDenoiseResult:
    if len(sigmas_video) != len(sigmas_audio):
        raise ValueError("progressive video/audio sigma schedules must align")
    if not 0 < switch_position < len(sigmas_video) - 1:
        raise ValueError("progressive switch position must be an interior state")

    low_video_rows, audio_rows = minimax_h3_denoise_loop(
        model=model,
        model_forward=model_forward,
        positive=low_positive,
        initial_video_rows=initial_low_video_rows,
        initial_audio_rows=initial_audio_rows,
        keyframe_cond_rows=low_keyframe_cond_rows,
        sigmas_video=sigmas_video[: switch_position + 1],
        sigmas_audio=sigmas_audio[: switch_position + 1],
        device=device,
        on_step=on_low_step,
    )
    low_target = low_video_rows[low_positive.video_target_slice]
    lifted_target, lift_receipt = lift_video_state_rows(
        low_target,
        temporal_length=(
            int(low_target.shape[0])
            // plan.low.spatial_tokens_per_video_latent
        ),
        source_canvas=plan.low,
        target_canvas=plan.high,
        video_sigma=sigmas_video[switch_position],
        noise_seed=lift_noise_seed,
    )
    if on_resolution_lift is not None:
        on_resolution_lift(lifted_target, audio_rows, lift_receipt)

    anchored_high_target = native_high_initial_video_target_rows is not None
    if anchored_high_target != (native_high_initial_audio_rows is not None):
        raise ValueError(
            "native high-resolution video/audio initial states must be provided together"
        )
    if anchored_high_target:
        high_initial_video = _initial_high_video_rows(
            high_positive,
            native_high_initial_video_target_rows,
            keyframe_rows=high_keyframe_cond_rows,
        )

        def anchored_high_step(
            step: int,
            video_rows: torch.Tensor,
            current_audio_rows: torch.Tensor,
        ) -> None:
            # The low branch already supplied callbacks 0..32.  Only expose
            # the last 16 native-high forwards as callbacks 33..48 so the
            # retained state contract remains 0,16,33-low/lifted,49-high.
            if on_high_step is not None and step >= switch_position:
                on_high_step(
                    step - switch_position,
                    video_rows,
                    current_audio_rows,
                )

        high_video_rows, audio_rows = minimax_h3_denoise_loop(
            model=model,
            model_forward=model_forward,
            positive=high_positive,
            initial_video_rows=high_initial_video,
            initial_audio_rows=native_high_initial_audio_rows,
            keyframe_cond_rows=high_keyframe_cond_rows,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=device,
            on_step=anchored_high_step,
        )
        high_target_source = "native_high_full_trajectory_from_coupled_iid_noise"
        high_model_forwards = len(sigmas_video) - 1
    else:
        high_initial_video = _initial_high_video_rows(
            high_positive,
            lifted_target,
            keyframe_rows=high_keyframe_cond_rows,
        )
        high_video_rows, audio_rows = minimax_h3_denoise_loop(
            model=model,
            model_forward=model_forward,
            positive=high_positive,
            initial_video_rows=high_initial_video,
            initial_audio_rows=audio_rows,
            keyframe_cond_rows=high_keyframe_cond_rows,
            sigmas_video=sigmas_video[switch_position:],
            sigmas_audio=sigmas_audio[switch_position:],
            device=device,
            on_step=on_high_step,
        )
        high_target_source = "continued_from_lifted_low_state"
        high_model_forwards = len(sigmas_video) - switch_position - 1
    return H3ProgressiveDenoiseResult(
        video_rows=high_video_rows,
        audio_rows=audio_rows,
        lifted_video_target_rows=lifted_target,
        lift_receipt=lift_receipt,
        high_target_source=high_target_source,
        low_model_forwards=switch_position,
        high_model_forwards=high_model_forwards,
    )


__all__ = [
    "H3ProgressiveDenoiseResult",
    "minimax_h3_progressive_denoise_loop",
]
