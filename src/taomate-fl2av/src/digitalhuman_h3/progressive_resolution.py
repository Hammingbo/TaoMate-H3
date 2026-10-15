# SPDX-License-Identifier: Apache-2.0
"""Shared progressive-resolution contracts for H3 ODE, training, and inference."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .denoise_schedule import DISTILLED_STATE_INDICES


PROGRESSIVE_RESOLUTION_SCHEMA = "digitalhuman_h3.progressive_resolution.v1"
PROGRESSIVE_LIFT_SCHEMA = "digitalhuman_h3.progressive_resolution_lift.v1"
PROGRESSIVE_TEACHER_TARGET_POLICY = (
    "coupled_iid_480p_native_prefix_plus_768p_native_endpoint_v1"
)
PROGRESSIVE_SWITCH_STATE_INDEX = 33
VIDEO_LATENT_SPATIAL_COMPRESSION = 16
VIDEO_PATCH_SPATIAL_SIZE = 2
CANVAS_MULTIPLE = VIDEO_LATENT_SPATIAL_COMPRESSION * VIDEO_PATCH_SPATIAL_SIZE


@dataclass(frozen=True)
class ProgressiveCanvas:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or not isinstance(self.width, int)
            or isinstance(self.height, bool)
            or not isinstance(self.height, int)
            or self.width <= 0
            or self.height <= 0
            or self.width % CANVAS_MULTIPLE
            or self.height % CANVAS_MULTIPLE
        ):
            raise ValueError(
                f"H3 progressive canvas must use positive multiples of {CANVAS_MULTIPLE}"
            )

    @property
    def latent_width(self) -> int:
        return self.width // VIDEO_LATENT_SPATIAL_COMPRESSION

    @property
    def latent_height(self) -> int:
        return self.height // VIDEO_LATENT_SPATIAL_COMPRESSION

    @property
    def token_width(self) -> int:
        return self.width // CANVAS_MULTIPLE

    @property
    def token_height(self) -> int:
        return self.height // CANVAS_MULTIPLE

    @property
    def spatial_tokens_per_video_latent(self) -> int:
        return self.token_width * self.token_height

    def as_dict(self) -> dict[str, int]:
        return {
            **asdict(self),
            "latent_width": self.latent_width,
            "latent_height": self.latent_height,
            "token_width": self.token_width,
            "token_height": self.token_height,
            "spatial_tokens_per_video_latent": self.spatial_tokens_per_video_latent,
        }


@dataclass(frozen=True)
class ProgressiveResolutionPlan:
    """Three student transitions with one explicit lift before transition three.

    The student keeps state 33 at low resolution until the second transition
    is complete.  A deterministic lift creates the high-resolution input to
    the third transition.  ODE extraction pairs that lifted input with the
    endpoint of a native-high teacher trajectory driven by the same coupled
    Gaussian source.  The high-frequency part removed by the low-resolution
    projection is restored at the lift before transition three.  Directly
    continuing the base teacher from that lifted state is not a valid teacher
    path; the distilled student is trained on the cross-resolution pair.
    Both forms of state 33 must be saved.
    """

    low: ProgressiveCanvas
    high: ProgressiveCanvas
    state_indices: tuple[int, int, int, int] = DISTILLED_STATE_INDICES
    switch_state_index: int = PROGRESSIVE_SWITCH_STATE_INDEX

    def __post_init__(self) -> None:
        if self.state_indices != DISTILLED_STATE_INDICES:
            raise ValueError("progressive H3 must retain teacher states 0,16,33,49")
        if self.switch_state_index != self.state_indices[2]:
            raise ValueError("progressive H3 resolution lift must occur at state 33")
        if self.high.width <= self.low.width or self.high.height <= self.low.height:
            raise ValueError("progressive high canvas must be larger on both spatial axes")
        # Both canvases are rounded independently to the nearest DiT token grid.
        # Therefore their post-rounding ratios need not agree within a fixed
        # percentage (for example, 3:2 becomes 736x480 and 1152x768).  Accept
        # the pair when the two rounding intervals contain a common source
        # aspect ratio.
        half_grid = CANVAS_MULTIPLE / 2

        def source_aspect_interval(canvas: ProgressiveCanvas) -> tuple[float, float]:
            return (
                (canvas.width - half_grid) / (canvas.height + half_grid),
                (canvas.width + half_grid) / (canvas.height - half_grid),
            )

        low_min, low_max = source_aspect_interval(self.low)
        high_min, high_max = source_aspect_interval(self.high)
        if max(low_min, high_min) > min(low_max, high_max):
            raise ValueError("progressive low/high canvases must preserve the source aspect")

    @property
    def transition_input_canvases(self) -> tuple[ProgressiveCanvas, ...]:
        return (self.low, self.low, self.high)

    @property
    def transition_output_canvases(self) -> tuple[ProgressiveCanvas, ...]:
        return (self.low, self.low, self.high)

    @property
    def canonical_history_cache_canvas(self) -> ProgressiveCanvas:
        """Use one final-resolution clean KV cache across all three transitions.

        Denoise-step K/V for the active chunk is transient and is never placed
        in history.  After a chunk reaches sigma zero, one high-resolution
        clean forward commits its K/V.  The next chunk's low-resolution Q can
        attend to those high-resolution K/V rows directly because H3 spatial
        RoPE uses sqrt-area-normalized coordinates and rectangular attention
        does not require equal query/key sequence lengths.  No K/V resize or
        historical transformer replay is involved at the state-33 lift.
        """

        return self.high

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PROGRESSIVE_RESOLUTION_SCHEMA,
            "state_indices": list(self.state_indices),
            "switch_state_index": self.switch_state_index,
            "low": self.low.as_dict(),
            "high": self.high.as_dict(),
            "transition_input_canvases": [
                canvas.as_dict() for canvas in self.transition_input_canvases
            ],
            "transition_output_canvases": [
                canvas.as_dict() for canvas in self.transition_output_canvases
            ],
            "saved_video_states": [
                "video_state_000",
                "video_state_016",
                "video_state_033_low",
                "video_state_033_lifted",
                "video_state_049",
            ],
            "audio_state_indices": list(self.state_indices),
            "canonical_history_cache": {
                "canvas": self.canonical_history_cache_canvas.as_dict(),
                "commit_sigma": 0.0,
                "model_timestep": 1.0,
                "model_timestep_convention": "one_minus_sigma",
                "reuse_across_all_transitions": True,
                "rope_coordinates": "sqrt_area_normalized",
            },
        }


def build_progressive_resolution_plan(
    *,
    low_width: int,
    low_height: int,
    high_width: int,
    high_height: int,
) -> ProgressiveResolutionPlan:
    return ProgressiveResolutionPlan(
        low=ProgressiveCanvas(width=low_width, height=low_height),
        high=ProgressiveCanvas(width=high_width, height=high_height),
    )


def progressive_transition_resolution_key(state_index: int) -> str:
    """Return the canvas used by a retained progressive student transition."""

    if state_index not in DISTILLED_STATE_INDICES[:-1]:
        raise ValueError("progressive transition state must be 0, 16, or 33")
    return "high" if state_index == PROGRESSIVE_SWITCH_STATE_INDEX else "low"


def _resize_spatial(latent: Any, *, height: int, width: int, mode: str) -> Any:
    import torch.nn.functional as functional

    batch, channels, temporal, source_h, source_w = latent.shape
    frames = latent.permute(0, 2, 1, 3, 4).reshape(
        batch * temporal, channels, source_h, source_w
    )
    if mode == "area":
        resized = functional.interpolate(frames, size=(height, width), mode=mode)
    else:
        resized = functional.interpolate(
            frames,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )
    return resized.reshape(batch, temporal, channels, height, width).permute(
        0, 2, 1, 3, 4
    ).contiguous()


@lru_cache(maxsize=16)
def _orthonormal_lowpass_projection(
    low_size: int,
    high_size: int,
) -> Any:
    """Return a low-pass matrix whose rows are orthonormal.

    Applying the matrix to an iid standard Gaussian therefore produces an
    iid standard Gaussian at the smaller resolution while retaining the
    strongest possible linear low-frequency coupling to the high-resolution
    source.
    """

    import torch
    import torch.nn.functional as functional

    if low_size <= 0 or high_size <= low_size:
        raise ValueError("orthonormal projection requires 0 < low < high")
    basis = torch.eye(high_size, dtype=torch.float64).reshape(
        high_size, 1, high_size
    )
    resize = functional.interpolate(
        basis,
        size=low_size,
        mode="linear",
        align_corners=False,
    )[:, 0, :].transpose(0, 1)
    gram = resize @ resize.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    inverse_sqrt = eigenvectors @ torch.diag(
        eigenvalues.clamp_min(1.0e-12).rsqrt()
    ) @ eigenvectors.transpose(0, 1)
    projection = inverse_sqrt @ resize
    return projection.to(dtype=torch.float32).contiguous()


def coupled_progressive_video_noise_rows(
    *,
    temporal_length: int,
    temporal_offset: int = 0,
    source_canvas: ProgressiveCanvas,
    target_canvas: ProgressiveCanvas,
    noise_seed: int,
) -> tuple[Any, Any, dict[str, Any]]:
    """Create coupled low/high iid Gaussian video noise.

    High-resolution noise is sampled directly from N(0,I).  A separable
    orthonormal low-pass projection creates the low-resolution noise, so its
    covariance is also exactly I.  Noise is indexed per temporal latent unit;
    full-clip and streaming-block reconstruction are therefore identical.
    """

    import torch

    from .model.packed_tokens import minimax_h3_patchify_video_latent

    if (
        isinstance(temporal_length, bool)
        or not isinstance(temporal_length, int)
        or temporal_length <= 0
    ):
        raise ValueError("temporal_length must be a positive integer")
    if (
        isinstance(temporal_offset, bool)
        or not isinstance(temporal_offset, int)
        or temporal_offset < 0
    ):
        raise ValueError("temporal_offset must be a non-negative integer")
    if (
        isinstance(noise_seed, bool)
        or not isinstance(noise_seed, int)
        or noise_seed < 0
    ):
        raise ValueError("noise_seed must be a non-negative integer")
    if (
        target_canvas.width <= source_canvas.width
        or target_canvas.height <= source_canvas.height
    ):
        raise ValueError("target canvas must be larger than source canvas")

    high_slices = []
    for temporal_index in range(
        temporal_offset,
        temporal_offset + temporal_length,
    ):
        generator = torch.Generator(device="cpu").manual_seed(
            (noise_seed + temporal_index * 1_000_003) % (2**63)
        )
        high_slices.append(
            torch.randn(
                1,
                24,
                1,
                target_canvas.latent_height,
                target_canvas.latent_width,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
        )
    high_latent = torch.cat(high_slices, dim=2)
    height_projection = _orthonormal_lowpass_projection(
        source_canvas.latent_height,
        target_canvas.latent_height,
    )
    width_projection = _orthonormal_lowpass_projection(
        source_canvas.latent_width,
        target_canvas.latent_width,
    )
    low_height = torch.einsum(
        "lh,bcthw->bctlw",
        height_projection,
        high_latent,
    )
    low_latent = torch.einsum(
        "kw,bctlw->bctlk",
        width_projection,
        low_height,
    ).contiguous()
    low_rows = minimax_h3_patchify_video_latent(
        low_latent,
        patch_size=(1, 2, 2),
    ).contiguous()
    high_rows = minimax_h3_patchify_video_latent(
        high_latent,
        patch_size=(1, 2, 2),
    ).contiguous()
    return low_rows, high_rows, {
        "schema": "digitalhuman_h3.progressive_noise_coupling.v1",
        "policy": "separable_orthonormal_lowpass_projection",
        "noise_seed": noise_seed,
        "temporal_offset": temporal_offset,
        "temporal_length": temporal_length,
        "low_marginal": "iid_standard_gaussian",
        "high_marginal": "iid_standard_gaussian",
        "shared_random_source": True,
    }


def lift_video_state_rows(
    rows: Any,
    *,
    temporal_length: int,
    temporal_offset: int = 0,
    source_canvas: ProgressiveCanvas,
    target_canvas: ProgressiveCanvas,
    video_sigma: float,
    noise_seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Lift one packed state and restore its coupled high-frequency noise.

    The residual is not independent noise.  It is the part of the same native
    high-resolution Gaussian source that is absent after projecting to the
    low canvas.  At sigma=1 the lifted low noise exactly reconstructs the
    native high noise; at sigma=0 this reduces to plain bilinear resize.
    """

    from .model.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    if (
        isinstance(temporal_length, bool)
        or not isinstance(temporal_length, int)
        or temporal_length <= 0
    ):
        raise ValueError("temporal_length must be a positive integer")
    if (
        isinstance(temporal_offset, bool)
        or not isinstance(temporal_offset, int)
        or temporal_offset < 0
    ):
        raise ValueError("temporal_offset must be a non-negative integer")
    if (
        isinstance(video_sigma, bool)
        or not isinstance(video_sigma, (int, float))
        or not math.isfinite(float(video_sigma))
        or not 0.0 <= float(video_sigma) <= 1.0
    ):
        raise ValueError("video_sigma must be finite and in [0,1]")
    if (
        isinstance(noise_seed, bool)
        or not isinstance(noise_seed, int)
        or noise_seed < 0
    ):
        raise ValueError("noise_seed must be a non-negative integer")
    if (
        target_canvas.width <= source_canvas.width
        or target_canvas.height <= source_canvas.height
    ):
        raise ValueError("target canvas must be larger than source canvas")

    latent = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=(
            temporal_length,
            source_canvas.token_height,
            source_canvas.token_width,
            24,
        ),
        patch_size=(1, 2, 2),
    ).float()
    interpolated = _resize_spatial(
        latent,
        height=target_canvas.latent_height,
        width=target_canvas.latent_width,
        mode="bilinear",
    )
    low_noise_rows, high_noise_rows, coupling_receipt = (
        coupled_progressive_video_noise_rows(
            temporal_length=temporal_length,
            temporal_offset=temporal_offset,
            source_canvas=source_canvas,
            target_canvas=target_canvas,
            noise_seed=noise_seed,
        )
    )
    low_noise = minimax_h3_unpatchify_video_tokens(
        low_noise_rows,
        latent_shape=(
            temporal_length,
            source_canvas.token_height,
            source_canvas.token_width,
            24,
        ),
        patch_size=(1, 2, 2),
    ).float()
    high_noise = minimax_h3_unpatchify_video_tokens(
        high_noise_rows,
        latent_shape=(
            temporal_length,
            target_canvas.token_height,
            target_canvas.token_width,
            24,
        ),
        patch_size=(1, 2, 2),
    ).float()
    interpolated_low_noise = _resize_spatial(
        low_noise,
        height=target_canvas.latent_height,
        width=target_canvas.latent_width,
        mode="bilinear",
    )
    coupled_residual = (high_noise - interpolated_low_noise).to(
        device=interpolated.device,
        dtype=interpolated.dtype,
    )
    lifted = interpolated + float(video_sigma) * coupled_residual
    packed = minimax_h3_patchify_video_latent(
        lifted,
        patch_size=(1, 2, 2),
    ).contiguous()
    expected_rows = temporal_length * target_canvas.spatial_tokens_per_video_latent
    if tuple(packed.shape) != (expected_rows, 96):
        raise RuntimeError("progressive lift produced an unexpected packed video shape")
    return packed, {
        "schema": PROGRESSIVE_LIFT_SCHEMA,
        "state_index": PROGRESSIVE_SWITCH_STATE_INDEX,
        "source_canvas": source_canvas.as_dict(),
        "target_canvas": target_canvas.as_dict(),
        "video_sigma": float(video_sigma),
        "noise_seed": noise_seed,
        "temporal_offset": temporal_offset,
        "noise_indexing": "per_temporal_latent_unit",
        "spatial_interpolation": "bilinear_align_corners_false",
        "high_frequency_residual": (
            "native_high_noise_minus_bilinear_projected_low_noise"
        ),
        "state_variance_policy": "coupled_native_high_noise_reconstruction",
        "initial_noise_coupling": coupling_receipt,
        "audio_transform": "identity",
    }


def resize_video_state_rows(
    rows: Any,
    *,
    temporal_length: int,
    source_canvas: ProgressiveCanvas,
    target_canvas: ProgressiveCanvas,
) -> Any:
    """Resize packed H3 video rows without changing temporal state or sigma."""

    from .model.packed_tokens import (
        minimax_h3_patchify_video_latent,
        minimax_h3_unpatchify_video_tokens,
    )

    latent = minimax_h3_unpatchify_video_tokens(
        rows,
        latent_shape=(
            temporal_length,
            source_canvas.token_height,
            source_canvas.token_width,
            24,
        ),
        patch_size=(1, 2, 2),
    ).float()
    mode = (
        "area"
        if target_canvas.height < source_canvas.height
        and target_canvas.width < source_canvas.width
        else "bilinear"
    )
    resized = _resize_spatial(
        latent,
        height=target_canvas.latent_height,
        width=target_canvas.latent_width,
        mode=mode,
    )
    return minimax_h3_patchify_video_latent(
        resized,
        patch_size=(1, 2, 2),
    ).contiguous()


__all__ = [
    "CANVAS_MULTIPLE",
    "PROGRESSIVE_LIFT_SCHEMA",
    "PROGRESSIVE_RESOLUTION_SCHEMA",
    "PROGRESSIVE_SWITCH_STATE_INDEX",
    "PROGRESSIVE_TEACHER_TARGET_POLICY",
    "ProgressiveCanvas",
    "ProgressiveResolutionPlan",
    "build_progressive_resolution_plan",
    "coupled_progressive_video_noise_rows",
    "lift_video_state_rows",
    "progressive_transition_resolution_key",
    "resize_video_state_rows",
]
