"""Parameter-free full-prefix latent memory shared by training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..errors import SchemaValidationError
from .geometry import StreamPhase


FULL_PREFIX_VIDEO_LATENT_UNITS = 10
FULL_PREFIX_AUDIO_CHANNELS = 2
FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL = 57
FULL_PREFIX_VIDEO_ONLY = "video_only"
FULL_PREFIX_AUDIO_VIDEO = "audio_video"
FULL_PREFIX_MODALITIES = (
    FULL_PREFIX_VIDEO_ONLY,
    FULL_PREFIX_AUDIO_VIDEO,
)


@dataclass(frozen=True)
class FullPrefixAVLatents:
    """Canonical full-prefix video and channel-major audio latent rows."""

    video: Any
    audio: Any


def _validate_positive_seconds(value: float, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SchemaValidationError(f"{label} must be positive")
    return float(value)


def _detached_finite_float_tensor(value: Any, *, label: str) -> Any:
    import torch

    if (
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or value.numel() == 0
        or not bool(torch.isfinite(value.detach()).all().item())
    ):
        raise SchemaValidationError(f"{label} must be a non-empty floating tensor")
    return value.detach().float()


class FullPrefixAVEMA:
    """Detached FP32 EMA of one canonical audio-video latent block.

    Video and audio use the same update coefficient and are committed as one
    indivisible pair.  This prevents the two modalities from drifting to
    different source-block ages.
    """

    def __init__(self, *, half_life_seconds: float) -> None:
        self.half_life_seconds = _validate_positive_seconds(
            half_life_seconds,
            label="full-prefix half-life",
        )
        self._value: FullPrefixAVLatents | None = None

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def view(self) -> FullPrefixAVLatents | None:
        if self._value is None:
            return None
        return FullPrefixAVLatents(
            video=self._value.video.detach(),
            audio=self._value.audio.detach(),
        )

    def update(
        self,
        canonical_clean: FullPrefixAVLatents,
        *,
        block_seconds: float,
    ) -> None:
        if not isinstance(canonical_clean, FullPrefixAVLatents):
            raise SchemaValidationError(
                "canonical full-prefix update must contain video and audio latents"
            )
        clean = FullPrefixAVLatents(
            video=_detached_finite_float_tensor(
                canonical_clean.video,
                label="canonical full-prefix video",
            ),
            audio=_detached_finite_float_tensor(
                canonical_clean.audio,
                label="canonical full-prefix audio",
            ),
        )
        if (
            clean.video.ndim != 2
            or int(clean.video.shape[0]) % FULL_PREFIX_VIDEO_LATENT_UNITS
            or clean.audio.ndim != 2
            or int(clean.audio.shape[0])
            != FULL_PREFIX_AUDIO_CHANNELS
            * FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
            or clean.video.device != clean.audio.device
        ):
            raise SchemaValidationError(
                "canonical full-prefix AV shapes/device differ from the native contract"
            )
        seconds = _validate_positive_seconds(block_seconds, label="block_seconds")
        if self._value is None:
            self._value = FullPrefixAVLatents(
                video=clean.video.clone(),
                audio=clean.audio.clone(),
            )
            return
        for label, current, previous in (
            ("video", clean.video, self._value.video),
            ("audio", clean.audio, self._value.audio),
        ):
            if tuple(current.shape) != tuple(previous.shape) or current.device != previous.device:
                raise SchemaValidationError(
                    f"canonical full-prefix {label} shape/device changed within a sequence"
                )
        beta = 1.0 - 2.0 ** (-seconds / self.half_life_seconds)
        self._value.video.mul_(1.0 - beta).add_(clean.video, alpha=beta)
        self._value.audio.mul_(1.0 - beta).add_(clean.audio, alpha=beta)

    def reset(self) -> None:
        self._value = None


class FullPrefixEMA:
    """Detached FP32 EMA of one canonical ten-unit video-latent block."""

    def __init__(self, *, half_life_seconds: float) -> None:
        self.half_life_seconds = _validate_positive_seconds(
            half_life_seconds,
            label="full-prefix half-life",
        )
        self._value: Any | None = None

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def view(self) -> Any | None:
        return None if self._value is None else self._value.detach()

    def update(self, canonical_clean_video: Any, *, block_seconds: float) -> None:
        clean = _detached_finite_float_tensor(
            canonical_clean_video,
            label="canonical full-prefix video",
        )
        seconds = _validate_positive_seconds(block_seconds, label="block_seconds")
        if self._value is None:
            self._value = clean.clone()
            return
        if tuple(clean.shape) != tuple(self._value.shape) or clean.device != self._value.device:
            raise SchemaValidationError(
                "canonical full-prefix shape/device changed within a sequence"
            )
        beta = 1.0 - 2.0 ** (-seconds / self.half_life_seconds)
        self._value.mul_(1.0 - beta).add_(clean, alpha=beta)

    def reset(self) -> None:
        self._value = None


def create_full_prefix_memory(
    *,
    modality: str,
    half_life_seconds: float,
) -> FullPrefixEMA | FullPrefixAVEMA:
    """Create the parameter-free latent EMA selected by the runtime contract."""

    if modality == FULL_PREFIX_VIDEO_ONLY:
        return FullPrefixEMA(half_life_seconds=half_life_seconds)
    if modality == FULL_PREFIX_AUDIO_VIDEO:
        return FullPrefixAVEMA(half_life_seconds=half_life_seconds)
    raise SchemaValidationError(
        "full-prefix modality must be video_only or audio_video"
    )


def canonical_full_prefix_rows(
    clean_phase_video_rows: Any,
    *,
    phase: StreamPhase,
    frame_rows: int,
) -> Any:
    """Take the last ten temporal latent units from a clean streaming phase."""

    import torch

    expected = phase.video_latent_count * frame_rows
    if (
        not torch.is_tensor(clean_phase_video_rows)
        or clean_phase_video_rows.ndim != 2
        or int(clean_phase_video_rows.shape[0]) != expected
        or phase.video_latent_count < FULL_PREFIX_VIDEO_LATENT_UNITS
    ):
        raise SchemaValidationError(
            "clean streaming phase cannot supply canonical full-prefix rows"
        )
    return (
        clean_phase_video_rows.reshape(
            phase.video_latent_count,
            frame_rows,
            int(clean_phase_video_rows.shape[-1]),
        )[-FULL_PREFIX_VIDEO_LATENT_UNITS:]
        .reshape(-1, int(clean_phase_video_rows.shape[-1]))
        .detach()
        .float()
        .clone()
    )


def canonical_full_prefix_audio_rows(
    clean_phase_audio_rows: Any,
    *,
    audio_latent_units_per_channel: int | None = None,
) -> Any:
    """Canonicalize clean channel-major audio to exactly 57 rows per channel.

    Native two-group H3 blocks contribute either 65 rows per channel for the
    initial block or 56/57 rows for later boundaries.  A 56-row boundary is
    left padded by repeating its first row; longer blocks retain the newest 57.
    """

    import torch

    if not torch.is_tensor(clean_phase_audio_rows) or clean_phase_audio_rows.ndim != 2:
        raise SchemaValidationError(
            "clean streaming audio must be a packed rank-two tensor"
        )
    total_rows = int(clean_phase_audio_rows.shape[0])
    if total_rows % FULL_PREFIX_AUDIO_CHANNELS:
        raise SchemaValidationError(
            "clean streaming audio rows do not contain two complete channels"
        )
    inferred = total_rows // FULL_PREFIX_AUDIO_CHANNELS
    if audio_latent_units_per_channel is None:
        audio_latent_units_per_channel = inferred
    if (
        isinstance(audio_latent_units_per_channel, bool)
        or not isinstance(audio_latent_units_per_channel, int)
        or audio_latent_units_per_channel != inferred
    ):
        raise SchemaValidationError(
            "clean streaming audio rows differ from the declared channel length"
        )
    if audio_latent_units_per_channel < (
        FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL - 1
    ):
        raise SchemaValidationError(
            "clean streaming audio cannot supply canonical full-prefix rows"
        )
    shaped = clean_phase_audio_rows.reshape(
        FULL_PREFIX_AUDIO_CHANNELS,
        audio_latent_units_per_channel,
        int(clean_phase_audio_rows.shape[-1]),
    )
    if audio_latent_units_per_channel == (
        FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL - 1
    ):
        shaped = torch.cat((shaped[:, :1], shaped), dim=1)
    else:
        shaped = shaped[:, -FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL:]
    return (
        shaped.reshape(
            FULL_PREFIX_AUDIO_CHANNELS
            * FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL,
            int(clean_phase_audio_rows.shape[-1]),
        )
        .detach()
        .float()
        .clone()
    )


def canonical_full_prefix_av_latents(
    clean_phase_video_rows: Any,
    clean_phase_audio_rows: Any,
    *,
    phase: StreamPhase,
    frame_rows: int,
) -> FullPrefixAVLatents:
    """Build the synchronized canonical AV latent pair for one clean block."""

    return FullPrefixAVLatents(
        video=canonical_full_prefix_rows(
            clean_phase_video_rows,
            phase=phase,
            frame_rows=frame_rows,
        ),
        audio=canonical_full_prefix_audio_rows(
            clean_phase_audio_rows,
            audio_latent_units_per_channel=phase.audio_latent_count,
        ),
    )


def place_full_prefix_rows(
    canonical_rows: Any,
    *,
    phase: StreamPhase,
    frame_rows: int,
) -> Any:
    """Place canonical memory in the first H3 phase canvas used for K/V capture."""

    import torch

    expected = FULL_PREFIX_VIDEO_LATENT_UNITS * frame_rows
    if (
        not torch.is_tensor(canonical_rows)
        or canonical_rows.ndim != 2
        or int(canonical_rows.shape[0]) != expected
        or phase.video_latent_count < FULL_PREFIX_VIDEO_LATENT_UNITS
    ):
        raise SchemaValidationError("canonical full-prefix packed rows differ")
    shaped = canonical_rows.reshape(
        FULL_PREFIX_VIDEO_LATENT_UNITS,
        frame_rows,
        int(canonical_rows.shape[-1]),
    )
    pad_units = phase.video_latent_count - FULL_PREFIX_VIDEO_LATENT_UNITS
    return torch.cat((shaped[:1].expand(pad_units, -1, -1), shaped), dim=0).reshape(
        phase.video_latent_count * frame_rows,
        int(canonical_rows.shape[-1]),
    )


def place_full_prefix_audio_rows(
    canonical_rows: Any,
    *,
    audio_latent_units_per_channel: int,
) -> Any:
    """Place canonical audio in a channel-major first-block audio canvas."""

    import torch

    expected = (
        FULL_PREFIX_AUDIO_CHANNELS
        * FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
    )
    if (
        not torch.is_tensor(canonical_rows)
        or canonical_rows.ndim != 2
        or int(canonical_rows.shape[0]) != expected
        or isinstance(audio_latent_units_per_channel, bool)
        or not isinstance(audio_latent_units_per_channel, int)
        or audio_latent_units_per_channel
        < FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
    ):
        raise SchemaValidationError("canonical full-prefix audio rows differ")
    shaped = canonical_rows.reshape(
        FULL_PREFIX_AUDIO_CHANNELS,
        FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL,
        int(canonical_rows.shape[-1]),
    )
    pad_units = (
        audio_latent_units_per_channel
        - FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL
    )
    return torch.cat(
        (shaped[:, :1].expand(-1, pad_units, -1), shaped),
        dim=1,
    ).reshape(
        FULL_PREFIX_AUDIO_CHANNELS * audio_latent_units_per_channel,
        int(canonical_rows.shape[-1]),
    )


def place_full_prefix_av_latents(
    canonical: FullPrefixAVLatents,
    *,
    phase: StreamPhase,
    frame_rows: int,
) -> FullPrefixAVLatents:
    """Place synchronized canonical AV memory in the capture phase canvas."""

    if not isinstance(canonical, FullPrefixAVLatents):
        raise SchemaValidationError(
            "full-prefix placement requires canonical video and audio latents"
        )
    return FullPrefixAVLatents(
        video=place_full_prefix_rows(
            canonical.video,
            phase=phase,
            frame_rows=frame_rows,
        ),
        audio=place_full_prefix_audio_rows(
            canonical.audio,
            audio_latent_units_per_channel=phase.audio_latent_count,
        ),
    )


__all__ = [
    "FULL_PREFIX_AUDIO_VIDEO",
    "FULL_PREFIX_AUDIO_CHANNELS",
    "FULL_PREFIX_AUDIO_LATENT_UNITS_PER_CHANNEL",
    "FULL_PREFIX_MODALITIES",
    "FULL_PREFIX_VIDEO_ONLY",
    "FULL_PREFIX_VIDEO_LATENT_UNITS",
    "FullPrefixAVEMA",
    "FullPrefixAVLatents",
    "FullPrefixEMA",
    "canonical_full_prefix_audio_rows",
    "canonical_full_prefix_av_latents",
    "canonical_full_prefix_rows",
    "create_full_prefix_memory",
    "place_full_prefix_audio_rows",
    "place_full_prefix_av_latents",
    "place_full_prefix_rows",
]
