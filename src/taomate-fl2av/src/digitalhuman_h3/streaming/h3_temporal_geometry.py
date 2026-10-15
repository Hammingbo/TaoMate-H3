from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from ..errors import SchemaValidationError


NATIVE_H3_REFERENCE_REVISION = "70fe2e0dd5e69a062fb146d0db35c7ac939f111f"

VIDEO_FPS = 24
AUDIO_SAMPLE_RATE_HZ = 32_000
AUDIO_LATENT_RATE_HZ = 40
VIDEO_PREFIX_FRAMES = 5
VIDEO_PREFIX_LATENT_UNITS = 2
VIDEO_GROUP_FRAMES = 17
VIDEO_GROUP_LATENT_UNITS = 5
MINIMUM_STEADY_BLOCK_SECONDS = Fraction(1, 1)
MAXIMUM_STEADY_BLOCK_SECONDS = Fraction(2, 1)


def round_fraction_half_even(value: Fraction) -> int:
    """Exactly reproduce Python's ``round`` for a non-negative fraction."""

    if not isinstance(value, Fraction) or value < 0:
        raise SchemaValidationError(
            "H3 temporal rounding requires a non-negative fraction"
        )
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def align_frame_count(frame_count: int) -> int:
    """Match the pinned H3 ``17n+5`` upward-alignment rule."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise SchemaValidationError("H3 frame count must be an integer")
    if frame_count <= 0:
        return 1
    return frame_count + (VIDEO_PREFIX_FRAMES - frame_count) % VIDEO_GROUP_FRAMES


def _require_aligned_frame_count(frame_count: int) -> None:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise SchemaValidationError("H3 frame count must be an integer")
    if frame_count == 1:
        return
    if (
        frame_count < VIDEO_PREFIX_FRAMES
        or (frame_count - VIDEO_PREFIX_FRAMES) % VIDEO_GROUP_FRAMES
    ):
        raise SchemaValidationError(
            "H3 native frame count must be 1 or satisfy the pinned 17n+5 rule"
        )


def video_latent_units_for_aligned_frames(frame_count: int) -> int:
    _require_aligned_frame_count(frame_count)
    if frame_count == 1:
        return 1
    groups = (frame_count - VIDEO_PREFIX_FRAMES) // VIDEO_GROUP_FRAMES
    return VIDEO_PREFIX_LATENT_UNITS + groups * VIDEO_GROUP_LATENT_UNITS


def audio_latent_boundary_for_frames(frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise SchemaValidationError("H3 frame count must be an integer")
    if frame_count < 0:
        raise SchemaValidationError("H3 frame count must be non-negative")
    return round_fraction_half_even(
        Fraction(frame_count * AUDIO_LATENT_RATE_HZ, VIDEO_FPS)
    )


def audio_sample_boundary_for_frames(frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int):
        raise SchemaValidationError("H3 frame count must be an integer")
    if frame_count < 0:
        raise SchemaValidationError("H3 frame count must be non-negative")
    return round_fraction_half_even(
        Fraction(frame_count * AUDIO_SAMPLE_RATE_HZ, VIDEO_FPS)
    )


@dataclass(frozen=True)
class H3NativeClipGeometry:
    frame_count: int
    video_latent_units: int
    audio_latent_units_per_channel: int
    audio_samples_per_channel: int

    @property
    def duration_seconds(self) -> Fraction:
        return Fraction(self.frame_count, VIDEO_FPS)


def native_clip_geometry(frame_count: int) -> H3NativeClipGeometry:
    """Resolve one already-aligned native H3 clip without float arithmetic."""

    _require_aligned_frame_count(frame_count)
    return H3NativeClipGeometry(
        frame_count=frame_count,
        video_latent_units=video_latent_units_for_aligned_frames(frame_count),
        audio_latent_units_per_channel=audio_latent_boundary_for_frames(frame_count),
        audio_samples_per_channel=audio_sample_boundary_for_frames(frame_count),
    )


def structural_steady_group_candidates(
    *,
    minimum_seconds: Fraction = MINIMUM_STEADY_BLOCK_SECONDS,
    maximum_seconds: Fraction = MAXIMUM_STEADY_BLOCK_SECONDS,
) -> tuple[int, ...]:
    """Return group counts whose first and steady blocks both fit the range."""

    if (
        not isinstance(minimum_seconds, Fraction)
        or not isinstance(maximum_seconds, Fraction)
        or minimum_seconds <= 0
        or maximum_seconds < minimum_seconds
    ):
        raise SchemaValidationError("H3 block-duration range is invalid")
    candidates: list[int] = []
    for groups in range(1, 129):
        steady = Fraction(groups * VIDEO_GROUP_FRAMES, VIDEO_FPS)
        first = Fraction(
            VIDEO_PREFIX_FRAMES + groups * VIDEO_GROUP_FRAMES,
            VIDEO_FPS,
        )
        if steady > maximum_seconds and first > maximum_seconds:
            break
        if (
            minimum_seconds <= steady <= maximum_seconds
            and minimum_seconds <= first <= maximum_seconds
        ):
            candidates.append(groups)
    return tuple(candidates)


@dataclass(frozen=True)
class H3StreamingBlockGeometry:
    block_index: int
    group_count: int
    first_block: bool
    start_frame: int
    end_frame: int
    start_video_latent_unit: int
    end_video_latent_unit: int
    start_audio_latent_unit_per_channel: int
    end_audio_latent_unit_per_channel: int
    start_audio_sample_per_channel: int
    end_audio_sample_per_channel: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def duration_seconds(self) -> Fraction:
        return Fraction(self.frame_count, VIDEO_FPS)

    @property
    def video_latent_units(self) -> int:
        return self.end_video_latent_unit - self.start_video_latent_unit

    @property
    def audio_latent_units_per_channel(self) -> int:
        return (
            self.end_audio_latent_unit_per_channel
            - self.start_audio_latent_unit_per_channel
        )

    @property
    def audio_samples_per_channel(self) -> int:
        return self.end_audio_sample_per_channel - self.start_audio_sample_per_channel


def build_group_block_plan(
    group_counts: Sequence[int],
    *,
    allow_short_terminal_flush: bool = False,
    allow_subsecond_blocks: bool = False,
    maximum_seconds: Fraction = MAXIMUM_STEADY_BLOCK_SECONDS,
) -> tuple[H3StreamingBlockGeometry, ...]:
    """Build phase-aware AV boundaries from native H3 temporal groups.

    The first block owns the codec's affine 5-frame/2-latent prefix. Audio
    boundaries are rounded cumulatively, never independently per block, so the
    40 Hz stream cannot drift against 24 fps video.
    """

    if isinstance(group_counts, (str, bytes)) or not isinstance(
        group_counts, Sequence
    ):
        raise SchemaValidationError("H3 group counts must be a sequence")
    if not group_counts:
        raise SchemaValidationError("H3 block plan must contain at least one block")
    parsed: list[int] = []
    for index, value in enumerate(group_counts):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SchemaValidationError(
                f"H3 block {index} group count must be a positive integer"
            )
        duration = Fraction(
            value * VIDEO_GROUP_FRAMES
            + (VIDEO_PREFIX_FRAMES if index == 0 else 0),
            VIDEO_FPS,
        )
        is_short_terminal = (
            allow_short_terminal_flush
            and index == len(group_counts) - 1
            and duration < MINIMUM_STEADY_BLOCK_SECONDS
        )
        lower_bound_satisfied = (
            allow_subsecond_blocks
            or duration >= MINIMUM_STEADY_BLOCK_SECONDS
        )
        if (
            not is_short_terminal
            and not lower_bound_satisfied
        ) or duration > maximum_seconds:
            raise SchemaValidationError(
                f"H3 block {index} duration {duration} escapes the selected duration gate"
            )
        parsed.append(value)

    blocks: list[H3StreamingBlockGeometry] = []
    end_frame = 0
    end_video_latent = 0
    for index, groups in enumerate(parsed):
        start_frame = end_frame
        start_video_latent = end_video_latent
        new_frames = groups * VIDEO_GROUP_FRAMES
        new_video_latents = groups * VIDEO_GROUP_LATENT_UNITS
        if index == 0:
            new_frames += VIDEO_PREFIX_FRAMES
            new_video_latents += VIDEO_PREFIX_LATENT_UNITS
        end_frame += new_frames
        end_video_latent += new_video_latents
        start_audio_latent = audio_latent_boundary_for_frames(start_frame)
        end_audio_latent = audio_latent_boundary_for_frames(end_frame)
        start_audio_sample = audio_sample_boundary_for_frames(start_frame)
        end_audio_sample = audio_sample_boundary_for_frames(end_frame)
        block = H3StreamingBlockGeometry(
            block_index=index,
            group_count=groups,
            first_block=index == 0,
            start_frame=start_frame,
            end_frame=end_frame,
            start_video_latent_unit=start_video_latent,
            end_video_latent_unit=end_video_latent,
            start_audio_latent_unit_per_channel=start_audio_latent,
            end_audio_latent_unit_per_channel=end_audio_latent,
            start_audio_sample_per_channel=start_audio_sample,
            end_audio_sample_per_channel=end_audio_sample,
        )
        expected = native_clip_geometry(end_frame)
        if (
            expected.video_latent_units != end_video_latent
            or expected.audio_latent_units_per_channel != end_audio_latent
            or expected.audio_samples_per_channel != end_audio_sample
        ):
            raise SchemaValidationError(
                f"H3 block {index} cumulative boundary diverges from native geometry"
            )
        blocks.append(block)
    return tuple(blocks)


def build_steady_block_plan(block_count: int) -> tuple[H3StreamingBlockGeometry, ...]:
    if isinstance(block_count, bool) or not isinstance(block_count, int):
        raise SchemaValidationError("H3 block count must be an integer")
    if block_count <= 0:
        raise SchemaValidationError("H3 block count must be positive")
    candidates = structural_steady_group_candidates()
    if candidates != (2,):
        raise SchemaValidationError(
            f"pinned H3 temporal geometry has unexpected candidates: {candidates!r}"
        )
    return build_group_block_plan([candidates[0]] * block_count)


__all__ = [
    "AUDIO_LATENT_RATE_HZ",
    "AUDIO_SAMPLE_RATE_HZ",
    "H3NativeClipGeometry",
    "H3StreamingBlockGeometry",
    "MAXIMUM_STEADY_BLOCK_SECONDS",
    "MINIMUM_STEADY_BLOCK_SECONDS",
    "NATIVE_H3_REFERENCE_REVISION",
    "VIDEO_FPS",
    "VIDEO_GROUP_FRAMES",
    "VIDEO_GROUP_LATENT_UNITS",
    "VIDEO_PREFIX_FRAMES",
    "VIDEO_PREFIX_LATENT_UNITS",
    "align_frame_count",
    "audio_latent_boundary_for_frames",
    "audio_sample_boundary_for_frames",
    "build_group_block_plan",
    "build_steady_block_plan",
    "native_clip_geometry",
    "round_fraction_half_even",
    "structural_steady_group_candidates",
    "video_latent_units_for_aligned_frames",
]
