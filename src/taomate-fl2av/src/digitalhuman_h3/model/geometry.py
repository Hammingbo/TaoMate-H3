# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 spatial and temporal request geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..denoise_schedule import select_time_shift_sigmas, time_shift_sigmas


FPS = 24
CANVAS_MULTIPLE = 32
DEFAULT_SHORT_EDGE = 768
DEFAULT_MAX_PIXELS = 768 * 1344


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    width: int
    height: int
    fps: int
    frame_count: int
    duration_seconds: float
    video_latent_t: int
    video_latent_h: int
    video_latent_w: int
    audio_latent_t: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    return int(frame_count) + (5 - int(frame_count)) % 17


def video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def frame_count_from_video_latent_t(value: int) -> int:
    if value == 1:
        return 1
    if value < 2 or (value - 2) % 5:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(value) - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    return int(round(float(duration_seconds) * 40.0))


def _parse_aspect_ratio(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", 1)
        width, height = int(left), int(right)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("aspect_ratio must use positive integer W:H") from exc
    if width <= 0 or height <= 0:
        raise ValueError("aspect_ratio must use positive integer W:H")
    ratio = width / height
    if not 0.25 <= ratio <= 4.0:
        raise ValueError("MiniMax H3 aspect ratio must be in [1:4, 4:1]")
    return width, height


def _aligned(value: float) -> int:
    return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE)) * CANVAS_MULTIPLE)


def resolve_reference_image_size(width: int, height: int) -> tuple[int, int]:
    """Official Ref2VA image canvas: 2048 short edge, nearest 32, no area cap.

    Reference images keep their own aspect ratio independently of the output.
    Source: SGLang ad5af539, MiniMax H3 reference_encoding.py.
    """
    if min(width, height) <= 0 or not 0.25 <= width / height <= 4.0:
        raise ValueError("reference image dimensions must be positive with ratio in [1:4, 4:1]")
    scale = 2048 / min(width, height)
    return _aligned(width * scale), _aligned(height * scale)


def resolve_geometry(
    *,
    duration_seconds: float,
    aspect_ratio: str = "16:9",
    width: int | None = None,
    height: int | None = None,
    short_edge: int = DEFAULT_SHORT_EDGE,
) -> MiniMaxH3Geometry:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    if (width is None) != (height is None):
        raise ValueError("width and height must be supplied together")
    if width is not None:
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        width, height = int(width), int(height)
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError("explicit width and height must be positive multiples of 32")
    else:
        ar_w, ar_h = _parse_aspect_ratio(aspect_ratio)
        ratio = ar_w / ar_h
        if ratio >= 1.0:
            resolved_w, resolved_h = short_edge * ratio, float(short_edge)
        else:
            resolved_w, resolved_h = float(short_edge), short_edge / ratio
        area = resolved_w * resolved_h
        if area > DEFAULT_MAX_PIXELS:
            scale = math.sqrt(DEFAULT_MAX_PIXELS / area)
            resolved_w *= scale
            resolved_h *= scale
        width, height = _aligned(resolved_w), _aligned(resolved_h)
    frames = align_frame_count(int(round(duration_seconds * FPS)))
    aligned_duration = frames / FPS
    return MiniMaxH3Geometry(
        width=width,
        height=height,
        fps=FPS,
        frame_count=frames,
        duration_seconds=aligned_duration,
        video_latent_t=video_latent_t(frames),
        video_latent_h=height // 16,
        video_latent_w=width // 16,
        audio_latent_t=audio_latent_t(aligned_duration),
    )


__all__ = [
    "CANVAS_MULTIPLE",
    "DEFAULT_SHORT_EDGE",
    "FPS",
    "MiniMaxH3Geometry",
    "align_frame_count",
    "audio_latent_t",
    "frame_count_from_video_latent_t",
    "resolve_geometry",
    "resolve_reference_image_size",
    "select_time_shift_sigmas",
    "time_shift_sigmas",
    "video_latent_t",
]
