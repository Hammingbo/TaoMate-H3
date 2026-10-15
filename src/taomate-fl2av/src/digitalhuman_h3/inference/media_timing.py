"""Media timing transforms shared by H3 long-video publication paths."""

from __future__ import annotations


def h3_delivery_frame_counts(*, seconds: int, fps: int = 24) -> tuple[int, int]:
    """Return native and published frame counts for one H3 request."""

    published = int(seconds) * int(fps)
    return published + 4, published


def endpoint_preserving_video_filter(
    *,
    native_frames: int,
    published_frames: int,
    fps: int = 24,
) -> str:
    """Compress the complete native request onto the published frame timeline."""

    if native_frames <= 1 or published_frames <= 1:
        raise ValueError("video timing requires at least two native and published frames")
    return (
        f"trim=end_frame={native_frames},"
        f"setpts=(PTS-STARTPTS)*{published_frames - 1}/{native_frames - 1},"
        f"fps={fps}:round=near"
    )


def exact_audio_delivery_filter(
    *,
    native_samples: int,
    published_samples: int,
    sample_rate: int = 32_000,
) -> str:
    """Retimestamp the complete native waveform to an exact delivery sample count."""

    if native_samples <= 0 or published_samples <= 0:
        raise ValueError("audio timing requires positive sample counts")
    return (
        f"atempo={native_samples}/{published_samples},"
        f"aresample={sample_rate},"
        f"apad=whole_len={published_samples},"
        f"atrim=end_sample={published_samples},"
        "asetpts=N/SR/TB"
    )


def s2v_audio_conditioning_filter(
    *,
    source_samples: int,
    native_samples: int,
    sample_rate: int = 32_000,
) -> str:
    """Expand one exact source window onto the native H3 audio timeline."""

    source_exact = (
        f"aresample={sample_rate},"
        f"apad=whole_len={source_samples},"
        f"atrim=end_sample={source_samples},"
        "asetpts=N/SR/TB"
    )
    return source_exact + "," + exact_audio_delivery_filter(
        native_samples=source_samples,
        published_samples=native_samples,
        sample_rate=sample_rate,
    )


def s2v_published_frame_count(
    *,
    source_duration_seconds: float,
    fps: int = 24,
) -> int:
    """Return the nearest whole-frame duration for the original GT audio."""

    if source_duration_seconds <= 0 or fps <= 0:
        raise ValueError("S2V publication requires positive duration and fps")
    return max(1, int(round(source_duration_seconds * fps)))


__all__ = [
    "endpoint_preserving_video_filter",
    "exact_audio_delivery_filter",
    "h3_delivery_frame_counts",
    "s2v_audio_conditioning_filter",
    "s2v_published_frame_count",
]
