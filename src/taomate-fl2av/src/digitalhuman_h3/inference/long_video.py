# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import json

import os

import shutil

import stat

import subprocess

import tempfile

from dataclasses import dataclass

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Callable, Mapping, Protocol, Sequence

from ..streaming.geometry import StreamPlan, canonical_continuation_plan, plan_duration


from .media_timing import (
    endpoint_preserving_video_filter,
    exact_audio_delivery_filter,
    h3_delivery_frame_counts,
)


FORMAL_ROLLOUT_SCHEMA_VERSION = "digitalhuman_h3.formal.rollout_boundary.v2"

FORMAL_REQUEST_COUNT = 12

FORMAL_REQUEST_SECONDS = 5

FORMAL_REQUEST_FRAMES = 124

FORMAL_OUTPUT_WIDTH = 768

FORMAL_OUTPUT_HEIGHT = 1_376

AUDIO_DECODE_MODES = ("auto", "one_shot", "legacy_per_segment")

VIDEO_DECODE_MODES = (
    "auto",
    "one_shot",
    "one_shot_affine_prefix_dedup",
    "legacy_per_segment",
)

VIDEO_AFFINE_PREFIX_LATENTS = 2

VIDEO_NOISE_REQUEST_STRIDE = 1_000_003

def _video_codec_native_frames(temporal_latents: int) -> int:
    if temporal_latents < VIDEO_AFFINE_PREFIX_LATENTS or (
        temporal_latents - VIDEO_AFFINE_PREFIX_LATENTS
    ) % 5:
        raise ValueError("video latents do not form one canonical codec stream")
    return 5 + 17 * (
        (temporal_latents - VIDEO_AFFINE_PREFIX_LATENTS) // 5
    )

def _final_mux_output_args(final_path: Path, *, expected_seconds: float) -> tuple[str, ...]:
    # A frame-count stop can end muxing before the final partial AAC packet.
    # Fractional-second requests already have an exact normalized video length;
    # use time trimming and a sample-rate movie timescale to retain their audio.
    timing = (
        ("-frames:v", str(expected_seconds * 24), "-t", str(expected_seconds))
        if float(expected_seconds).is_integer()
        else ("-t", str(expected_seconds), "-movie_timescale", "32000")
    )
    return (
        *timing,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-profile:a",
        "aac_low",
        "-b:a",
        "192k",
        "-ar",
        "32000",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        "-n",
        str(final_path),
    )

def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"required media tool is absent: {name}")
    return path

def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)

def _rank_zero_action(
    runtime: H3Generator,
    label: str,
    action: Callable[[], None],
) -> None:
    error: str | None = None
    if runtime.is_main_process:
        try:
            action()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = runtime.broadcast(error)
    if error:
        raise RuntimeError(f"{label} failed on rank zero: {error}")

def _rank_zero_value(
    runtime: H3Generator,
    label: str,
    action: Callable[[], Any],
) -> Any:
    payload: dict[str, Any] | None = None
    if runtime.is_main_process:
        try:
            payload = {"value": action(), "error": None}
        except Exception as exc:
            payload = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
    payload = runtime.broadcast(payload)
    if not isinstance(payload, dict) or payload.get("error"):
        detail = payload.get("error") if isinstance(payload, dict) else "invalid payload"
        raise RuntimeError(f"{label} failed on rank zero: {detail}")
    return payload.get("value")

def _audio_sample_count(audio: Mapping[str, Any], *, label: str) -> int:
    try:
        sample_rate = int(audio["sample_rate"])
        duration_ts = int(audio["duration_ts"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has no exact audio sample count") from exc
    if sample_rate != 32_000 or audio.get("time_base") != "1/32000":
        raise RuntimeError(f"{label} has an unexpected audio time base")
    return duration_ts

def _normalize_segment(
    source: Path,
    destination: Path,
    *,
    seconds: int,
    source_probe: Mapping[str, Any],
    include_audio: bool = True,
    native_frames: int | None = None,
    published_frames: int | None = None,
) -> None:
    if native_frames is None or published_frames is None:
        if native_frames is not None or published_frames is not None:
            raise ValueError(
                "native_frames and published_frames must be supplied together"
            )
        native_frames, published_frames = h3_delivery_frame_counts(seconds=seconds)
    command = [
        _binary("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if include_audio:
        _, source_audio = _av_streams(source_probe, label="raw segment")
        command.extend(
            (
                "-map",
                "0:a:0",
                "-af",
                exact_audio_delivery_filter(
                    native_samples=_audio_sample_count(
                        source_audio, label="raw segment"
                    ),
                    published_samples=seconds * 32_000,
                ),
            )
        )
    command.extend(
        (
            "-vf",
            endpoint_preserving_video_filter(
                native_frames=native_frames,
                published_frames=published_frames,
            ),
            "-frames:v",
            str(published_frames),
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
        )
    )
    if include_audio:
        command.extend(
            (
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "32000",
                "-ac",
                "2",
            )
        )
    else:
        command.append("-an")
    command.extend(
        (
            "-movflags",
            "+faststart",
            "-n",
            str(destination),
        )
    )
    _run(command)

def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            _binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            (
                "format=duration:stream=codec_type,codec_name,pix_fmt,width,height,"
                "r_frame_rate,nb_frames,duration,time_base,duration_ts,"
                "sample_rate,channels"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("ffprobe returned an invalid object")
    return value

def _video_stream(probe: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"{label} has no streams")
    video = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"),
        None,
    )
    if not isinstance(video, dict):
        raise RuntimeError(f"{label} must contain video")
    return video

def _require_no_audio_stream(probe: Mapping[str, Any], *, label: str) -> None:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"{label} has no streams")
    if any(
        isinstance(item, dict) and item.get("codec_type") == "audio"
        for item in streams
    ):
        raise RuntimeError(f"{label} unexpectedly contains intermediate audio")

def _av_streams(probe: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"{label} has no streams")
    video = _video_stream(probe, label=label)
    audio = next(
        (item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"),
        None,
    )
    if not isinstance(audio, dict):
        raise RuntimeError(f"{label} must contain video and audio")
    return video, audio

def _require_h3_av_streams(
    video: Mapping[str, Any],
    audio: Mapping[str, Any] | None,
    *,
    label: str,
    frames: int,
    short_edge: int | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> None:
    try:
        actual_frames = int(video["nb_frames"])
        width = int(video["width"])
        height = int(video["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has invalid video metadata") from exc
    if (expected_width is None) != (expected_height is None):
        raise ValueError("expected_width and expected_height must be supplied together")
    if short_edge is not None and expected_width is not None:
        raise ValueError("short-edge and exact-resolution contracts are mutually exclusive")
    geometry_matches = (
        (width, height) == (expected_width, expected_height)
        if expected_width is not None and expected_height is not None
        else short_edge is not None and min(width, height) == short_edge
    )
    if (
        actual_frames != frames
        or video.get("r_frame_rate") != "24/1"
        or video.get("codec_name") != "h264"
        or video.get("pix_fmt") != "yuv420p"
        or not geometry_matches
    ):
        raise RuntimeError(f"{label} differs from the pinned H3 video contract")
    if audio is None:
        return
    try:
        channels = int(audio["channels"])
        sample_rate = int(audio["sample_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} has invalid AV metadata") from exc
    if (
        audio.get("codec_name") != "aac"
        or sample_rate != 32_000
        or channels != 2
    ):
        raise RuntimeError(f"{label} differs from the pinned H3 AV contract")

def _strict_decode(path: Path, *, require_audio: bool = True) -> None:
    command = [
        _binary("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:v:0",
    ]
    if require_audio:
        command.extend(("-map", "0:a:0"))
    command.extend(("-f", "null", "-"))
    _run(command)

def _validated_segment_streams(
    probe: Mapping[str, Any],
    *,
    label: str,
    frames: int,
    expected_width: int | None,
    expected_height: int | None,
    require_audio: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if require_audio:
        video, audio = _av_streams(probe, label=label)
    else:
        video, audio = _video_stream(probe, label=label), None
        _require_no_audio_stream(probe, label=label)
    _require_h3_av_streams(
        video,
        audio,
        label=label,
        frames=frames,
        short_edge=768 if expected_width is None else None,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    return video, audio

def _validate_segment(
    path: Path,
    *,
    seconds: int,
    expected_width: int | None = None,
    expected_height: int | None = None,
    require_audio: bool = True,
) -> dict[str, Any]:
    probe = _probe(path)
    _, audio = _validated_segment_streams(
        probe,
        label="normalized segment",
        frames=seconds * 24,
        expected_width=expected_width,
        expected_height=expected_height,
        require_audio=require_audio,
    )
    if abs(float(probe.get("format", {}).get("duration", -1)) - seconds) > 0.01:
        raise RuntimeError(f"normalized segment does not match {seconds}s contract")
    if require_audio and audio is not None and _audio_sample_count(
        audio, label="normalized segment"
    ) != seconds * 32_000:
        raise RuntimeError("normalized segment audio sample count differs")
    if require_audio:
        _strict_decode(path)
    else:
        _strict_decode(path, require_audio=False)
    return probe
