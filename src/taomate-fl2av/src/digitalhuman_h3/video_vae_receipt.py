from __future__ import annotations

import os
from typing import Any, Sequence


LOCAL_TEMPORAL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_temporal_output.v1"
)
PARALLEL_TEMPORAL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_parallel_temporal_output.v1"
)
BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_temporal_output_balanced.v1"
)
BALANCED_PARALLEL_TEMPORAL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_parallel_temporal_output_balanced.v1"
)
TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_temporal_window_dp_output.v1"
)
TEMPORAL_WINDOW_DP_PARALLEL_OUTPUT_SCHEMA = (
    "digitalhuman_h3.inference.video_vae_temporal_window_dp_parallel_output.v1"
)


def balanced_temporal_frame_shard(
    output_frames: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    if output_frames < 1:
        raise ValueError("Video VAE output frame count must be positive")
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("Video VAE balanced frame shard rank is invalid")
    base, remainder = divmod(output_frames, world_size)
    count = base + int(rank < remainder)
    start = rank * base + min(rank, remainder)
    return start, count


def balanced_temporal_window_shard(
    window_count: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Assign one contiguous temporal-window range to each decode rank."""

    if window_count < world_size or world_size < 1:
        raise ValueError(
            "Video VAE temporal-window DP requires at least one window per rank"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError("Video VAE temporal-window DP rank is invalid")
    base, remainder = divmod(window_count, world_size)
    count = base + int(rank < remainder)
    start = rank * base + min(rank, remainder)
    return start, count


def require_single_host_temporal_window_dp(*, world_size: int) -> None:
    """Reject the local-/dev/shm owner assembly on a multi-host process group."""

    if world_size <= 1:
        return
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    if local_world_size != world_size:
        raise ValueError(
            "Video VAE temporal-window DP requires one host: "
            "LOCAL_WORLD_SIZE must equal world_size"
        )


def build_parallel_video_vae_temporal_output_receipt(
    rank_receipts: Sequence[Any],
) -> dict[str, Any]:
    receipts = [
        dict(item) if isinstance(item, dict) else item for item in rank_receipts
    ]
    world_size = len(receipts)
    if world_size < 1:
        raise RuntimeError("Video VAE temporal output receipt list is empty")

    schemas = {
        item.get("schema") if isinstance(item, dict) else None
        for item in receipts
    }
    if schemas == {BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA}:
        return _build_balanced_parallel_receipt(receipts)
    if schemas == {TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA}:
        return _build_temporal_window_dp_parallel_receipt(receipts)
    if schemas != {LOCAL_TEMPORAL_OUTPUT_SCHEMA}:
        raise RuntimeError("Video VAE rank temporal output schemas differ")

    output_frames: int | None = None
    logical_frames: int | None = None
    dropped_frames: int | None = None
    for rank, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal output receipt is absent"
            )
        expected = {
            "schema": LOCAL_TEMPORAL_OUTPUT_SCHEMA,
            "video_parallel_rank": rank,
            "video_parallel_world_size": world_size,
            "output_retained_rank0_only": True,
            "full_output_retained": rank == 0,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"Video VAE rank {rank} temporal output receipt differs at {key}"
                )
        for key in ("logical_frames", "dropped_frames", "output_frames"):
            value = receipt.get(key)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"Video VAE rank {rank} temporal output receipt has invalid {key}"
                )
        if receipt["output_frames"] <= 0:
            raise RuntimeError("Video VAE temporal output frame count is not positive")
        if (
            receipt["logical_frames"] - receipt["dropped_frames"]
            != receipt["output_frames"]
        ):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal output accounting differs"
            )
        expected_retained = receipt["output_frames"] if rank == 0 else 1
        if receipt.get("retained_output_frames") != expected_retained:
            raise RuntimeError(
                f"Video VAE rank {rank} retained output frame count differs"
            )
        current_counts = (
            receipt["output_frames"],
            receipt["logical_frames"],
            receipt["dropped_frames"],
        )
        if output_frames is None:
            output_frames, logical_frames, dropped_frames = current_counts
        elif current_counts != (output_frames, logical_frames, dropped_frames):
            raise RuntimeError(
                "Video VAE temporal output accounting differs across ranks"
            )

    assert output_frames is not None
    assert logical_frames is not None
    assert dropped_frames is not None
    return {
        "schema": PARALLEL_TEMPORAL_OUTPUT_SCHEMA,
        "video_parallel_world_size": world_size,
        "output_retained_rank0_only": True,
        "full_output_rank": 0,
        "output_frames": output_frames,
        "logical_frames": logical_frames,
        "dropped_frames": dropped_frames,
        "rank_receipts": receipts,
    }


def _build_balanced_parallel_receipt(
    receipts: list[Any],
) -> dict[str, Any]:
    world_size = len(receipts)
    output_frames: int | None = None
    logical_frames: int | None = None
    dropped_frames: int | None = None
    next_start = 0
    for rank, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"Video VAE rank {rank} balanced temporal output receipt is absent"
            )
        expected = {
            "schema": BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA,
            "video_parallel_rank": rank,
            "video_parallel_world_size": world_size,
            "output_retained_rank0_only": False,
            "full_output_retained": world_size == 1,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"Video VAE rank {rank} balanced receipt differs at {key}"
                )
        for key in (
            "logical_frames",
            "dropped_frames",
            "output_frames",
            "retained_frame_start",
            "retained_output_frames",
        ):
            value = receipt.get(key)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"Video VAE rank {rank} balanced receipt has invalid {key}"
                )
        if receipt["output_frames"] <= 0:
            raise RuntimeError("Video VAE temporal output frame count is not positive")
        if (
            receipt["logical_frames"] - receipt["dropped_frames"]
            != receipt["output_frames"]
        ):
            raise RuntimeError(
                f"Video VAE rank {rank} balanced temporal accounting differs"
            )
        expected_start, expected_count = balanced_temporal_frame_shard(
            receipt["output_frames"],
            rank=rank,
            world_size=world_size,
        )
        if (
            receipt["retained_frame_start"],
            receipt["retained_output_frames"],
        ) != (expected_start, expected_count):
            raise RuntimeError(
                f"Video VAE rank {rank} balanced frame shard differs"
            )
        if expected_start != next_start:
            raise RuntimeError("Video VAE balanced frame shards are not contiguous")
        next_start += expected_count
        current_counts = (
            receipt["output_frames"],
            receipt["logical_frames"],
            receipt["dropped_frames"],
        )
        if output_frames is None:
            output_frames, logical_frames, dropped_frames = current_counts
        elif current_counts != (output_frames, logical_frames, dropped_frames):
            raise RuntimeError(
                "Video VAE balanced temporal accounting differs across ranks"
            )

    assert output_frames is not None
    assert logical_frames is not None
    assert dropped_frames is not None
    if next_start != output_frames:
        raise RuntimeError("Video VAE balanced frame shards do not cover output")
    return {
        "schema": BALANCED_PARALLEL_TEMPORAL_OUTPUT_SCHEMA,
        "video_parallel_world_size": world_size,
        "output_retained_rank0_only": False,
        "full_output_rank": 0 if world_size == 1 else None,
        "output_frames": output_frames,
        "logical_frames": logical_frames,
        "dropped_frames": dropped_frames,
        "rank_receipts": receipts,
    }


def _build_temporal_window_dp_parallel_receipt(
    receipts: list[Any],
) -> dict[str, Any]:
    world_size = len(receipts)
    output_frames: int | None = None
    logical_frames: int | None = None
    dropped_frames: int | None = None
    temporal_window_count: int | None = None
    next_frame = 0
    next_window = 0
    decoded_windows = 0
    decoded_microbatches = 0
    decode_calls_saved = 0
    boundary_halos = 0
    configured_microbatch_size: int | None = None
    maximum_microbatch_size = 0
    peak_packed_input_bytes = 0
    for rank, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window DP receipt is absent"
            )
        expected = {
            "schema": TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA,
            "video_parallel_rank": rank,
            "video_parallel_world_size": world_size,
            "output_retained_rank0_only": False,
            "full_output_retained": world_size == 1,
            "spatial_tile_parallel_world_size": 1,
            "per_window_collective_count": 0,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"Video VAE rank {rank} temporal-window DP receipt differs at {key}"
                )
        current_counts = (
            receipt.get("output_frames"),
            receipt.get("logical_frames"),
            receipt.get("dropped_frames"),
            receipt.get("temporal_window_count"),
        )
        if any(type(value) is not int or value < 0 for value in current_counts):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window DP accounting is invalid"
            )
        if current_counts[0] <= 0 or current_counts[3] < world_size:
            raise RuntimeError("Video VAE temporal-window DP geometry is invalid")
        if current_counts[1] - current_counts[2] != current_counts[0]:
            raise RuntimeError(
                "Video VAE temporal-window DP frame accounting differs"
            )
        if output_frames is None:
            (
                output_frames,
                logical_frames,
                dropped_frames,
                temporal_window_count,
            ) = current_counts
        elif current_counts != (
            output_frames,
            logical_frames,
            dropped_frames,
            temporal_window_count,
        ):
            raise RuntimeError(
                "Video VAE temporal-window DP accounting differs across ranks"
            )

        window_start = receipt.get("owned_temporal_window_start")
        window_count = receipt.get("owned_temporal_window_count")
        frame_start = receipt.get("retained_frame_start")
        frame_count = receipt.get("retained_output_frames")
        halo_count = receipt.get("boundary_halo_window_count")
        decoded_count = receipt.get("decoded_temporal_window_count")
        microbatch_size = receipt.get("temporal_window_microbatch_size")
        microbatch_count = receipt.get("decoded_temporal_microbatch_count")
        maximum_rank_microbatch_size = receipt.get(
            "maximum_decoded_temporal_microbatch_size"
        )
        rank_decode_calls_saved = receipt.get(
            "temporal_window_decode_calls_saved"
        )
        rank_peak_packed_input_bytes = receipt.get("peak_packed_input_bytes")
        for value in (
            window_start,
            window_count,
            frame_start,
            frame_count,
            halo_count,
            decoded_count,
            microbatch_size,
            microbatch_count,
            maximum_rank_microbatch_size,
            rank_decode_calls_saved,
            rank_peak_packed_input_bytes,
        ):
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"Video VAE rank {rank} temporal-window DP shard is invalid"
                )
        expected_window_start, expected_window_count = (
            balanced_temporal_window_shard(
                int(temporal_window_count), rank=rank, world_size=world_size
            )
        )
        if (window_start, window_count) != (
            expected_window_start,
            expected_window_count,
        ):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window DP ownership differs"
            )
        if window_start != next_window or frame_start != next_frame:
            raise RuntimeError(
                "Video VAE temporal-window DP shards are not contiguous"
            )
        if decoded_count != window_count + halo_count:
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window DP decode count differs"
            )
        if microbatch_size < 1:
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window microbatch is invalid"
            )
        if configured_microbatch_size is None:
            configured_microbatch_size = microbatch_size
        elif microbatch_size != configured_microbatch_size:
            raise RuntimeError(
                "Video VAE temporal-window microbatch size differs across ranks"
            )
        if (
            microbatch_count < 1
            or microbatch_count
            < (decoded_count + microbatch_size - 1) // microbatch_size
            or microbatch_count > decoded_count
            or maximum_rank_microbatch_size < 1
            or maximum_rank_microbatch_size > microbatch_size
            or maximum_rank_microbatch_size > decoded_count
            or rank_decode_calls_saved != decoded_count - microbatch_count
            or rank_peak_packed_input_bytes < 1
        ):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window microbatch accounting differs"
            )
        if receipt.get(
            "estimated_decoder_activation_peak_multiplier_vs_batch1"
        ) != float(maximum_rank_microbatch_size):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window memory estimate differs"
            )
        if receipt.get("memory_estimate_scope") != (
            "exact_packed_input_bytes_and_linear_activation_multiplier"
        ):
            raise RuntimeError(
                f"Video VAE rank {rank} temporal-window memory scope differs"
            )
        next_window += window_count
        next_frame += frame_count
        decoded_windows += decoded_count
        decoded_microbatches += microbatch_count
        decode_calls_saved += rank_decode_calls_saved
        maximum_microbatch_size = max(
            maximum_microbatch_size,
            maximum_rank_microbatch_size,
        )
        peak_packed_input_bytes = max(
            peak_packed_input_bytes,
            rank_peak_packed_input_bytes,
        )
        boundary_halos += halo_count

    assert output_frames is not None
    assert logical_frames is not None
    assert dropped_frames is not None
    assert temporal_window_count is not None
    assert configured_microbatch_size is not None
    if next_window != temporal_window_count or next_frame != output_frames:
        raise RuntimeError(
            "Video VAE temporal-window DP shards do not cover the output"
        )
    return {
        "schema": TEMPORAL_WINDOW_DP_PARALLEL_OUTPUT_SCHEMA,
        "video_parallel_world_size": world_size,
        "output_retained_rank0_only": False,
        "full_output_rank": 0 if world_size == 1 else None,
        "output_frames": output_frames,
        "logical_frames": logical_frames,
        "dropped_frames": dropped_frames,
        "temporal_window_count": temporal_window_count,
        "decoded_temporal_window_count": decoded_windows,
        "temporal_window_microbatch_size": configured_microbatch_size,
        "decoded_temporal_microbatch_count": decoded_microbatches,
        "maximum_decoded_temporal_microbatch_size": maximum_microbatch_size,
        "temporal_window_decode_calls_saved": decode_calls_saved,
        "peak_packed_input_bytes": peak_packed_input_bytes,
        "estimated_decoder_activation_peak_multiplier_vs_batch1": float(
            maximum_microbatch_size
        ),
        "memory_estimate_scope": (
            "exact_packed_input_bytes_and_linear_activation_multiplier"
        ),
        "boundary_halo_window_count": boundary_halos,
        "spatial_tile_parallel_world_size": 1,
        "per_window_collective_count": 0,
        "rank_receipts": receipts,
    }


def validated_parallel_video_vae_temporal_output_receipt(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError("parallel Video VAE temporal output receipt is absent")
    rank_receipts = value.get("rank_receipts")
    if not isinstance(rank_receipts, list):
        raise RuntimeError("parallel Video VAE rank receipts are absent")
    expected = build_parallel_video_vae_temporal_output_receipt(rank_receipts)
    if value != expected:
        raise RuntimeError("parallel Video VAE temporal output receipt differs")
    return expected


__all__ = [
    "BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA",
    "BALANCED_PARALLEL_TEMPORAL_OUTPUT_SCHEMA",
    "balanced_temporal_frame_shard",
    "balanced_temporal_window_shard",
    "build_parallel_video_vae_temporal_output_receipt",
    "LOCAL_TEMPORAL_OUTPUT_SCHEMA",
    "PARALLEL_TEMPORAL_OUTPUT_SCHEMA",
    "TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA",
    "TEMPORAL_WINDOW_DP_PARALLEL_OUTPUT_SCHEMA",
    "require_single_host_temporal_window_dp",
    "validated_parallel_video_vae_temporal_output_receipt",
]
