"""Decode and save native MiniMax H3 audio-video outputs."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
import time
import types
import weakref
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .components import (
    configure_bundled_video_vae_tile_parallel,
    load_bundled_audio_vae,
    load_bundled_video_vae,
    read_component_config,
)
from .packed_tokens import minimax_h3_patchify_video_latent
from ..video_vae_receipt import (
    BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA,
    LOCAL_TEMPORAL_OUTPUT_SCHEMA,
    TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA,
    balanced_temporal_frame_shard,
    balanced_temporal_window_shard,
)


def _reverse_normalize(
    latent: torch.Tensor,
    *,
    mean_values: list[float],
    std_values: list[float],
) -> torch.Tensor:
    mean = torch.as_tensor(mean_values, dtype=latent.dtype, device=latent.device)
    std = torch.as_tensor(std_values, dtype=latent.dtype, device=latent.device)
    if mean.ndim != 1 or mean.shape != std.shape or int(latent.shape[1]) != int(mean.numel()):
        raise ValueError("H3 VAE latent normalization geometry differs")
    shape = [1] * latent.ndim
    shape[1] = int(mean.numel())
    return latent * std.view(shape) + mean.view(shape)


def _canonical_video(
    frames: torch.Tensor,
    *,
    batch_size: int,
    preserve_dtype: bool = False,
) -> torch.Tensor:
    if frames.ndim == 4:
        frames = frames.reshape(batch_size, -1, *frames.shape[1:]).transpose(1, 2)
    if frames.ndim != 5 or int(frames.shape[0]) != batch_size:
        raise ValueError(f"unexpected decoded H3 video shape {tuple(frames.shape)}")
    if preserve_dtype:
        return frames.contiguous()
    return frames.float().contiguous()


class _IncrementalTemporalDecode:
    """Exact bundled temporal decode driven by an ordered latent iterator."""

    def __init__(
        self,
        *,
        active_vae: Any,
        expected_tokens: int,
        video_parallel_rank: int,
        video_parallel_world_size: int,
        temporal_cat_dtype: torch.dtype | None,
    ) -> None:
        if expected_tokens < 1:
            raise ValueError("incremental Video VAE requires source latents")
        self.active_vae = active_vae
        self.expected_tokens = int(expected_tokens)
        self.video_parallel_rank = int(video_parallel_rank)
        self.video_parallel_world_size = int(video_parallel_world_size)
        self.temporal_cat_dtype = temporal_cat_dtype
        self.head_tokens = int(
            active_vae.isolated_first_frame and active_vae.frame_pre_padding == 0
        )
        self.tail_tokens = int(active_vae.isolated_last_frame)
        self.body_tokens = self.expected_tokens - self.head_tokens - self.tail_tokens
        if self.body_tokens < 1:
            raise ValueError("incremental Video VAE source is shorter than head/tail")
        pseudo_total_tokens = self.body_tokens + int(active_vae.token_drop)
        remainder = pseudo_total_tokens % int(active_vae.tokens_chunk_size)
        self.pad_tokens = (
            0
            if remainder == 0
            else int(active_vae.tokens_chunk_size) - remainder
        )
        pseudo_total_tokens += self.pad_tokens
        self.num_chunks = pseudo_total_tokens // int(active_vae.tokens_chunk_size)
        self.num_chunks -= int(active_vae.token_drop > 0)
        if self.num_chunks < 1:
            raise ValueError("incremental Video VAE planned no temporal chunks")
        self.source: torch.Tensor | None = None
        self.received_tokens = 0
        self.next_chunk = 0
        self.decoded_overlap: torch.Tensor | None = None
        self.decoded: torch.Tensor | None = None
        self.return_frame: torch.Tensor | None = None
        self.write_pos = 0
        self.logical_frames = 0
        self.dropped_frames = 0
        self.output_frames: int | None = None
        self.total_frames: int | None = None
        self.pad_frames: int | None = None
        self.decoded_chunks = 0
        self.cuda_events: list[tuple[Any, Any]] = []
        self.active_wall_seconds = 0.0

    def append(self, normalized_latents: torch.Tensor) -> None:
        if normalized_latents.ndim != 5 or int(normalized_latents.shape[2]) < 1:
            raise ValueError("incremental Video VAE append has invalid shape")
        if self.source is None:
            shape = list(normalized_latents.shape)
            shape[2] = self.expected_tokens
            self.source = torch.empty(
                shape,
                dtype=normalized_latents.dtype,
                device=normalized_latents.device,
            )
        elif (
            tuple(normalized_latents.shape[:2])
            + tuple(normalized_latents.shape[3:])
            != tuple(self.source.shape[:2]) + tuple(self.source.shape[3:])
            or normalized_latents.dtype != self.source.dtype
            or normalized_latents.device != self.source.device
        ):
            raise ValueError("incremental Video VAE append geometry differs")
        stop = self.received_tokens + int(normalized_latents.shape[2])
        if stop > self.expected_tokens:
            raise ValueError("incremental Video VAE received too many latents")
        self.source[:, :, self.received_tokens:stop].copy_(normalized_latents)
        self.received_tokens = stop
        self._decode_ready(final=False)

    def _padded_body(self) -> torch.Tensor:
        assert self.source is not None
        body = self.source[
            :,
            :,
            self.head_tokens : self.head_tokens + self.body_tokens,
            ...,
        ]
        if self.pad_tokens:
            body = torch.cat(
                [
                    body,
                    body[:, :, -1:, ...].repeat(
                        1, 1, self.pad_tokens, 1, 1
                    ),
                ],
                dim=2,
            )
        return body

    def _ensure_output_plan(self) -> None:
        if self.output_frames is not None:
            return
        assert self.source is not None
        body = self._padded_body()
        z_head = self.source[:, :, :1, ...] if self.head_tokens else None
        z_tail = self.source[:, :, -1:, ...] if self.tail_tokens else None
        total, pad, output = self.active_vae._decode_temporal_output_frame_plan(
            body,
            z_head,
            z_tail,
            self.num_chunks,
            self.pad_tokens,
        )
        if output <= 0:
            raise ValueError("incremental Video VAE planned no output frames")
        self.total_frames = int(total)
        self.pad_frames = int(pad)
        self.output_frames = int(output)

    def _write_part(self, part: torch.Tensor) -> None:
        frames = int(part.shape[2])
        if frames <= 0:
            return
        self.logical_frames += frames
        assert self.output_frames is not None
        retain_full = self.video_parallel_rank == 0
        if retain_full and self.decoded is None:
            shape = list(part.shape)
            shape[2] = self.output_frames
            self.decoded = torch.empty(
                shape,
                dtype=part.dtype,
                device=torch.device("cpu"),
            )
        copy_frames = min(frames, max(0, self.output_frames - self.write_pos))
        if copy_frames > 0:
            host_part = part[:, :, :copy_frames].detach().to(device="cpu")
            if retain_full:
                assert self.decoded is not None
                self.decoded[
                    :, :, self.write_pos : self.write_pos + copy_frames, ...
                ].copy_(host_part)
            elif self.return_frame is None:
                self.return_frame = host_part[:, :, :1, ...].contiguous()
            self.write_pos += copy_frames
        self.dropped_frames += frames - copy_frames

    def _decode_chunk(self, chunk_index: int) -> None:
        chunk_started = time.perf_counter()
        assert self.source is not None
        self._ensure_output_plan()
        active_vae = self.active_vae
        body = (
            self._padded_body()
            if chunk_index == self.num_chunks - 1
            else self.source[
                :,
                :,
                self.head_tokens : self.head_tokens + self.body_tokens,
                ...,
            ]
        )
        token_start = chunk_index * int(active_vae.tokens_chunk_size)
        token_end = (
            token_start
            + int(active_vae.tokens_chunk_size)
            + int(active_vae.token_overlap)
        )
        clip_z = body[:, :, token_start:token_end, ...]
        if chunk_index == 0 and self.head_tokens:
            clip_z = torch.cat([self.source[:, :, :1, ...], clip_z], dim=2)
        if chunk_index == self.num_chunks - 1 and self.tail_tokens:
            clip_z = torch.cat([clip_z, self.source[:, :, -1:, ...]], dim=2)
        cuda_start = None
        cuda_stop = None
        if clip_z.device.type == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_stop = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        clip_decoded = active_vae._adaptive_decode(clip_z)
        self.decoded_chunks += 1
        if (
            self.temporal_cat_dtype is not None
            and clip_decoded.dtype != self.temporal_cat_dtype
        ):
            clip_decoded = clip_decoded.to(self.temporal_cat_dtype)
        if clip_decoded.device != self.source.device:
            clip_decoded = clip_decoded.to(self.source.device)

        decoded_tail = None
        if chunk_index == 0 and self.head_tokens:
            self._write_part(
                clip_decoded[
                    :,
                    :,
                    int(active_vae.vae_ratio_t) - 1 : int(active_vae.vae_ratio_t),
                    ...,
                ]
            )
            clip_decoded = clip_decoded[:, :, int(active_vae.vae_ratio_t) :, ...]
        if chunk_index == self.num_chunks - 1 and self.tail_tokens:
            decoded_tail = clip_decoded[:, :, -1:, ...]
            clip_decoded = clip_decoded[
                :, :, : -int(active_vae.vae_ratio_t), ...
            ]

        chunk_dec = int(active_vae.tokens_chunk_size) * int(active_vae.vae_ratio_t)
        split_count = int(active_vae.token_drop > 0) + 1
        for split_index in range(split_count):
            frame_start = split_index * chunk_dec
            frame_stop = min(frame_start + chunk_dec, int(clip_decoded.shape[2]))
            part = clip_decoded[:, :, frame_start:frame_stop, ...]
            part = part[:, :, int(active_vae.frame_pre_padding) :, ...]
            if split_index == 0:
                if self.decoded_overlap is not None:
                    part = active_vae.blend(
                        self.decoded_overlap,
                        part,
                        int(active_vae.frame_overlap),
                        dim=-3,
                    )
                    self.decoded_overlap = None
                self._write_part(part)
            else:
                self.decoded_overlap = part.contiguous()
        if chunk_index == self.num_chunks - 1:
            if self.decoded_overlap is not None:
                self._write_part(self.decoded_overlap)
                self.decoded_overlap = None
            if decoded_tail is not None:
                self._write_part(decoded_tail)
        if cuda_stop is not None:
            cuda_stop.record()
            self.cuda_events.append((cuda_start, cuda_stop))
        self.active_wall_seconds += time.perf_counter() - chunk_started

    def _decode_ready(self, *, final: bool) -> None:
        available_body = min(
            max(self.received_tokens - self.head_tokens, 0), self.body_tokens
        )
        while self.next_chunk < self.num_chunks:
            is_last = self.next_chunk == self.num_chunks - 1
            token_stop = (
                self.next_chunk * int(self.active_vae.tokens_chunk_size)
                + int(self.active_vae.tokens_chunk_size)
                + int(self.active_vae.token_overlap)
            )
            if is_last:
                if not final:
                    break
            elif token_stop > available_body:
                break
            self._decode_chunk(self.next_chunk)
            self.next_chunk += 1

    def finish(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.source is None or self.received_tokens != self.expected_tokens:
            raise RuntimeError("incremental Video VAE source timeline is incomplete")
        self._decode_ready(final=True)
        if self.next_chunk != self.num_chunks:
            raise RuntimeError("incremental Video VAE did not decode every chunk")
        assert self.total_frames is not None
        assert self.pad_frames is not None
        assert self.output_frames is not None
        if (
            self.logical_frames != self.total_frames
            or self.dropped_frames != self.pad_frames
            or self.write_pos != self.output_frames
        ):
            raise RuntimeError(
                "incremental Video VAE frame plan mismatch: "
                f"logical_frames={self.logical_frames} total_frames={self.total_frames} "
                f"dropped_frames={self.dropped_frames} pad_frames={self.pad_frames} "
                f"write_pos={self.write_pos} output_frames={self.output_frames}"
            )
        output = self.decoded if self.video_parallel_rank == 0 else self.return_frame
        if output is None:
            raise RuntimeError("incremental Video VAE produced no retained output")
        active_cuda_seconds = None
        if self.cuda_events:
            assert self.source is not None
            torch.cuda.synchronize(self.source.device)
            active_cuda_seconds = sum(
                float(start.elapsed_time(stop))
                for start, stop in self.cuda_events
            ) / 1000.0
        return output, {
            "schema": LOCAL_TEMPORAL_OUTPUT_SCHEMA,
            "video_parallel_rank": self.video_parallel_rank,
            "video_parallel_world_size": self.video_parallel_world_size,
            "output_retained_rank0_only": True,
            "full_output_retained": self.video_parallel_rank == 0,
            "logical_frames": self.logical_frames,
            "dropped_frames": self.dropped_frames,
            "output_frames": self.output_frames,
            "retained_output_frames": (
                self.output_frames if self.video_parallel_rank == 0 else 1
            ),
            "async_d2h": False,
            "online_temporal_decode": True,
            "decoded_temporal_chunks": self.decoded_chunks,
            "active_wall_seconds": self.active_wall_seconds,
            "active_cuda_seconds": active_cuda_seconds,
        }


class _TemporalWindowDPDecode:
    """Decode disjoint temporal windows on replicated VAE ranks.

    Each rank owns one contiguous range of bundled FL2VA temporal windows.
    A rank after rank zero recomputes exactly one left-halo window so the
    bundled five-frame overlap blend remains local and follows the original
    ordered arithmetic.  Spatial tiling is rank-local in this mode, therefore no
    tile result collective occurs inside ``_adaptive_decode``.  Optional
    owner-local microbatching only concatenates shape-compatible windows on
    the sample axis; window geometry and ordered assembly stay unchanged.
    """

    def __init__(
        self,
        *,
        active_vae: Any,
        source: torch.Tensor,
        video_parallel_rank: int,
        video_parallel_world_size: int,
        temporal_cat_dtype: torch.dtype | None,
        temporal_window_microbatch_size: int = 1,
        pixel_postprocess: Any | None = None,
        gpu_rgb24_output: bool = False,
        fused_gpu_rgb24_output: bool = False,
        pixel_denormalize_mean: tuple[float, float, float] | None = None,
        pixel_denormalize_std: tuple[float, float, float] | None = None,
    ) -> None:
        self.active_vae = active_vae
        self.source = source
        self.video_parallel_rank = int(video_parallel_rank)
        self.defer_output_sync = False
        self.video_parallel_world_size = int(video_parallel_world_size)
        self.temporal_cat_dtype = temporal_cat_dtype
        if temporal_window_microbatch_size < 1:
            raise ValueError(
                "temporal-window DP microbatch size must be positive"
            )
        self.temporal_window_microbatch_size = int(
            temporal_window_microbatch_size
        )
        if gpu_rgb24_output and pixel_postprocess is None:
            raise ValueError(
                "temporal-window DP GPU RGB24 output requires pixel postprocess"
            )
        self.pixel_postprocess = pixel_postprocess
        self.gpu_rgb24_output = bool(gpu_rgb24_output)
        self.fused_gpu_rgb24_output = bool(fused_gpu_rgb24_output)
        self.pixel_denormalize_mean = pixel_denormalize_mean
        self.pixel_denormalize_std = pixel_denormalize_std
        self.rgb24_output_backend = "disabled"
        self.head_tokens = int(
            active_vae.isolated_first_frame and active_vae.frame_pre_padding == 0
        )
        self.tail_tokens = int(active_vae.isolated_last_frame)
        body_tokens = int(source.shape[2]) - self.head_tokens - self.tail_tokens
        if body_tokens < 1:
            raise ValueError("temporal-window DP source is shorter than head/tail")
        self.body = source[
            :, :, self.head_tokens : self.head_tokens + body_tokens, ...
        ]
        pseudo_total_tokens = body_tokens + int(active_vae.token_drop)
        remainder = pseudo_total_tokens % int(active_vae.tokens_chunk_size)
        self.pad_tokens = (
            0
            if remainder == 0
            else int(active_vae.tokens_chunk_size) - remainder
        )
        if self.pad_tokens:
            self.body = torch.cat(
                [
                    self.body,
                    self.body[:, :, -1:, ...].repeat(
                        1, 1, self.pad_tokens, 1, 1
                    ),
                ],
                dim=2,
            )
        pseudo_total_tokens += self.pad_tokens
        self.num_chunks = (
            pseudo_total_tokens // int(active_vae.tokens_chunk_size)
            - int(active_vae.token_drop > 0)
        )
        self.chunk_start, self.chunk_count = balanced_temporal_window_shard(
            self.num_chunks,
            rank=self.video_parallel_rank,
            world_size=self.video_parallel_world_size,
        )
        self.chunk_stop = self.chunk_start + self.chunk_count
        self.z_head = source[:, :, :1, ...] if self.head_tokens else None
        self.z_tail = source[:, :, -1:, ...] if self.tail_tokens else None
        (
            self.total_frames,
            self.pad_frames,
            self.output_frames,
        ) = active_vae._decode_temporal_output_frame_plan(
            self.body,
            self.z_head,
            self.z_tail,
            self.num_chunks,
            self.pad_tokens,
        )
        self.primary_frame_counts, final_overlap_frames = (
            self._window_frame_counts()
        )
        planned_total = (
            self.head_tokens
            + sum(self.primary_frame_counts)
            + final_overlap_frames
            + self.tail_tokens
        )
        if planned_total != self.total_frames:
            raise RuntimeError(
                "temporal-window DP frame plan differs from bundled decoder"
            )
        self.retained_frame_start = self.head_tokens + sum(
            self.primary_frame_counts[: self.chunk_start]
        )
        retained_frames = sum(
            self.primary_frame_counts[self.chunk_start : self.chunk_stop]
        )
        if self.chunk_start == 0:
            retained_frames += self.head_tokens
        if self.chunk_stop == self.num_chunks:
            retained_frames += final_overlap_frames + self.tail_tokens
        self.retained_output_frames = min(
            retained_frames,
            max(0, self.output_frames - self.retained_frame_start),
        )
        if self.retained_output_frames < 1:
            raise RuntimeError("temporal-window DP rank owns no output frames")
        self.boundary_halo_window_count = int(
            self.chunk_start > 0 and int(active_vae.token_drop > 0)
        )
        self.decoded_temporal_window_count = 0
        self.decoded_temporal_microbatch_count = 0
        self.maximum_decoded_temporal_microbatch_size = 0
        self.peak_packed_input_bytes = 0
        self.active_wall_seconds = 0.0
        self.cuda_events: list[tuple[Any, Any]] = []
        self.copy_stream = (
            torch.cuda.Stream(device=source.device)
            if source.device.type == "cuda"
            else None
        )
        self.decoded_cpu: torch.Tensor | None = None
        self.write_pos = 0

    def _window_frame_counts(self) -> tuple[list[int], int]:
        active_vae = self.active_vae
        chunk_dec = int(active_vae.tokens_chunk_size) * int(
            active_vae.vae_ratio_t
        )
        split_count = int(active_vae.token_drop > 0) + 1
        primary_counts: list[int] = []
        final_overlap_frames = 0
        for chunk_index in range(self.num_chunks):
            token_start = chunk_index * int(active_vae.tokens_chunk_size)
            token_stop = token_start + int(active_vae.tokens_chunk_size) + int(
                active_vae.token_overlap
            )
            clip_tokens = max(
                0,
                min(token_stop, int(self.body.shape[2]))
                - min(token_start, int(self.body.shape[2])),
            )
            if chunk_index == 0 and self.z_head is not None:
                clip_tokens += int(self.z_head.shape[2])
            if chunk_index == self.num_chunks - 1 and self.z_tail is not None:
                clip_tokens += int(self.z_tail.shape[2])
            clip_frames = clip_tokens * int(active_vae.vae_ratio_t)
            if chunk_index == 0 and self.z_head is not None:
                clip_frames -= int(active_vae.vae_ratio_t)
            if chunk_index == self.num_chunks - 1 and self.z_tail is not None:
                clip_frames -= int(active_vae.vae_ratio_t)
            counts = []
            for split_index in range(split_count):
                frame_start = split_index * chunk_dec
                frame_stop = min(frame_start + chunk_dec, clip_frames)
                counts.append(
                    max(
                        0,
                        frame_stop
                        - frame_start
                        - int(active_vae.frame_pre_padding),
                    )
                )
            primary_counts.append(counts[0])
            if split_count > 1:
                final_overlap_frames = counts[1]
        return primary_counts, final_overlap_frames

    def _window_input(self, chunk_index: int) -> torch.Tensor:
        active_vae = self.active_vae
        token_start = chunk_index * int(active_vae.tokens_chunk_size)
        token_stop = token_start + int(active_vae.tokens_chunk_size) + int(
            active_vae.token_overlap
        )
        clip_z = self.body[:, :, token_start:token_stop, ...]
        if chunk_index == 0 and self.z_head is not None:
            clip_z = torch.cat([self.z_head, clip_z], dim=2)
        if chunk_index == self.num_chunks - 1 and self.z_tail is not None:
            clip_z = torch.cat([clip_z, self.z_tail], dim=2)
        return clip_z

    @staticmethod
    def _compatible_window_inputs(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> bool:
        return (
            tuple(left.shape[1:]) == tuple(right.shape[1:])
            and left.dtype == right.dtype
            and left.device == right.device
        )

    def _postprocess_window(
        self,
        chunk_index: int,
        clip_decoded: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        active_vae = self.active_vae
        decoded_head = None
        if chunk_index == 0 and self.z_head is not None:
            ratio = int(active_vae.vae_ratio_t)
            decoded_head = clip_decoded[:, :, ratio - 1 : ratio, ...]
            clip_decoded = clip_decoded[:, :, ratio:, ...]
        decoded_tail = None
        if chunk_index == self.num_chunks - 1 and self.z_tail is not None:
            decoded_tail = clip_decoded[:, :, -1:, ...]
            clip_decoded = clip_decoded[
                :, :, : -int(active_vae.vae_ratio_t), ...
            ]

        chunk_dec = int(active_vae.tokens_chunk_size) * int(
            active_vae.vae_ratio_t
        )
        parts: list[torch.Tensor] = []
        for split_index in range(int(active_vae.token_drop > 0) + 1):
            frame_start = split_index * chunk_dec
            frame_stop = min(frame_start + chunk_dec, int(clip_decoded.shape[2]))
            parts.append(
                clip_decoded[
                    :,
                    :,
                    frame_start:frame_stop,
                    ...,
                ][:, :, int(active_vae.frame_pre_padding) :, ...]
            )
        primary = parts[0]
        overlap = parts[1].contiguous() if len(parts) > 1 else None
        return decoded_head, primary, overlap, decoded_tail

    def _decode_windows(
        self,
        chunk_indices: Sequence[int],
        clip_inputs: Sequence[torch.Tensor],
    ) -> list[
        tuple[
            torch.Tensor | None,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ]:
        if not chunk_indices or len(chunk_indices) != len(clip_inputs):
            raise ValueError("temporal-window DP microbatch is empty or misaligned")
        if len(chunk_indices) > self.temporal_window_microbatch_size:
            raise ValueError("temporal-window DP microbatch exceeds configured size")
        first = clip_inputs[0]
        if any(
            not self._compatible_window_inputs(first, candidate)
            for candidate in clip_inputs[1:]
        ):
            raise ValueError("temporal-window DP microbatch input shapes differ")

        started = time.perf_counter()
        packed = (
            first
            if len(clip_inputs) == 1
            else torch.cat(list(clip_inputs), dim=0)
        )
        self.peak_packed_input_bytes = max(
            self.peak_packed_input_bytes,
            int(packed.numel()) * int(packed.element_size()),
        )
        cuda_start = None
        cuda_stop = None
        if packed.device.type == "cuda":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_stop = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        decoded = self.active_vae._adaptive_decode(packed)
        self.decoded_temporal_window_count += len(chunk_indices)
        self.decoded_temporal_microbatch_count += 1
        self.maximum_decoded_temporal_microbatch_size = max(
            self.maximum_decoded_temporal_microbatch_size,
            len(chunk_indices),
        )
        if (
            self.temporal_cat_dtype is not None
            and decoded.dtype != self.temporal_cat_dtype
        ):
            decoded = decoded.to(self.temporal_cat_dtype)
        if decoded.device != self.source.device:
            decoded = decoded.to(self.source.device)
        source_batch_size = int(self.source.shape[0])
        expected_batch_size = source_batch_size * len(chunk_indices)
        if int(decoded.shape[0]) != expected_batch_size:
            raise RuntimeError(
                "temporal-window DP decoder changed the packed batch size"
            )
        decoded_windows = list(decoded.split(source_batch_size, dim=0))
        processed = [
            self._postprocess_window(chunk_index, decoded_window)
            for chunk_index, decoded_window in zip(
                chunk_indices,
                decoded_windows,
                strict=True,
            )
        ]
        if cuda_stop is not None:
            cuda_stop.record()
            self.cuda_events.append((cuda_start, cuda_stop))
        self.active_wall_seconds += time.perf_counter() - started
        return processed

    def _write_part(self, part: torch.Tensor) -> None:
        part_frames = int(part.shape[2])
        copy_frames = min(
            part_frames,
            max(0, self.retained_output_frames - self.write_pos),
        )
        if copy_frames <= 0:
            return
        source = part[:, :, :copy_frames, ...].detach()
        if self.gpu_rgb24_output:
            converted = None
            if (
                self.fused_gpu_rgb24_output
                and self.pixel_denormalize_mean is not None
                and self.pixel_denormalize_std is not None
            ):
                from digitalhuman_h3.inference.fused_kernels import (
                    try_h3_video_vae_rgb24_exact,
                )

                converted = try_h3_video_vae_rgb24_exact(
                    source,
                    mean_values=self.pixel_denormalize_mean,
                    std_values=self.pixel_denormalize_std,
                )
            if converted is None:
                source = (
                    self.pixel_postprocess(source)
                    .float()
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                )
                self.rgb24_output_backend = "torch_eager"
            else:
                source = converted
                self.rgb24_output_backend = "triton_exact"
        if self.decoded_cpu is None:
            shape = list(part.shape)
            shape[2] = self.retained_output_frames
            self.decoded_cpu = torch.empty(
                shape,
                dtype=source.dtype,
                device=torch.device("cpu"),
                pin_memory=self.copy_stream is not None,
            )
        destination = self.decoded_cpu[
            :, :, self.write_pos : self.write_pos + copy_frames, ...
        ]
        if self.copy_stream is None:
            destination.copy_(source.to(device="cpu"))
        else:
            compute_stream = torch.cuda.current_stream(device=source.device)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(compute_stream)
                destination.copy_(source, non_blocking=True)
                source.record_stream(self.copy_stream)
        self.write_pos += copy_frames

    def decode(self) -> tuple[torch.Tensor, dict[str, Any]]:
        active_vae = self.active_vae
        previous_overlap = None
        scheduled_indices = list(
            range(
                self.chunk_start - self.boundary_halo_window_count,
                self.chunk_stop,
            )
        )
        schedule_position = 0
        while schedule_position < len(scheduled_indices):
            batch_indices = [scheduled_indices[schedule_position]]
            batch_inputs = [self._window_input(batch_indices[0])]
            while (
                len(batch_indices) < self.temporal_window_microbatch_size
                and schedule_position + len(batch_indices)
                < len(scheduled_indices)
            ):
                candidate_index = scheduled_indices[
                    schedule_position + len(batch_indices)
                ]
                candidate = self._window_input(candidate_index)
                if not self._compatible_window_inputs(batch_inputs[0], candidate):
                    break
                batch_indices.append(candidate_index)
                batch_inputs.append(candidate)

            decoded_batch = self._decode_windows(batch_indices, batch_inputs)
            for chunk_index, decoded_window in zip(
                batch_indices,
                decoded_batch,
                strict=True,
            ):
                head, primary, overlap, tail = decoded_window
                if chunk_index < self.chunk_start:
                    previous_overlap = overlap
                    if previous_overlap is None:
                        raise RuntimeError(
                            "temporal-window DP boundary halo produced no overlap"
                        )
                    continue
                if head is not None:
                    self._write_part(head)
                if previous_overlap is not None:
                    primary = active_vae.blend(
                        previous_overlap,
                        primary,
                        int(active_vae.frame_overlap),
                        dim=-3,
                    )
                self._write_part(primary)
                previous_overlap = overlap
                if chunk_index == self.num_chunks - 1:
                    if previous_overlap is not None:
                        self._write_part(previous_overlap)
                        previous_overlap = None
                    if tail is not None:
                        self._write_part(tail)
            schedule_position += len(batch_indices)

        if self.copy_stream is not None and not self.defer_output_sync:
            self.copy_stream.synchronize()
        if self.decoded_cpu is None or self.write_pos != self.retained_output_frames:
            raise RuntimeError(
                "temporal-window DP retained frame count differs from plan"
            )
        active_cuda_seconds = None
        if self.cuda_events and not self.defer_output_sync:
            torch.cuda.synchronize(self.source.device)
            active_cuda_seconds = sum(
                float(start.elapsed_time(stop))
                for start, stop in self.cuda_events
            ) / 1000.0
        return self.decoded_cpu, {
            "schema": TEMPORAL_WINDOW_DP_LOCAL_OUTPUT_SCHEMA,
            "video_parallel_rank": self.video_parallel_rank,
            "video_parallel_world_size": self.video_parallel_world_size,
            "output_retained_rank0_only": False,
            "full_output_retained": self.video_parallel_world_size == 1,
            "logical_frames": int(self.total_frames),
            "dropped_frames": int(self.pad_frames),
            "output_frames": int(self.output_frames),
            "retained_frame_start": self.retained_frame_start,
            "retained_output_frames": self.retained_output_frames,
            "temporal_window_count": self.num_chunks,
            "owned_temporal_window_start": self.chunk_start,
            "owned_temporal_window_count": self.chunk_count,
            "boundary_halo_window_count": self.boundary_halo_window_count,
            "decoded_temporal_window_count": self.decoded_temporal_window_count,
            "temporal_window_microbatch_size": (
                self.temporal_window_microbatch_size
            ),
            "decoded_temporal_microbatch_count": (
                self.decoded_temporal_microbatch_count
            ),
            "maximum_decoded_temporal_microbatch_size": (
                self.maximum_decoded_temporal_microbatch_size
            ),
            "temporal_window_decode_calls_saved": (
                self.decoded_temporal_window_count
                - self.decoded_temporal_microbatch_count
            ),
            "peak_packed_input_bytes": self.peak_packed_input_bytes,
            "estimated_decoder_activation_peak_multiplier_vs_batch1": float(
                self.maximum_decoded_temporal_microbatch_size
            ),
            "memory_estimate_scope": (
                "exact_packed_input_bytes_and_linear_activation_multiplier"
            ),
            "spatial_tile_parallel_world_size": 1,
            "per_window_collective_count": 0,
            "async_d2h": self.copy_stream is not None,
            "owner_assembly": "contiguous_window_owner_to_cpu_rgb24_shard",
            "gpu_rgb24_output": self.gpu_rgb24_output,
            "gpu_rgb24_output_backend": self.rgb24_output_backend,
            "cpu_output_dtype": str(self.decoded_cpu.dtype),
            "active_wall_seconds": self.active_wall_seconds,
            "active_cuda_seconds": active_cuda_seconds,
        }


class H3LatentDecoder:
    """Preloaded H3 AV-VAE decoder used by recurring training evaluation."""

    def __init__(
        self,
        *,
        video_vae_root: Path,
        audio_vae_root: Path,
        video_parallel_group: Any | None = None,
        video_parallel_rank: int = 0,
        video_parallel_world_size: int = 1,
        load_audio: bool = True,
    ) -> None:
        self.video_parallel_group = video_parallel_group
        self.video_parallel_rank = video_parallel_rank
        self.video_parallel_world_size = video_parallel_world_size
        self.video_config = read_component_config(video_vae_root)
        self.audio_config = read_component_config(audio_vae_root) if load_audio else None
        self.video_vae = load_bundled_video_vae(
            video_vae_root,
            tile_parallel_group=video_parallel_group,
            tile_parallel_rank=video_parallel_rank,
            tile_parallel_world_size=video_parallel_world_size,
        ).cpu().eval()
        self.audio_vae = (
            load_bundled_audio_vae(audio_vae_root).cpu().eval()
            if load_audio
            else None
        )
        self._video_vae_fp8_linears_enabled = False
        self._video_vae_fp8_attention_enabled = False
        self._video_vae_exact_fusions_enabled = False
        self._video_vae_mlp_spatial_stride = 1
        self._video_vae_attention_kv_spatial_stride = 1
        self._video_vae_temporal_token_overlap: int | None = None
        self._video_vae_cpu_temporal_output_enabled = False
        self._video_vae_temporal_window_dp_enabled = False
        self._video_vae_temporal_window_microbatch_size = 1
        self._video_vae_gpu_rgb24_output_enabled = False
        self._video_vae_original_decode_temporal_streaming: Any | None = None
        self.last_video_decode_receipt: dict[str, Any] | None = None
        self.video_vae_fp8_receipt: dict[str, Any] | None = None
        self.video_vae_fp8_attention_receipt: dict[str, Any] | None = None
        self.video_vae_exact_fusions_receipt: dict[str, Any] | None = None
        self.video_vae_sparse_token_receipt: dict[str, Any] | None = None

    def enable_video_vae_fp8_linears(self) -> None:
        self._video_vae_fp8_linears_enabled = True

    def enable_video_vae_fp8_attention(self) -> None:
        self._video_vae_fp8_attention_enabled = True

    def enable_video_vae_exact_fusions(self) -> None:
        self._video_vae_exact_fusions_enabled = True

    def enable_video_vae_temporal_window_dp(
        self,
        *,
        microbatch_size: int = 1,
        gpu_rgb24_output: bool = False,
    ) -> None:
        """Assign complete FL2VA temporal windows to decoder ranks."""

        if self.video_parallel_world_size < 2:
            raise ValueError(
                "Video VAE temporal-window DP requires multiple decoder ranks"
            )
        if microbatch_size < 1:
            raise ValueError(
                "Video VAE temporal-window microbatch size must be positive"
            )
        self._video_vae_temporal_window_dp_enabled = True
        self._video_vae_temporal_window_microbatch_size = int(microbatch_size)
        self._video_vae_gpu_rgb24_output_enabled = bool(gpu_rgb24_output)

    def configure_video_vae_temporal_overlap(self, token_overlap: int) -> None:
        if token_overlap < 0:
            raise ValueError("video VAE temporal token overlap cannot be negative")
        video_vae = self.video_vae.model
        video_vae.token_overlap = int(token_overlap)
        video_vae.frame_overlap = max(
            video_vae.token_overlap * video_vae.vae_ratio_t
            - video_vae.frame_pre_padding,
            0,
        )
        video_vae.processor.token_overlap = video_vae.token_overlap
        video_vae.processor.frame_overlap = video_vae.frame_overlap
        if token_overlap == 0 and not hasattr(
            video_vae, "_h3_original_decode_temporal_pad_frames"
        ):
            video_vae._h3_original_decode_temporal_pad_frames = (
                video_vae._decode_temporal_pad_frames
            )

            def decode_temporal_pad_frames(
                active_vae: Any,
                latent: torch.Tensor,
                pad_tokens: int,
            ) -> int:
                if int(active_vae.token_overlap) == 0:
                    return 0
                return active_vae._h3_original_decode_temporal_pad_frames(
                    latent, pad_tokens
                )

            video_vae._decode_temporal_pad_frames = types.MethodType(
                decode_temporal_pad_frames,
                video_vae,
            )
        self._video_vae_temporal_token_overlap = int(token_overlap)

    def enable_video_vae_cpu_temporal_output(
        self,
        *,
        balanced_across_ranks: bool = False,
    ) -> None:
        """Keep the assembled decoded video on CPU while VAE tiles stay on CUDA."""

        video_vae = self.video_vae.model
        if self._video_vae_cpu_temporal_output_enabled:
            return
        self._video_vae_original_decode_temporal_streaming = (
            video_vae._decode_temporal_streaming
        )
        decoder_ref = weakref.ref(self)
        video_parallel_rank = self.video_parallel_rank
        video_parallel_world_size = self.video_parallel_world_size
        retain_full_output = video_parallel_rank == 0 and not balanced_across_ranks

        def decode_temporal_streaming(
            active_vae: Any,
            z: torch.Tensor,
            z_head: torch.Tensor | None,
            z_tail: torch.Tensor | None,
            num_chunks: int,
            pad_tokens: int,
            temporal_cat_dtype: torch.dtype | None,
        ) -> torch.Tensor:
            total_frames, pad_frames, output_frames = (
                active_vae._decode_temporal_output_frame_plan(
                    z, z_head, z_tail, num_chunks, pad_tokens
                )
            )
            if output_frames <= 0:
                raise ValueError(
                    "decode_temporal streaming planned non-positive "
                    f"output_frames={output_frames} total_frames={total_frames} "
                    f"pad_frames={pad_frames}"
                )

            if balanced_across_ranks:
                retained_frame_start, retained_output_frames = (
                    balanced_temporal_frame_shard(
                        output_frames,
                        rank=video_parallel_rank,
                        world_size=video_parallel_world_size,
                    )
                )
            else:
                retained_frame_start = 0
                retained_output_frames = output_frames if retain_full_output else 1

            chunk_dec = active_vae.tokens_chunk_size * active_vae.vae_ratio_t
            split_count = int(active_vae.token_drop > 0) + 1
            decoded = None
            return_frame = None
            decoded_overlap = None
            copy_stream = (
                torch.cuda.Stream(device=z.device)
                if balanced_across_ranks and z.device.type == "cuda"
                else None
            )
            write_pos = 0
            logical_frames = 0
            dropped_frames = 0

            def copy_part_to_host(
                destination: torch.Tensor,
                source: torch.Tensor,
            ) -> None:
                if copy_stream is None:
                    destination.copy_(source.detach().to(device="cpu"))
                    return
                compute_stream = torch.cuda.current_stream(device=source.device)
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_stream(compute_stream)
                    destination.copy_(source.detach(), non_blocking=True)
                    source.record_stream(copy_stream)

            def write_part(part: torch.Tensor) -> None:
                nonlocal decoded, return_frame, write_pos
                nonlocal logical_frames, dropped_frames
                part_frames = int(part.shape[2])
                if part_frames <= 0:
                    return
                logical_frames += part_frames
                if (retain_full_output or balanced_across_ranks) and decoded is None:
                    output_shape = list(part.shape)
                    output_shape[2] = retained_output_frames
                    decoded = torch.empty(
                        output_shape,
                        dtype=part.dtype,
                        device=torch.device("cpu"),
                        pin_memory=copy_stream is not None,
                    )
                remaining = output_frames - write_pos
                copy_frames = min(part_frames, max(0, remaining))
                if copy_frames > 0:
                    if balanced_across_ranks:
                        global_start = write_pos
                        global_stop = write_pos + copy_frames
                        retained_stop = retained_frame_start + retained_output_frames
                        copy_start = max(global_start, retained_frame_start)
                        copy_stop = min(global_stop, retained_stop)
                        if copy_stop > copy_start:
                            part_start = copy_start - global_start
                            part_stop = copy_stop - global_start
                            destination_start = copy_start - retained_frame_start
                            destination_stop = copy_stop - retained_frame_start
                            assert decoded is not None
                            copy_part_to_host(
                                decoded[
                                    :,
                                    :,
                                    destination_start:destination_stop,
                                    :,
                                    :,
                                ],
                                part[:, :, part_start:part_stop],
                            )
                    elif retain_full_output:
                        host_part = part[:, :, :copy_frames].detach().to(
                            device=torch.device("cpu")
                        )
                        assert decoded is not None
                        decoded[
                            :, :, write_pos : write_pos + copy_frames, :, :
                        ].copy_(host_part)
                        del host_part
                    elif return_frame is None:
                        return_frame = (
                            part[:, :, :1]
                            .detach()
                            .to(device=torch.device("cpu"))
                            .contiguous()
                        )
                    write_pos += copy_frames
                dropped_frames += part_frames - copy_frames

            for chunk_index in range(num_chunks):
                token_start = chunk_index * active_vae.tokens_chunk_size
                token_end = (
                    token_start
                    + active_vae.tokens_chunk_size
                    + active_vae.token_overlap
                )
                clip_z = z[:, :, token_start:token_end, :, :]
                if chunk_index == 0 and z_head is not None:
                    clip_z = torch.cat([z_head, clip_z], dim=2)
                if chunk_index == num_chunks - 1 and z_tail is not None:
                    clip_z = torch.cat([clip_z, z_tail], dim=2)

                clip_decoded = active_vae._adaptive_decode(clip_z)
                if (
                    temporal_cat_dtype is not None
                    and clip_decoded.dtype != temporal_cat_dtype
                ):
                    clip_decoded = clip_decoded.to(temporal_cat_dtype)
                if clip_decoded.device != z.device:
                    clip_decoded = clip_decoded.to(z.device)

                decoded_tail = None
                if chunk_index == 0 and z_head is not None:
                    write_part(
                        clip_decoded[
                            :,
                            :,
                            active_vae.vae_ratio_t - 1 : active_vae.vae_ratio_t,
                            :,
                            :,
                        ]
                    )
                    clip_decoded = clip_decoded[
                        :, :, active_vae.vae_ratio_t :, :, :
                    ]
                if chunk_index == num_chunks - 1 and z_tail is not None:
                    decoded_tail = clip_decoded[:, :, -1:, :, :]
                    clip_decoded = clip_decoded[
                        :, :, : -active_vae.vae_ratio_t, :, :
                    ]

                for split_index in range(split_count):
                    frame_start = split_index * chunk_dec
                    frame_end = min(
                        frame_start + chunk_dec, int(clip_decoded.shape[2])
                    )
                    clip_chunk = clip_decoded[
                        :, :, frame_start:frame_end, :, :
                    ]
                    clip_chunk = clip_chunk[
                        :, :, active_vae.frame_pre_padding :, :, :
                    ]
                    if split_index == 0:
                        if decoded_overlap is not None:
                            clip_chunk = active_vae.blend(
                                decoded_overlap,
                                clip_chunk,
                                active_vae.frame_overlap,
                                dim=-3,
                            )
                            decoded_overlap = None
                        write_part(clip_chunk)
                    else:
                        decoded_overlap = clip_chunk.contiguous()

                if chunk_index == num_chunks - 1:
                    if decoded_overlap is not None:
                        write_part(decoded_overlap)
                        decoded_overlap = None
                    if decoded_tail is not None:
                        write_part(decoded_tail)
                del clip_decoded, clip_z

            if copy_stream is not None:
                copy_stream.synchronize()

            if (retain_full_output or balanced_across_ranks) and decoded is None:
                raise RuntimeError(
                    "decode_temporal streaming produced no output tensor"
                )
            if (
                not retain_full_output
                and not balanced_across_ranks
                and return_frame is None
            ):
                raise RuntimeError(
                    "decode_temporal streaming produced no return frame"
                )
            if (
                logical_frames != total_frames
                or dropped_frames != pad_frames
                or write_pos != output_frames
            ):
                raise RuntimeError(
                    "decode_temporal streaming frame plan mismatch: "
                    f"logical_frames={logical_frames} total_frames={total_frames} "
                    f"dropped_frames={dropped_frames} pad_frames={pad_frames} "
                    f"write_pos={write_pos} output_frames={output_frames}"
                )
            decoder = decoder_ref()
            if decoder is None:
                raise RuntimeError(
                    "Video VAE temporal-output decoder was released during decode"
                )
            decoder.last_video_decode_receipt = {
                "schema": (
                    BALANCED_LOCAL_TEMPORAL_OUTPUT_SCHEMA
                    if balanced_across_ranks
                    else LOCAL_TEMPORAL_OUTPUT_SCHEMA
                ),
                "video_parallel_rank": video_parallel_rank,
                "video_parallel_world_size": video_parallel_world_size,
                "output_retained_rank0_only": not balanced_across_ranks,
                "full_output_retained": (
                    video_parallel_world_size == 1
                    if balanced_across_ranks
                    else retain_full_output
                ),
                "logical_frames": logical_frames,
                "dropped_frames": dropped_frames,
                "output_frames": output_frames,
                "retained_output_frames": retained_output_frames,
                "async_d2h": copy_stream is not None,
            }
            if balanced_across_ranks:
                decoder.last_video_decode_receipt["retained_frame_start"] = (
                    retained_frame_start
                )
            return (
                decoded
                if (retain_full_output or balanced_across_ranks)
                else return_frame
            )

        video_vae._decode_temporal_streaming = types.MethodType(
            decode_temporal_streaming,
            video_vae,
        )
        self._video_vae_cpu_temporal_output_enabled = True

    def disable_video_vae_cpu_temporal_output(self) -> None:
        """Restore the bundled VAE temporal decoder after a scoped decode."""

        if not self._video_vae_cpu_temporal_output_enabled:
            return
        original = self._video_vae_original_decode_temporal_streaming
        if original is not None:
            self.video_vae.model._decode_temporal_streaming = original
        self._video_vae_original_decode_temporal_streaming = None
        self._video_vae_cpu_temporal_output_enabled = False

    def enable_video_vae_sparse_tokens(
        self,
        *,
        mlp_spatial_stride: int = 1,
        attention_kv_spatial_stride: int = 1,
    ) -> None:
        if mlp_spatial_stride < 1 or attention_kv_spatial_stride < 1:
            raise ValueError("video VAE sparse-token strides must be positive")
        self._video_vae_mlp_spatial_stride = int(mlp_spatial_stride)
        self._video_vae_attention_kv_spatial_stride = int(
            attention_kv_spatial_stride
        )

    @staticmethod
    def _video_vae_spatial_indices(
        *,
        torch_module: Any,
        sequence: int,
        tile_rows: int,
        tile_columns: int,
        stride: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        suffix_tokens = 5
        patch_tokens = sequence - suffix_tokens
        spatial_tokens = tile_rows * tile_columns
        if patch_tokens <= 0 or patch_tokens % spatial_tokens:
            return None
        temporal = patch_tokens // spatial_tokens
        selected = torch_module.arange(
            patch_tokens,
            dtype=torch.long,
            device=device,
        ).view(temporal, tile_rows, tile_columns)[:, ::stride, ::stride]
        suffix = torch_module.arange(
            patch_tokens,
            sequence,
            dtype=torch.long,
            device=device,
        )
        return torch_module.cat((selected.reshape(-1), suffix), dim=0)

    @staticmethod
    def _enable_sparse_mlp_(
        video_vae: Any,
        *,
        spatial_stride: int,
    ) -> dict[str, Any]:
        decoder = video_vae.model.decoder
        tile_rows = int(video_vae.model.decoder_tile_size) // int(
            video_vae.model.vae_ratio
        )
        tile_columns = tile_rows
        patched = 0
        for block in decoder.transformer_blocks:
            module = block.ff
            if hasattr(module, "_h3_original_forward"):
                continue
            module._h3_original_forward = module.forward
            module._h3_sparse_spatial_stride = spatial_stride
            module._h3_sparse_tile_rows = tile_rows
            module._h3_sparse_tile_columns = tile_columns

            def forward(feed_forward: Any, hidden_states: torch.Tensor) -> torch.Tensor:
                if hidden_states.ndim != 3:
                    return feed_forward._h3_original_forward(hidden_states)
                sequence = int(hidden_states.shape[1])
                indices = H3LatentDecoder._video_vae_spatial_indices(
                    torch_module=torch,
                    sequence=sequence,
                    tile_rows=feed_forward._h3_sparse_tile_rows,
                    tile_columns=feed_forward._h3_sparse_tile_columns,
                    stride=feed_forward._h3_sparse_spatial_stride,
                    device=hidden_states.device,
                )
                if indices is None:
                    return feed_forward._h3_original_forward(hidden_states)
                selected = feed_forward._h3_original_forward(
                    hidden_states.index_select(1, indices)
                )
                suffix_tokens = 5
                patch_tokens = sequence - suffix_tokens
                temporal = patch_tokens // (
                    feed_forward._h3_sparse_tile_rows
                    * feed_forward._h3_sparse_tile_columns
                )
                sampled_rows = (
                    feed_forward._h3_sparse_tile_rows
                    + feed_forward._h3_sparse_spatial_stride
                    - 1
                ) // feed_forward._h3_sparse_spatial_stride
                sampled_columns = (
                    feed_forward._h3_sparse_tile_columns
                    + feed_forward._h3_sparse_spatial_stride
                    - 1
                ) // feed_forward._h3_sparse_spatial_stride
                sampled_patch_tokens = temporal * sampled_rows * sampled_columns
                sampled_patches = selected[:, :sampled_patch_tokens]
                batch = int(sampled_patches.shape[0])
                channels = int(sampled_patches.shape[-1])
                sampled_patches = sampled_patches.view(
                    batch,
                    temporal,
                    sampled_rows,
                    sampled_columns,
                    channels,
                ).permute(0, 1, 4, 2, 3)
                restored = torch.nn.functional.interpolate(
                    sampled_patches.reshape(
                        batch * temporal,
                        channels,
                        sampled_rows,
                        sampled_columns,
                    ),
                    size=(
                        feed_forward._h3_sparse_tile_rows,
                        feed_forward._h3_sparse_tile_columns,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )
                restored = restored.view(
                    batch,
                    temporal,
                    channels,
                    feed_forward._h3_sparse_tile_rows,
                    feed_forward._h3_sparse_tile_columns,
                ).permute(0, 1, 3, 4, 2).reshape(batch, patch_tokens, channels)
                return torch.cat((restored, selected[:, sampled_patch_tokens:]), dim=1)

            module.forward = types.MethodType(forward, module)
            patched += 1
        if patched == 0:
            raise RuntimeError("no video VAE decoder MLP modules were selected")
        return {
            "mlp_module_count": patched,
            "mlp_spatial_stride": spatial_stride,
            "tile_latent_rows": tile_rows,
            "tile_latent_columns": tile_columns,
        }

    @staticmethod
    def _enable_fp8_attention_(
        video_vae: Any,
        *,
        kv_spatial_stride: int = 1,
    ) -> dict[str, Any]:
        from flash_attn_interface import flash_attn_func

        from digitalhuman_h3.inference.fused_kernels import (
            quantize_fp8_e4m3_per_head,
        )

        patched = 0
        tile_rows = int(video_vae.model.decoder_tile_size) // int(
            video_vae.model.vae_ratio
        )
        tile_columns = tile_rows
        for module in video_vae.modules():
            if not all(
                hasattr(module, name)
                for name in ("perform_attention", "heads", "dim_head")
            ):
                continue
            if hasattr(module, "_h3_original_perform_attention"):
                continue
            module._h3_original_perform_attention = module.perform_attention

            def perform_attention(
                attention: Any,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                pack_info: dict[str, Any] | None = None,
            ) -> torch.Tensor:
                active_pack_info = {} if pack_info is None else pack_info
                if active_pack_info.get("mask_mod") is not None or active_pack_info.get(
                    "block_sparse"
                ) is not None:
                    return attention._h3_original_perform_attention(
                        query, key, value, active_pack_info
                    )
                if kv_spatial_stride > 1:
                    indices = H3LatentDecoder._video_vae_spatial_indices(
                        torch_module=torch,
                        sequence=int(key.shape[1]),
                        tile_rows=tile_rows,
                        tile_columns=tile_columns,
                        stride=kv_spatial_stride,
                        device=key.device,
                    )
                    if indices is not None:
                        key = key.index_select(1, indices)
                        value = value.index_select(1, indices)
                q_fp8, q_descale = quantize_fp8_e4m3_per_head(query)
                k_fp8, k_descale = quantize_fp8_e4m3_per_head(key)
                v_fp8, v_descale = quantize_fp8_e4m3_per_head(value)
                output = flash_attn_func(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    causal=False,
                    num_splits=1,
                )
                return output[0] if isinstance(output, tuple) else output

            module.perform_attention = types.MethodType(perform_attention, module)
            patched += 1
        if patched == 0:
            raise RuntimeError("no video VAE attention modules were selected for FP8")
        return {
            "schema": "digitalhuman_h3.inference.video_vae_fp8_attention.v1",
            "format": "float8_e4m3fn",
            "scale": "dynamic_per_batch_head",
            "backend": "flash_attention_3",
            "attention_module_count": patched,
            "attention_kv_spatial_stride": kv_spatial_stride,
        }

    def decode(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        device: torch.device,
        offload_after_decode: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        frames = self.decode_video(
            video_latents=video_latents,
            device=device,
            offload_after_decode=offload_after_decode,
        )
        waveform, sample_rate = self.decode_audio(
            audio_latents=audio_latents,
            device=device,
            offload_after_decode=offload_after_decode,
        )
        return frames, waveform, sample_rate

    def decode_video(
        self,
        *,
        video_latents: torch.Tensor,
        device: torch.device,
        offload_after_decode: bool = True,
    ) -> torch.Tensor:
        self.last_video_decode_receipt = None
        video_vae = self.video_vae.to(device).eval()
        temporal_dp_active = False
        original_run_tile_tasks = None
        try:
            if (
                self._video_vae_exact_fusions_enabled
                and self.video_vae_exact_fusions_receipt is None
            ):
                from digitalhuman_h3.inference.video_vae_exact_fusions import (
                    install_h3_video_vae_exact_fusions_,
                )

                self.video_vae_exact_fusions_receipt = (
                    install_h3_video_vae_exact_fusions_(
                        video_vae.model.decoder,
                        device=device,
                    )
                )
            if (
                self._video_vae_fp8_linears_enabled
                and self.video_vae_fp8_receipt is None
            ):
                from digitalhuman_h3.inference.fp8 import (
                    quantize_aligned_linears_dynamic_fp8_,
                )
                from digitalhuman_h3.inference.fused_kernels import (
                    enable_h3_inference_fused_kernels,
                )

                enable_h3_inference_fused_kernels()
                self.video_vae_fp8_receipt = quantize_aligned_linears_dynamic_fp8_(
                    torch, video_vae
                )
            if (
                self._video_vae_mlp_spatial_stride > 1
                and self.video_vae_sparse_token_receipt is None
            ):
                self.video_vae_sparse_token_receipt = self._enable_sparse_mlp_(
                    video_vae,
                    spatial_stride=self._video_vae_mlp_spatial_stride,
                )
            if (
                self._video_vae_fp8_attention_enabled
                and self.video_vae_fp8_attention_receipt is None
            ):
                self.video_vae_fp8_attention_receipt = self._enable_fp8_attention_(
                    video_vae,
                    kv_spatial_stride=self._video_vae_attention_kv_spatial_stride,
                )
            if self.video_vae_sparse_token_receipt is not None:
                self.video_vae_sparse_token_receipt[
                    "attention_kv_spatial_stride"
                ] = self._video_vae_attention_kv_spatial_stride
                self.video_vae_sparse_token_receipt[
                    "schema"
                ] = "digitalhuman_h3.inference.video_vae_sparse_tokens.v1"
            configure_bundled_video_vae_tile_parallel(
                video_vae,
                group=self.video_parallel_group,
                rank=self.video_parallel_rank,
                world_size=self.video_parallel_world_size,
            )
            visual = _reverse_normalize(
                video_latents.to(device=device, dtype=torch.float32),
                mean_values=self.video_config["latents_mean"],
                std_values=self.video_config["latents_std"],
            )
            prepare = getattr(video_vae, "prepare_decoder_autocast_weights", None)
            if callable(prepare):
                prepare(torch.float16)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if self._video_vae_temporal_window_dp_enabled:
                    vae_module = importlib.import_module(
                        video_vae.model.__class__.__module__
                    )
                    resolve_temporal_dtype = getattr(
                        vae_module, "_resolve_temporal_cat_dtype", None
                    )
                    temporal_cat_dtype = (
                        resolve_temporal_dtype()
                        if callable(resolve_temporal_dtype)
                        else None
                    )
                    transform_rev = getattr(
                        video_vae.processor, "transform_rev", None
                    )
                    transform_mean = getattr(transform_rev, "mean", None)
                    transform_std = getattr(transform_rev, "std", None)
                    fused_pixel_parameters = (
                        transform_mean is not None
                        and transform_std is not None
                        and len(transform_mean) == len(transform_std) == 3
                    )
                    pixel_denormalize_mean = (
                        tuple(float(item) for item in transform_mean)
                        if fused_pixel_parameters
                        else None
                    )
                    pixel_denormalize_std = (
                        tuple(float(item) for item in transform_std)
                        if fused_pixel_parameters
                        else None
                    )
                    configure_bundled_video_vae_tile_parallel(
                        video_vae,
                        group=None,
                        rank=0,
                        world_size=1,
                    )
                    temporal_dp_active = True
                    if bool(video_vae.model.stack_tiling):
                        original_run_tile_tasks = video_vae.model._run_tile_tasks
                        owner_world_size = self.video_parallel_world_size

                        def run_owner_tile_tasks(
                            active_vae: Any,
                            tiles: Sequence[torch.Tensor],
                            tile_indices: Sequence[int],
                            forward_fn: Any,
                            stack_tiling: bool,
                            cls_agg: Any | None = None,
                        ) -> list[torch.Tensor]:
                            if not stack_tiling or not tile_indices:
                                return original_run_tile_tasks(
                                    tiles,
                                    tile_indices,
                                    forward_fn,
                                    stack_tiling,
                                    cls_agg,
                                )
                            decoded_tiles: list[torch.Tensor | None] = [
                                None
                            ] * len(tile_indices)
                            virtual_rank_count = min(
                                owner_world_size, len(tile_indices)
                            )
                            for virtual_rank in range(virtual_rank_count):
                                positions = list(
                                    range(
                                        virtual_rank,
                                        len(tile_indices),
                                        owner_world_size,
                                    )
                                )
                                virtual_indices = [
                                    tile_indices[position] for position in positions
                                ]
                                virtual_outputs = original_run_tile_tasks(
                                    tiles,
                                    virtual_indices,
                                    forward_fn,
                                    True,
                                    cls_agg,
                                )
                                for position, output in zip(
                                    positions, virtual_outputs, strict=True
                                ):
                                    decoded_tiles[position] = output
                            if any(output is None for output in decoded_tiles):
                                raise RuntimeError(
                                    "temporal-window DP missed a spatial tile"
                                )
                            return [
                                output
                                for output in decoded_tiles
                                if output is not None
                            ]

                        video_vae.model._run_tile_tasks = types.MethodType(
                            run_owner_tile_tasks,
                            video_vae.model,
                        )
                    temporal = _TemporalWindowDPDecode(
                        active_vae=video_vae.model,
                        source=visual,
                        video_parallel_rank=self.video_parallel_rank,
                        video_parallel_world_size=self.video_parallel_world_size,
                        temporal_cat_dtype=temporal_cat_dtype,
                        temporal_window_microbatch_size=(
                            self._video_vae_temporal_window_microbatch_size
                        ),
                        pixel_postprocess=video_vae.processor.revert_tensor,
                        gpu_rgb24_output=(
                            self._video_vae_gpu_rgb24_output_enabled
                        ),
                        fused_gpu_rgb24_output=(
                            self._video_vae_exact_fusions_enabled
                        ),
                        pixel_denormalize_mean=pixel_denormalize_mean,
                        pixel_denormalize_std=pixel_denormalize_std,
                    )
                    frames, receipt = temporal.decode()
                    self.last_video_decode_receipt = receipt
                else:
                    frames = video_vae.decode_base(visual)
                if not self._video_vae_gpu_rgb24_output_enabled:
                    frames = video_vae.processor.revert_tensor(frames)
            frames = _canonical_video(
                frames,
                batch_size=int(video_latents.shape[0]),
                preserve_dtype=self._video_vae_gpu_rgb24_output_enabled,
            ).cpu()
            return frames
        finally:
            if temporal_dp_active:
                if original_run_tile_tasks is not None:
                    video_vae.model._run_tile_tasks = original_run_tile_tasks
                configure_bundled_video_vae_tile_parallel(
                    video_vae,
                    group=self.video_parallel_group,
                    rank=self.video_parallel_rank,
                    world_size=self.video_parallel_world_size,
                )
            if offload_after_decode:
                self.video_vae.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def decode_video_incremental(
        self,
        *,
        video_latent_segments: Iterable[torch.Tensor],
        expected_tokens: int,
        device: torch.device,
        offload_after_decode: bool = True,
    ) -> torch.Tensor:
        """Decode one global timeline while its ordered latent suffixes arrive.

        This is not per-request VAE decoding.  It preserves the bundled
        ``decode_temporal`` chunk grid, lookahead, overlap blend, isolated
        head/tail handling, and final padding trim; only the availability of
        each temporal window changes.
        """

        self.last_video_decode_receipt = None
        if self._video_vae_temporal_window_dp_enabled:
            raise ValueError(
                "Video VAE temporal-window DP requires a complete latent tensor"
            )
        video_vae = self.video_vae.to(device).eval()
        temporal: _IncrementalTemporalDecode | None = None
        try:
            if (
                self._video_vae_exact_fusions_enabled
                and self.video_vae_exact_fusions_receipt is None
            ):
                from digitalhuman_h3.inference.video_vae_exact_fusions import (
                    install_h3_video_vae_exact_fusions_,
                )

                self.video_vae_exact_fusions_receipt = (
                    install_h3_video_vae_exact_fusions_(
                        video_vae.model.decoder,
                        device=device,
                    )
                )
            if (
                self._video_vae_fp8_linears_enabled
                and self.video_vae_fp8_receipt is None
            ):
                from digitalhuman_h3.inference.fp8 import (
                    quantize_aligned_linears_dynamic_fp8_,
                )
                from digitalhuman_h3.inference.fused_kernels import (
                    enable_h3_inference_fused_kernels,
                )

                enable_h3_inference_fused_kernels()
                self.video_vae_fp8_receipt = quantize_aligned_linears_dynamic_fp8_(
                    torch, video_vae
                )
            if (
                self._video_vae_mlp_spatial_stride > 1
                and self.video_vae_sparse_token_receipt is None
            ):
                self.video_vae_sparse_token_receipt = self._enable_sparse_mlp_(
                    video_vae,
                    spatial_stride=self._video_vae_mlp_spatial_stride,
                )
            if (
                self._video_vae_fp8_attention_enabled
                and self.video_vae_fp8_attention_receipt is None
            ):
                self.video_vae_fp8_attention_receipt = self._enable_fp8_attention_(
                    video_vae,
                    kv_spatial_stride=self._video_vae_attention_kv_spatial_stride,
                )
            if self.video_vae_sparse_token_receipt is not None:
                self.video_vae_sparse_token_receipt[
                    "attention_kv_spatial_stride"
                ] = self._video_vae_attention_kv_spatial_stride
                self.video_vae_sparse_token_receipt[
                    "schema"
                ] = "digitalhuman_h3.inference.video_vae_sparse_tokens.v1"
            configure_bundled_video_vae_tile_parallel(
                video_vae,
                group=self.video_parallel_group,
                rank=self.video_parallel_rank,
                world_size=self.video_parallel_world_size,
            )
            prepare = getattr(video_vae, "prepare_decoder_autocast_weights", None)
            if callable(prepare):
                prepare(torch.float16)
            vae_module = importlib.import_module(video_vae.model.__class__.__module__)
            resolve_temporal_dtype = getattr(
                vae_module, "_resolve_temporal_cat_dtype", None
            )
            temporal_cat_dtype = (
                resolve_temporal_dtype()
                if callable(resolve_temporal_dtype)
                else None
            )
            temporal = _IncrementalTemporalDecode(
                active_vae=video_vae.model,
                expected_tokens=expected_tokens,
                video_parallel_rank=self.video_parallel_rank,
                video_parallel_world_size=self.video_parallel_world_size,
                temporal_cat_dtype=temporal_cat_dtype,
            )
            received_segments = 0
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                for latent_segment in video_latent_segments:
                    normalized = _reverse_normalize(
                        latent_segment.to(device=device, dtype=torch.float32),
                        mean_values=self.video_config["latents_mean"],
                        std_values=self.video_config["latents_std"],
                    )
                    temporal.append(normalized)
                    received_segments += 1
                    del normalized
                if received_segments < 1:
                    raise RuntimeError(
                        "incremental Video VAE received no latent segments"
                    )
                decoded, receipt = temporal.finish()
                frames = video_vae.processor.revert_tensor(decoded)
            frames = _canonical_video(frames, batch_size=int(decoded.shape[0])).cpu()
            receipt["received_latent_segments"] = received_segments
            receipt["expected_video_latents"] = int(expected_tokens)
            self.last_video_decode_receipt = receipt
            return frames
        finally:
            if temporal is not None:
                del temporal
            if offload_after_decode:
                self.video_vae.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def decode_audio(
        self,
        *,
        audio_latents: torch.Tensor,
        device: torch.device,
        offload_after_decode: bool = True,
    ) -> tuple[torch.Tensor, int]:
        if self.audio_vae is None or self.audio_config is None:
            raise RuntimeError("H3 audio VAE decoder is not loaded on this rank")
        audio_vae = self.audio_vae.to(device).eval()
        try:
            audio = _reverse_normalize(
                audio_latents.to(device=device, dtype=torch.float32),
                mean_values=self.audio_config["latents_mean"],
                std_values=self.audio_config["latents_std"],
            )
            with torch.inference_mode():
                waveform = audio_vae.decode(audio)
            if waveform.ndim != 3 or int(waveform.shape[1]) != 1:
                raise ValueError(
                    f"unexpected decoded H3 audio shape {tuple(waveform.shape)}"
                )
            waveform = waveform.permute(1, 0, 2).contiguous().float().cpu()
            return waveform, int(audio_vae.sample_rate)
        finally:
            if offload_after_decode:
                self.audio_vae.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    def visible_frame_keyframe_rows(
        self,
        *,
        video_latents: torch.Tensor,
        frame_index: int,
        device: torch.device,
        offload_after_encode: bool = True,
    ) -> torch.Tensor:
        """Decode one published frame and re-encode its clean FL2VA rows.

        This keeps long-video handoff aligned with the frame a viewer actually
        sees.  The video VAE is offloaded before the next DiT call, so periodic
        evaluation does not add a persistent GPU model.
        """

        from PIL import Image

        video_vae = self.video_vae.to(device).eval()
        try:
            configure_bundled_video_vae_tile_parallel(
                video_vae,
                group=None,
                rank=0,
                world_size=1,
            )
            visual = _reverse_normalize(
                video_latents.to(device=device, dtype=torch.float32),
                mean_values=self.video_config["latents_mean"],
                std_values=self.video_config["latents_std"],
            )
            prepare = getattr(video_vae, "prepare_decoder_autocast_weights", None)
            if callable(prepare):
                prepare(torch.float16)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                frames = video_vae.decode_base(visual)
                frames = video_vae.processor.revert_tensor(frames)
            frames = _canonical_video(
                frames, batch_size=int(video_latents.shape[0])
            )
            resolved_index = int(frame_index)
            if resolved_index < 0:
                resolved_index += int(frames.shape[2])
            if not 0 <= resolved_index < int(frames.shape[2]):
                raise ValueError(
                    f"published frame index {frame_index} is outside decoded "
                    f"frame count {int(frames.shape[2])}"
                )
            pixels = (
                frames[0, :, resolved_index]
                .permute(1, 2, 0)
                .detach()
                .clamp(0, 1)
                .mul(255.0)
                .round()
                .to(dtype=torch.uint8, device="cpu")
                .numpy()
            )
            image = Image.fromarray(pixels, mode="RGB")
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.default_generator.manual_seed(42)
                if device.type == "cuda":
                    torch.cuda.manual_seed(42)
                latent = video_vae.encode_images(
                    image, use_fp16_latent=True
                )[0]
            latent = latent.detach().to(device="cpu", dtype=torch.float32)
            if latent.ndim == 4:
                latent = latent[None]
            if latent.ndim != 5 or int(latent.shape[1]) != 24:
                raise ValueError(
                    f"unexpected H3 keyframe latent shape {tuple(latent.shape)}"
                )
            mean = torch.tensor(
                self.video_config["latents_mean"], dtype=torch.float32
            ).view(1, 24, 1, 1, 1)
            std = torch.tensor(
                self.video_config["latents_std"], dtype=torch.float32
            ).view(1, 24, 1, 1, 1)
            return minimax_h3_patchify_video_latent(
                (latent - mean) / std,
                patch_size=(1, 2, 2),
            ).float().contiguous()
        finally:
            if offload_after_encode:
                self.video_vae.cpu()
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def decode_h3_latents(
    *,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    video_vae_root: Path,
    audio_vae_root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    decoder = H3LatentDecoder(
        video_vae_root=video_vae_root,
        audio_vae_root=audio_vae_root,
    )
    return decoder.decode(
        video_latents=video_latents,
        audio_latents=audio_latents,
        device=device,
        offload_after_decode=False,
    )


def save_h3_video_mp4(
    path: Path,
    *,
    frames: torch.Tensor,
    fps: int,
    video_filter: str | None = None,
    published_frames: int | None = None,
    crf: int = 12,
    verify_decode: bool = True,
) -> Path:
    """Atomically publish a decodable video-only H3 segment."""

    path = path.resolve()
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.suffix.lower() != ".mp4":
        raise ValueError("H3 decoded output path must end in .mp4")
    if (
        frames.ndim != 5
        or int(frames.shape[0]) != 1
        or int(frames.shape[1]) != 3
    ):
        raise ValueError(f"unexpected H3 output frames shape {tuple(frames.shape)}")
    if int(frames.shape[2]) < 1 or fps <= 0:
        raise ValueError("H3 video output requires frames and a positive fps")
    if video_filter is not None and published_frames is None:
        raise ValueError("filtered H3 video output requires published_frames")
    if published_frames is not None and published_frames < 1:
        raise ValueError("published H3 video frame count must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("H3 video CRF must be in [0, 51]")
    video = frames[0].permute(1, 2, 3, 0).detach().cpu().numpy()
    if frames.dtype != torch.uint8:
        video = (video.clip(0, 1) * 255.0).round().astype("uint8")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to save H3 decoded video")
    with tempfile.TemporaryDirectory(prefix="digitalhuman-h3-video-") as temporary:
        encoded = Path(temporary) / "video.mp4"
        frame_height, frame_width = video.shape[1:3]
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{frame_width}x{frame_height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
        ]
        if video_filter is not None:
            command.extend(("-vf", video_filter))
        if published_frames is not None:
            command.extend(("-frames:v", str(published_frames)))
        command.extend(
            (
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(encoded),
            )
        )
        process = subprocess.run(
            command,
            input=video.tobytes(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode H3 video: {detail}")
        if verify_decode:
            verify = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(encoded),
                    "-map",
                    "0:v:0",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if verify.returncode != 0:
                detail = verify.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"H3 MP4 video decode verification failed: {detail}"
                )
        published_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".part",
                dir=path.parent,
                delete=False,
            ) as handle:
                published_temp = Path(handle.name)
            shutil.copyfile(encoded, published_temp)
            published_temp.replace(path)
        finally:
            if published_temp is not None:
                published_temp.unlink(missing_ok=True)
    return path.resolve(strict=True)


def write_h3_rgb24_shard(
    path: Path,
    *,
    frames: torch.Tensor,
    frame_start: int,
) -> dict[str, Any]:
    """Write one contiguous decoded-frame shard without a giant bytes copy."""

    if frames.ndim != 5 or int(frames.shape[0]) != 1 or int(frames.shape[1]) != 3:
        raise ValueError(f"unexpected H3 output frames shape {tuple(frames.shape)}")
    if frame_start < 0:
        raise ValueError("H3 RGB24 shard frame start must be non-negative")
    path = path.resolve()
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    video = frames[0].permute(1, 2, 3, 0).detach().cpu().contiguous().numpy()
    if frames.dtype != torch.uint8:
        video = (video.clip(0, 1) * 255.0).round().astype("uint8")
    frame_count, frame_height, frame_width = video.shape[:3]
    with path.open("wb") as handle:
        video.tofile(handle)
    return {
        "path": str(path.resolve(strict=True)),
        "frame_start": int(frame_start),
        "frame_count": int(frame_count),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "rgb24_bytes": int(video.nbytes),
    }


def save_h3_video_mp4_from_rgb24_shards(
    path: Path,
    *,
    shards: Sequence[Mapping[str, Any]],
    fps: int,
    video_filter: str | None = None,
    published_frames: int | None = None,
    crf: int = 12,
    verify_decode: bool = True,
) -> Path:
    """Encode ordered RGB24 shard files with the canonical H3 video settings."""

    path = path.resolve()
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.suffix.lower() != ".mp4":
        raise ValueError("H3 decoded output path must end in .mp4")
    if not shards or fps <= 0:
        raise ValueError("H3 sharded video output requires frames and a positive fps")
    if video_filter is not None and published_frames is None:
        raise ValueError("filtered H3 video output requires published_frames")
    if published_frames is not None and published_frames < 1:
        raise ValueError("published H3 video frame count must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("H3 video CRF must be in [0, 51]")

    ordered = sorted(shards, key=lambda item: int(item["frame_start"]))
    frame_width = int(ordered[0]["frame_width"])
    frame_height = int(ordered[0]["frame_height"])
    next_frame = 0
    shard_paths: list[Path] = []
    for shard in ordered:
        frame_start = int(shard["frame_start"])
        frame_count = int(shard["frame_count"])
        if frame_start != next_frame:
            raise RuntimeError("H3 RGB24 frame shards are not contiguous")
        if (
            int(shard["frame_width"]) != frame_width
            or int(shard["frame_height"]) != frame_height
        ):
            raise RuntimeError("H3 RGB24 frame shard geometry differs")
        expected_bytes = frame_count * frame_width * frame_height * 3
        shard_path = Path(str(shard["path"])).resolve(strict=True)
        if int(shard["rgb24_bytes"]) != expected_bytes:
            raise RuntimeError("H3 RGB24 frame shard byte count differs")
        if shard_path.stat().st_size != expected_bytes:
            raise RuntimeError("H3 RGB24 frame shard file size differs")
        shard_paths.append(shard_path)
        next_frame += frame_count
    if next_frame < 1:
        raise ValueError("H3 sharded video output cannot be empty")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to save H3 decoded video")
    with tempfile.TemporaryDirectory(prefix="digitalhuman-h3-video-") as temporary:
        encoded = Path(temporary) / "video.mp4"
        command = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{frame_width}x{frame_height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
        ]
        if video_filter is not None:
            command.extend(("-vf", video_filter))
        if published_frames is not None:
            command.extend(("-frames:v", str(published_frames)))
        command.extend(
            (
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(encoded),
            )
        )
        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            assert process.stdin is not None
            try:
                for shard_path in shard_paths:
                    with shard_path.open("rb") as handle:
                        shutil.copyfileobj(
                            handle,
                            process.stdin,
                            length=16 * 1024 * 1024,
                        )
                process.stdin.close()
                returncode = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
            stderr_file.seek(0)
            stderr = stderr_file.read()
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode H3 video: {detail}")
        if verify_decode:
            verify = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(encoded),
                    "-map",
                    "0:v:0",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if verify.returncode != 0:
                detail = verify.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"H3 MP4 video decode verification failed: {detail}"
                )
        published_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".part",
                dir=path.parent,
                delete=False,
            ) as handle:
                published_temp = Path(handle.name)
            shutil.copyfile(encoded, published_temp)
            published_temp.replace(path)
        finally:
            if published_temp is not None:
                published_temp.unlink(missing_ok=True)
    return path.resolve(strict=True)


def save_h3_mp4(
    path: Path,
    *,
    frames: torch.Tensor,
    audio: torch.Tensor,
    fps: int,
    sample_rate: int,
    lossless_audio_path: Path | None = None,
) -> Path:
    """Publish one decodable MP4 and, when requested, decoded float WAV audio."""

    path = path.resolve()
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.suffix.lower() != ".mp4":
        raise ValueError("H3 decoded output path must end in .mp4")
    lossless_audio = None
    if lossless_audio_path is not None:
        lossless_audio = lossless_audio_path.expanduser().resolve()
        if (
            lossless_audio.suffix.lower() != ".wav"
            or lossless_audio.parent != path.parent
        ):
            raise ValueError(
                "H3 lossless audio sidecar must be a WAV beside the decoded MP4"
            )
    if frames.ndim != 5 or int(frames.shape[0]) != 1 or int(frames.shape[1]) != 3:
        raise ValueError(f"unexpected H3 output frames shape {tuple(frames.shape)}")
    if audio.ndim != 3 or int(audio.shape[0]) != 1 or int(audio.shape[1]) not in {1, 2}:
        raise ValueError(f"unexpected H3 output audio shape {tuple(audio.shape)}")
    if int(frames.shape[2]) < 1 or int(audio.shape[2]) < 1:
        raise ValueError("H3 decoded output cannot contain empty video or audio")
    if fps <= 0 or sample_rate <= 0:
        raise ValueError("H3 output fps and sample rate must be positive")
    video = frames[0].permute(1, 2, 3, 0).detach().clamp(0, 1).cpu().numpy()
    video = (video * 255.0).round().astype("uint8")
    waveform = audio[0].detach().float().transpose(0, 1).cpu().numpy()
    if not waveform.any():
        raise RuntimeError("H3 decoded audio is exactly silent")
    wavfile = importlib.import_module("scipy.io.wavfile")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to save H3 decoded video")
    with tempfile.TemporaryDirectory(prefix="digitalhuman-h3-output-") as temporary:
        root = Path(temporary)
        silent = root / "video.mp4"
        wav = root / "audio.wav"
        muxed = root / "audio_video.mp4"
        frame_height, frame_width = video.shape[1:3]
        encode = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{frame_width}x{frame_height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "12",
                "-pix_fmt",
                "yuv420p",
                str(silent),
            ],
            input=video.tobytes(),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if encode.returncode != 0:
            detail = encode.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to encode H3 video: {detail}")
        # Preserve the existing mux behavior while allowing quality evaluation
        # to retain the original decoded float samples before AAC encoding.
        wavfile.write(wav, sample_rate, waveform.clip(-1, 1))
        mux = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(silent),
                "-i",
                str(wav),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-ar",
                str(sample_rate),
                "-ac",
                str(waveform.shape[1]),
                "-shortest",
                "-movflags",
                "+faststart",
                str(muxed),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if mux.returncode != 0:
            detail = mux.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed to mux H3 audio/video: {detail}")
        verify = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-i",
                str(muxed),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if verify.returncode != 0:
            detail = verify.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"H3 MP4 video/audio decode verification failed: {detail}"
            )
        published_temp: Path | None = None
        lossless_temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".part",
                dir=path.parent,
                delete=False,
            ) as handle:
                published_temp = Path(handle.name)
            shutil.copyfile(muxed, published_temp)
            if lossless_audio is not None:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{lossless_audio.name}.",
                    suffix=".wav",
                    dir=path.parent,
                    delete=False,
                ) as handle:
                    lossless_temp = Path(handle.name)
                wavfile.write(lossless_temp, sample_rate, waveform)
                lossless_temp.replace(lossless_audio)
            published_temp.replace(path)
        finally:
            if published_temp is not None:
                published_temp.unlink(missing_ok=True)
            if lossless_temp is not None:
                lossless_temp.unlink(missing_ok=True)
    return path.resolve(strict=True)


__all__ = [
    "H3LatentDecoder",
    "decode_h3_latents",
    "save_h3_mp4",
    "save_h3_video_mp4",
    "save_h3_video_mp4_from_rgb24_shards",
    "write_h3_rgb24_shard",
]
