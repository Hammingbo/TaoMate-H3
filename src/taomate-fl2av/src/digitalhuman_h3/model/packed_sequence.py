# SPDX-License-Identifier: Apache-2.0
"""Native MiniMax H3 T2VA/FL2VA packed-sequence builder.

The row layout and fp64 position-grid math are kept byte-for-byte compatible
with the pinned Apache-2.0 MiniMax H3 reference behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .architecture import MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
from .task_profiles import MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES


_INTERP = 32
_T_GROUP = 5
_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
_FRAME_RESCALE = 5.0 / 3.0
_PATCH_H = 2
_PATCH_W = 2


def _keyframe_cond_frame_indices(
    *,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if not include_keyframe_cond:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted when keyframe cond is not included"
            )
        return []
    if keyframe_frame_indices is None:
        raise ValueError("strict fl2va packed layout requires keyframe_frame_indices")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in keyframe_frame_indices
    ):
        raise ValueError(
            "strict fl2va packed layout requires integer keyframe_frame_indices"
        )
    out = list(keyframe_frame_indices)
    if tuple(out) not in MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "strict fl2va packed layout requires keyframe_frame_indices in "
            f"{MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES!r}, got {out!r}"
        )
    return out


def _resolve_keyframe_frame_indices(
    frame_indices: Sequence[int],
    *,
    frame_count: int | None,
) -> list[int]:
    if frame_indices and frame_count is None:
        raise ValueError(
            "frame_count is required when keyframe_frame_indices are provided"
        )
    if frame_count is None:
        return []
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    seen: dict[int, int] = {}
    resolved: list[int] = []
    for block_index, semantic_index in enumerate(frame_indices):
        if semantic_index == -1:
            resolved_index = frame_count - 1
        elif 0 <= semantic_index < frame_count:
            resolved_index = semantic_index
        else:
            raise ValueError(
                f"keyframe frame index {semantic_index} must be -1 or in "
                f"[0, {frame_count})"
            )
        previous = seen.get(resolved_index)
        if previous is not None:
            raise ValueError(
                f"keyframe frame index at block {block_index} resolves to "
                f"{resolved_index}, already bound by block {previous}"
            )
        seen[resolved_index] = block_index
        resolved.append(resolved_index)
    return resolved


def _temporal_position_span(temporal_length: int) -> float:
    spans = np.ones(int(temporal_length), dtype=np.float64) * _FRAME_RESCALE
    for token_index in range(_T_GROUP):
        spans[token_index::_T_GROUP] *= _FRAME_PER_TOKEN[token_index]
    return float(spans.sum())


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * _INTERP
    return torch.from_numpy(grid).to(torch.float64)


def _video_t_grid(n: int, origin: float) -> torch.Tensor:
    spans = torch.tensor(
        [_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )


def _video_t_span(n: int) -> float:
    return sum(_FRAME_RESCALE * _FRAME_PER_TOKEN[k % _T_GROUP] for k in range(n))


def _positive_block_int(
    block: Mapping[str, object],
    key: str,
    path: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = block.get(key)
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path}.{key} must be a {qualifier} integer")
    return value


def _cat_ranges(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    audio_is_condition: bool = False,
    include_keyframe_cond: bool,
    keyframe_frame_indices: list[int] | tuple[int, ...] | None = None,
    frame_count: int | None = None,
) -> dict[str, Any]:
    """Build one positive-branch packed layout: text/cond/audio/video/pad."""

    if min(text_len, latent_t, latent_h, latent_w, audio_t, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 packed-sequence dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")
    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    frame_rows = ph * pw
    cond_frame_indices = _keyframe_cond_frame_indices(
        include_keyframe_cond=include_keyframe_cond,
        keyframe_frame_indices=keyframe_frame_indices,
    )
    resolved_cond_frame_indices = _resolve_keyframe_frame_indices(
        cond_frame_indices,
        frame_count=frame_count,
    )
    cond_rows = len(cond_frame_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + cond_rows + audio_rows + video_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    cond_sl = slice(text_len, text_len + cond_rows)
    audio_sl = slice(cond_sl.stop, cond_sl.stop + audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + video_rows)
    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    img_pos = (
        torch.cat([torch.arange(cond_sl.start, cond_sl.stop), target_img_pos])
        if cond_rows
        else target_img_pos
    )
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[cond_rows:] = True
    audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    audio_update_mask = torch.full(
        (audio_rows,), not audio_is_condition, dtype=torch.bool
    )
    text_pos = torch.arange(0, text_len)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    t_grid = _video_t_grid(latent_t, float(text_len))
    sqrt_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)
    video_grid = grid[video_sl].view(latent_t, frame_rows, 3)
    video_grid[:, :, 0] = t_grid[:, None]
    video_grid[:, :, 1:] = frame[None]
    for block_index, pixel_index in enumerate(resolved_cond_frame_indices):
        cond_block = slice(
            cond_sl.start + block_index * frame_rows,
            cond_sl.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            cond_t = float(text_len)
        elif frame_count is not None and pixel_index == frame_count - 1:
            cond_t = float(text_len) + _temporal_position_span(latent_t) - _FRAME_RESCALE
        else:
            raise ValueError(
                "fl2va packed layout only supports first/last keyframe anchors, "
                f"got resolved frame index {pixel_index}"
            )
        grid[cond_block, 0] = cond_t
        grid[cond_block, 1:] = frame
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    token_tags[img_pos] = 0
    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "audio_update_mask": audio_update_mask,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
    }


def minimax_h3_audio_only_packed_sequence(
    *,
    text_len: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    audio_channel: int = 2,
) -> dict[str, Any]:
    """Build the Base teacher's text/audio sequence without video tokens."""

    if min(text_len, audio_t, latent_h, latent_w, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 audio-only dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")
    audio_rows = audio_t * audio_channel
    used = text_len + audio_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    text_sl = slice(0, text_len)
    audio_sl = slice(text_len, text_len + audio_rows)
    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    sqrt_area = np.sqrt(latent_h * latent_w)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_sl] = 2
    return {
        "seq_len": seq_len,
        "img_pos": torch.empty(0, dtype=torch.long),
        "audio_pos": torch.arange(audio_sl.start, audio_sl.stop),
        "audio_update_mask": torch.ones(audio_rows, dtype=torch.bool),
        "text_pos": torch.arange(text_len),
        "update_mask": torch.empty(0, dtype=torch.bool),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def minimax_h3_audio_only_frozen_prefix_packed_sequence(
    *,
    text_len: int,
    ref_audio_t: int,
    audio_t: int,
    latent_h: int,
    latent_w: int,
    audio_channel: int = 2,
    reference_time_start: int | None = None,
    target_time_start: int | None = None,
) -> dict[str, Any]:
    """Build one audio-only request with a frozen clean reference prefix."""

    if min(text_len, ref_audio_t, audio_t, latent_h, latent_w, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 audio-reference dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")

    ref_rows = ref_audio_t * audio_channel
    target_rows = audio_t * audio_channel
    used = text_len + ref_rows + target_rows
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )
    text_sl = slice(0, text_len)
    ref_sl = slice(text_sl.stop, text_sl.stop + ref_rows)
    target_sl = slice(ref_sl.stop, ref_sl.stop + target_rows)

    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)
    ref_start = text_len if reference_time_start is None else reference_time_start
    target_start = (
        ref_start + ref_audio_t
        if target_time_start is None
        else target_time_start
    )
    ref_t_grid = float(ref_start) + torch.arange(ref_audio_t, dtype=torch.float64)
    target_t_grid = float(target_start) + torch.arange(audio_t, dtype=torch.float64)
    grid[ref_sl, 0] = ref_t_grid.repeat(audio_channel)
    grid[target_sl, 0] = target_t_grid.repeat(audio_channel)

    sqrt_area = np.sqrt(latent_h * latent_w)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, sqrt_area)
    for audio_sl, temporal_rows in (
        (ref_sl, ref_audio_t),
        (target_sl, audio_t),
    ):
        grid[audio_sl.start : audio_sl.start + temporal_rows, 2] = float(w_grid[0])
        grid[audio_sl.start + temporal_rows : audio_sl.stop, 2] = float(w_grid[-1])

    audio_pos = torch.arange(ref_sl.start, target_sl.stop)
    audio_update_mask = torch.zeros(ref_rows + target_rows, dtype=torch.bool)
    audio_update_mask[ref_rows:] = True
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_pos] = 2
    return {
        "seq_len": seq_len,
        "img_pos": torch.empty(0, dtype=torch.long),
        "audio_pos": audio_pos,
        "audio_update_mask": audio_update_mask,
        "text_pos": torch.arange(text_len),
        "update_mask": torch.empty(0, dtype=torch.bool),
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


def minimax_h3_packed_sequence_ref2va(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    ref_blocks: Sequence[Mapping[str, object]],
    audio_channel: int = 2,
) -> dict[str, Any]:
    """Build the official Ref2VA text/ref/audio/video packed layout.

    Reference blocks are consumed in presentation order. Video references keep
    their channel-major audio rows immediately before their visual rows. Only
    the final target audio/video rows are marked for denoising updates.
    """

    if min(text_len, latent_t, latent_h, latent_w, audio_t, audio_channel) <= 0:
        raise ValueError("all MiniMax H3 packed-sequence dimensions must be positive")
    if latent_h % _PATCH_H or latent_w % _PATCH_W:
        raise ValueError("latent_h and latent_w must be divisible by the 2x2 patch")

    parsed: list[dict[str, object]] = []
    ref_visual_rows = 0
    ref_audio_rows = 0
    for index, raw in enumerate(ref_blocks):
        path = f"ref_blocks[{index}]"
        kind = raw.get("kind", raw.get("type"))
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind must be a string")
        if kind == "image":
            rh = _positive_block_int(raw, "latent_h", path)
            rw = _positive_block_int(raw, "latent_w", path)
            if rh % _PATCH_H or rw % _PATCH_W:
                raise ValueError(f"{path} latent shape must be divisible by 2x2")
            rows = (rh // _PATCH_H) * (rw // _PATCH_W)
            parsed.append(
                {"kind": kind, "latent_h": rh, "latent_w": rw, "rows": rows}
            )
            ref_visual_rows += rows
        elif kind == "audio":
            rt = _positive_block_int(raw, "ref_audio_t", path, allow_zero=True)
            rows = rt * audio_channel
            parsed.append({"kind": kind, "ref_audio_t": rt, "audio_rows": rows})
            ref_audio_rows += rows
        elif kind in {"video", "video_audio"}:
            rt = _positive_block_int(raw, "ref_audio_t", path, allow_zero=True)
            vt = _positive_block_int(raw, "latent_t", path)
            vh = _positive_block_int(raw, "latent_h", path)
            vw = _positive_block_int(raw, "latent_w", path)
            if vh % _PATCH_H or vw % _PATCH_W:
                raise ValueError(f"{path} latent shape must be divisible by 2x2")
            frame_rows = (vh // _PATCH_H) * (vw // _PATCH_W)
            audio_rows = rt * audio_channel
            video_rows = vt * frame_rows
            parsed.append(
                {
                    "kind": kind,
                    "ref_audio_t": rt,
                    "latent_t": vt,
                    "latent_h": vh,
                    "latent_w": vw,
                    "frame_rows": frame_rows,
                    "audio_rows": audio_rows,
                    "video_rows": video_rows,
                }
            )
            ref_audio_rows += audio_rows
            ref_visual_rows += video_rows
        else:
            raise ValueError(f"{path}.kind is unsupported: {kind!r}")

    ph, pw = latent_h // _PATCH_H, latent_w // _PATCH_W
    target_frame_rows = ph * pw
    target_video_rows = latent_t * target_frame_rows
    target_audio_rows = audio_t * audio_channel
    used = (
        text_len
        + ref_visual_rows
        + ref_audio_rows
        + target_audio_rows
        + target_video_rows
    )
    seq_len = (
        (used + MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT - 1)
        // MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        * MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    )

    text_sl = slice(0, text_len)
    cursor = text_len
    block_slices: list[dict[str, object]] = []
    for item in parsed:
        kind = str(item["kind"])
        if kind == "image":
            visual_sl = slice(cursor, cursor + int(item["rows"]))
            cursor = visual_sl.stop
            block_slices.append({**item, "visual_sl": visual_sl})
        elif kind == "audio":
            audio_sl = slice(cursor, cursor + int(item["audio_rows"]))
            cursor = audio_sl.stop
            block_slices.append({**item, "audio_sl": audio_sl})
        else:
            audio_sl = slice(cursor, cursor + int(item["audio_rows"]))
            visual_sl = slice(audio_sl.stop, audio_sl.stop + int(item["video_rows"]))
            cursor = visual_sl.stop
            block_slices.append(
                {**item, "audio_sl": audio_sl, "visual_sl": visual_sl}
            )

    audio_sl = slice(cursor, cursor + target_audio_rows)
    video_sl = slice(audio_sl.stop, audio_sl.stop + target_video_rows)
    grid = torch.zeros(seq_len, 3, dtype=torch.float64)
    grid[text_sl, 0] = torch.arange(text_len, dtype=torch.float64)

    target_area = np.sqrt(latent_h * latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, _PATCH_H, target_area)
    w_grid = _axis_from_sqrt_area(latent_w, _PATCH_W, target_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    target_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    ref_visual_positions: list[torch.Tensor] = []
    ref_audio_positions: list[torch.Tensor] = []
    t_cursor = float(text_len)
    for item in block_slices:
        kind = str(item["kind"])
        if kind == "image":
            visual_sl = item["visual_sl"]
            assert isinstance(visual_sl, slice)
            ref_visual_positions.append(torch.arange(visual_sl.start, visual_sl.stop))
            rh, rw = int(item["latent_h"]), int(item["latent_w"])
            area = np.sqrt(rh * rw)
            ref_hh, ref_ww = torch.meshgrid(
                _axis_from_sqrt_area(rh, _PATCH_H, area),
                _axis_from_sqrt_area(rw, _PATCH_W, area),
                indexing="ij",
            )
            grid[visual_sl, 0] = t_cursor
            grid[visual_sl, 1] = ref_hh.reshape(-1)
            grid[visual_sl, 2] = ref_ww.reshape(-1)
            t_cursor += 1.0
            continue

        ref_audio_sl = item["audio_sl"]
        assert isinstance(ref_audio_sl, slice)
        ref_t = int(item["ref_audio_t"])
        ref_audio_positions.append(torch.arange(ref_audio_sl.start, ref_audio_sl.stop))
        if kind == "audio":
            ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
            grid[ref_audio_sl, 0] = ref_t_grid.repeat(audio_channel)
            if ref_t:
                grid[ref_audio_sl.start : ref_audio_sl.start + ref_t, 2] = float(w_grid[0])
                grid[ref_audio_sl.start + ref_t : ref_audio_sl.stop, 2] = float(w_grid[-1])
            t_cursor += float(ref_t)
            continue

        visual_sl = item["visual_sl"]
        assert isinstance(visual_sl, slice)
        vt, vh, vw = int(item["latent_t"]), int(item["latent_h"]), int(item["latent_w"])
        ref_visual_positions.append(torch.arange(visual_sl.start, visual_sl.stop))
        ref_area = np.sqrt(vh * vw)
        rv_h = _axis_from_sqrt_area(vh, _PATCH_H, ref_area)
        rv_w = _axis_from_sqrt_area(vw, _PATCH_W, ref_area)
        rv_hh, rv_ww = torch.meshgrid(rv_h, rv_w, indexing="ij")
        ref_t_grid = t_cursor + torch.arange(ref_t, dtype=torch.float64)
        grid[ref_audio_sl, 0] = ref_t_grid.repeat(audio_channel)
        if ref_t:
            grid[ref_audio_sl.start : ref_audio_sl.start + ref_t, 2] = float(rv_w[0])
            grid[ref_audio_sl.start + ref_t : ref_audio_sl.stop, 2] = float(rv_w[-1])
        rv_frame = torch.stack([rv_hh.reshape(-1), rv_ww.reshape(-1)], dim=-1)
        rv_grid = grid[visual_sl].view(vt, int(item["frame_rows"]), 3)
        rv_grid[:, :, 0] = _video_t_grid(vt, t_cursor)[:, None]
        rv_grid[:, :, 1:] = rv_frame[None]
        t_cursor += max(float(ref_t), _video_t_span(vt))

    audio_t_grid = t_cursor + torch.arange(audio_t, dtype=torch.float64)
    grid[audio_sl, 0] = audio_t_grid.repeat(audio_channel)
    grid[audio_sl.start : audio_sl.start + audio_t, 2] = float(w_grid[0])
    grid[audio_sl.start + audio_t : audio_sl.stop, 2] = float(w_grid[-1])
    video_grid = grid[video_sl].view(latent_t, target_frame_rows, 3)
    video_grid[:, :, 0] = _video_t_grid(latent_t, t_cursor)[:, None]
    video_grid[:, :, 1:] = target_frame[None]

    target_img_pos = torch.arange(video_sl.start, video_sl.stop)
    target_audio_pos = torch.arange(audio_sl.start, audio_sl.stop)
    img_pos = _cat_ranges(ref_visual_positions + [target_img_pos])
    audio_pos = _cat_ranges(ref_audio_positions + [target_audio_pos])
    update_mask = torch.zeros(img_pos.shape[0], dtype=torch.bool)
    update_mask[ref_visual_rows:] = True
    audio_update_mask = torch.zeros(audio_pos.shape[0], dtype=torch.bool)
    audio_update_mask[ref_audio_rows:] = True
    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_sl] = 1
    token_tags[audio_pos] = 2
    token_tags[img_pos] = 0
    return {
        "seq_len": seq_len,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": torch.arange(text_len),
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": grid,
        "token_tags": token_tags,
        "cu_seqlens": torch.tensor([0, used, seq_len], dtype=torch.int32),
    }


__all__ = [
    "minimax_h3_audio_only_frozen_prefix_packed_sequence",
    "minimax_h3_audio_only_packed_sequence",
    "minimax_h3_packed_sequence",
    "minimax_h3_packed_sequence_ref2va",
]
