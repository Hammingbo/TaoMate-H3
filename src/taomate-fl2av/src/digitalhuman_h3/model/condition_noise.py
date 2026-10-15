# SPDX-License-Identifier: Apache-2.0
"""Released MiniMax H3 visual-condition noise augmentation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .packed_tokens import minimax_h3_patchify_video_latent


def minimax_h3_imgvid_cond_noise_aug_rows(
    clean_rows: torch.Tensor,
    *,
    condition_shapes: Sequence[Sequence[int]],
    target_latent_t: int,
    imgvid_cond_num_frames: int,
    generator: torch.Generator,
    noise_aug: float = 0.999,
) -> torch.Tensor:
    if not 0.0 <= noise_aug <= 1.0:
        raise ValueError("noise_aug must be in [0, 1]")
    if noise_aug == 1.0:
        return clean_rows
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != 96:
        raise ValueError("visual condition rows must be [N,96]")
    shapes = [tuple(int(item) for item in shape) for shape in condition_shapes]
    if not shapes or any(len(shape) != 3 or min(shape) <= 0 for shape in shapes):
        raise ValueError("condition_shapes must contain positive (T,H,W) triples")
    expected = sum(t * (h // 2) * (w // 2) for t, h, w in shapes)
    if expected != int(clean_rows.shape[0]):
        raise ValueError("visual condition row count differs from condition shapes")
    outputs: list[torch.Tensor] = []
    offset = 0
    timestep = torch.tensor(noise_aug, dtype=torch.float32, device=clean_rows.device)
    for latent_t, latent_h, latent_w in shapes:
        full_t = int(target_latent_t) + int(imgvid_cond_num_frames)
        noise = torch.randn(
            1,
            24,
            full_t,
            latent_h,
            latent_w,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )[:, :, :latent_t]
        noise_rows = minimax_h3_patchify_video_latent(
            noise, patch_size=(1, 2, 2)
        ).to(clean_rows.device)
        count = int(noise_rows.shape[0])
        clean = clean_rows[offset : offset + count].float()
        outputs.append(timestep * clean + (1.0 - timestep) * noise_rows)
        offset += count
    return torch.cat(outputs, dim=0).contiguous()


__all__ = ["minimax_h3_imgvid_cond_noise_aug_rows"]
