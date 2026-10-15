# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen presentation construction without a serving framework."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
TEXT_TAG = 1
VIDEO_TAG = 0


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _vision_ids(tokenizer: Any, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("image token count must be positive")
    return (
        [tokenizer.convert_tokens_to_ids(VISION_START)]
        + [tokenizer.convert_tokens_to_ids(IMAGE_PAD)] * int(count)
        + [tokenizer.convert_tokens_to_ids(VISION_END)]
    )


def minimax_h3_text_only_ids(tokenizer: Any, prompt: str) -> torch.Tensor:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    return torch.tensor(_text_ids(tokenizer, prompt), dtype=torch.long)


def minimax_h3_multi_image_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    if not image_token_counts:
        raise ValueError("image_token_counts must be non-empty")
    ids: list[int] = []
    tags: list[int] = []
    for index, count in enumerate(image_token_counts, start=1):
        label = _text_ids(tokenizer, f"<Picture {index}>: ")
        vision = _vision_ids(tokenizer, int(count))
        ids.extend(label)
        tags.extend([TEXT_TAG] * len(label))
        ids.extend(vision)
        tags.extend([VIDEO_TAG] * len(vision))
    text = _text_ids(tokenizer, prompt)
    ids.extend(text)
    tags.extend([TEXT_TAG] * len(text))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(tags, dtype=torch.long)


__all__ = [
    "IMAGE_PAD",
    "TEXT_TAG",
    "VIDEO_TAG",
    "VISION_END",
    "VISION_START",
    "minimax_h3_multi_image_presentation",
    "minimax_h3_text_only_ids",
]
