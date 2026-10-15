"""Previous-request C0 video tail, kept independently of rolling AV history."""

import os

from .cache import TemporaryVideoPrefix, VideoKV, VIDEO_TOKEN_TAG


def prompt_local_prefix_video_latents():
    return int(os.environ.get("H3_PROMPT_START_LOCAL_PREFIX_VIDEO_LATENTS", "2"))


class PromptLocalVideoPrefix:
    def __init__(self, temporal_latents):
        if temporal_latents < 0:
            raise ValueError("prompt-local video prefix length must be nonnegative")
        self.temporal_latents = temporal_latents
        self.active = None
        self.pending = None
        self.active_source_block = None
        self.pending_source_block = None
        self.captures = 0

    def begin_request(self):
        if self.active is not None:
            self.active.release()
        self.active, self.pending = self.pending, None
        self.active_source_block = self.pending_source_block
        self.pending_source_block = None

    def capture_c0(self, cache, phase_index, rows_per_temporal_latent):
        if not self.temporal_latents or phase_index != 0:
            return
        if cache._staged is not None or not cache._commit_token_counts:
            raise RuntimeError("prompt-local video prefix requires a completed C0 commit")

        import torch

        video_rows = [
            row for row, tag in enumerate(cache._commit_token_tags[-1])
            if tag == VIDEO_TOKEN_TAG
        ]
        requested_rows = self.temporal_latents * rows_per_temporal_latent
        if (
            rows_per_temporal_latent <= 0
            or len(video_rows) % rows_per_temporal_latent
            or len(video_rows) < requested_rows
        ):
            raise ValueError("C0 commit cannot supply the requested complete video tail")
        selected_rows = video_rows[-requested_rows:]
        current_tokens = cache._commit_token_counts[-1]
        rows_by_device = {}
        layers = {}
        for name in cache.contract.layer_names:
            if cache.segmented_fp8_history:
                block = cache._history_blocks[name][-1]
                key, value = block.key, block.value
            else:
                block = cache._history[name]
                key, value = block.key[-current_tokens:], block.value[-current_tokens:]
            rows = rows_by_device.get(key.device)
            if rows is None:
                rows = torch.tensor(selected_rows, dtype=torch.long, device=key.device)
                rows_by_device[key.device] = rows
            # index_select owns its storage; rolling-history eviction cannot
            # invalidate the prefix while the next request is using it.
            layers[name] = VideoKV(
                key=key.index_select(0, rows).detach(),
                value=value.index_select(0, rows).detach(),
            )
        prefix = TemporaryVideoPrefix(layers, contract=cache.contract)
        if self.pending is not None:
            self.pending.release()
        self.pending = prefix
        self.pending_source_block = cache.committed_blocks - 1
        self.captures += 1

    def release(self):
        for prefix in (self.active, self.pending):
            if prefix is not None:
                prefix.release()
        self.active = self.pending = None
        self.active_source_block = self.pending_source_block = None

    def report(self):
        return {
            "temporal_latents": self.temporal_latents,
            "active_tokens": self.active.token_count if self.active is not None else 0,
            "pending_tokens": self.pending.token_count if self.pending is not None else 0,
            "active_source_block": self.active_source_block,
            "pending_source_block": self.pending_source_block,
            "captures": self.captures,
        }
