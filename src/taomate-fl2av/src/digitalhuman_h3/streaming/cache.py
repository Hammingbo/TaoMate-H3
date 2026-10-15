from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


MAIN_LAYER_NAMES = tuple(f"blocks.{index}.attn" for index in range(50))
VIDEO_TOKEN_TAG = 0
AUDIO_TOKEN_TAG = 2
AV_TOKEN_TAGS = frozenset({VIDEO_TOKEN_TAG, AUDIO_TOKEN_TAG})


@dataclass(frozen=True)
class KVContract:
    layer_names: tuple[str, ...] = MAIN_LAYER_NAMES
    local_heads: int = 14
    head_dim: int = 128
    dtype: str = "torch.bfloat16"


def kv_contract_from_model(model: Any) -> KVContract:
    """Resolve post-Ulysses local K/V heads from the live H3 topology."""

    arch = getattr(model, "arch", None)
    context = getattr(model, "parallel_context", None)
    total_heads = int(getattr(arch, "num_attention_heads", 0))
    head_dim = int(getattr(arch, "attention_head_dim", 0))
    tp_size = int(getattr(context, "tp_world_size", 0))
    ulysses_size = int(getattr(context, "ulysses_world_size", 0))
    parallel_size = tp_size * ulysses_size
    if (
        total_heads <= 0
        or head_dim <= 0
        or parallel_size <= 0
        or total_heads % tp_size
    ):
        raise ValueError("live H3 attention topology cannot resolve local K/V heads")
    return KVContract(
        local_heads=(total_heads // tp_size + ulysses_size - 1) // ulysses_size,
        head_dim=head_dim,
    )


@dataclass(frozen=True)
class VideoKV:
    key: Any
    value: Any
    modality: str = "video"
    sol_summary_cache: dict[tuple[int, int], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class AVKV:
    key: Any
    value: Any
    modality: str = "audio_video"
    fp8_key: Any | None = None
    fp8_value: Any | None = None
    fp8_key_descale: Any | None = None
    fp8_value_descale: Any | None = None
    key_summary: Any | None = None
    sol_summary_cache: dict[tuple[int, int], Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )


def _validate_video_pair(pair: VideoKV, contract: KVContract, *, layer: str) -> None:
    import torch

    if not isinstance(pair, VideoKV) or pair.modality != "video":
        raise TypeError(f"{layer} prefix must be an explicitly video-only VideoKV")
    key, value = pair.key, pair.value
    if not torch.is_tensor(key) or not torch.is_tensor(value):
        raise TypeError(f"{layer} K/V must be tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError(f"{layer} K/V must have the same [tokens, heads, dim] shape")
    if tuple(key.shape[1:]) != (contract.local_heads, contract.head_dim):
        raise ValueError(
            f"{layer} K/V local shape must be [{contract.local_heads}, {contract.head_dim}]"
        )
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError(f"{layer} K/V must be bfloat16")
    if not key.is_cuda or not value.is_cuda or key.device != value.device:
        raise ValueError(f"{layer} K/V must share one CUDA device")


def _validate_av_pair(pair: AVKV, contract: KVContract, *, layer: str) -> None:
    import torch

    if not isinstance(pair, AVKV) or pair.modality != "audio_video":
        raise TypeError(f"{layer} persistent prefix must be an audio-video AVKV")
    key, value = pair.key, pair.value
    if not torch.is_tensor(key) or not torch.is_tensor(value):
        raise TypeError(f"{layer} K/V must be tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError(f"{layer} K/V must have the same [tokens, heads, dim] shape")
    if tuple(key.shape[1:]) != (contract.local_heads, contract.head_dim):
        raise ValueError(
            f"{layer} K/V local shape must be [{contract.local_heads}, {contract.head_dim}]"
        )
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise ValueError(f"{layer} K/V must be bfloat16")
    if not key.is_cuda or not value.is_cuda or key.device != value.device:
        raise ValueError(f"{layer} K/V must share one CUDA device")


def _validate_committed_media_tags(tags: tuple[int, ...]) -> None:
    if not tags or frozenset(tags) != AV_TOKEN_TAGS:
        raise ValueError("clean H3 commit must contain both video and audio rows")


class TemporaryVideoPrefix:
    """Ephemeral per-layer video self-attention K/V.

    This object is never inserted into persistent history. Audio and generic
    cross-modal entries are rejected by construction.
    """

    def __init__(
        self,
        layers: Mapping[str, VideoKV],
        *,
        contract: KVContract | None = None,
    ) -> None:
        self.contract = contract or KVContract()
        if set(layers) != set(self.contract.layer_names):
            raise ValueError("temporary full-prefix must cover all 50 main attention layers")
        self._layers = {name: layers[name] for name in self.contract.layer_names}
        for name, pair in self._layers.items():
            _validate_video_pair(pair, self.contract, layer=name)
        self.token_count = int(self._layers[self.contract.layer_names[0]].key.shape[0])
        if any(int(pair.key.shape[0]) != self.token_count for pair in self._layers.values()):
            raise ValueError("temporary full-prefix token counts differ by layer")

    def get(self, layer: str) -> VideoKV:
        return self._layers[layer]

    def release(self) -> None:
        """Release the ephemeral full-prefix K/V after its final forward."""

        self._layers.clear()


class TemporaryAVPrefix:
    """Ephemeral per-layer full-prefix K/V containing canonical audio and video."""

    def __init__(
        self,
        layers: Mapping[str, AVKV],
        *,
        video_token_count: int,
        audio_token_count_per_channel: int,
        contract: KVContract | None = None,
    ) -> None:
        self.contract = contract or KVContract()
        if set(layers) != set(self.contract.layer_names):
            raise ValueError("temporary AV full-prefix must cover all 50 main layers")
        if (
            isinstance(video_token_count, bool)
            or not isinstance(video_token_count, int)
            or video_token_count <= 0
        ):
            raise ValueError("temporary AV full-prefix video count must be positive")
        if (
            isinstance(audio_token_count_per_channel, bool)
            or not isinstance(audio_token_count_per_channel, int)
            or audio_token_count_per_channel <= 0
        ):
            raise ValueError(
                "temporary AV full-prefix per-channel audio count must be positive"
            )
        self.video_token_count = video_token_count
        self.audio_token_count_per_channel = audio_token_count_per_channel
        self.audio_token_count = 2 * audio_token_count_per_channel
        self.token_count = self.video_token_count + self.audio_token_count
        self._layers = {name: layers[name] for name in self.contract.layer_names}
        for name, pair in self._layers.items():
            _validate_av_pair(pair, self.contract, layer=name)
            if int(pair.key.shape[0]) != self.token_count:
                raise ValueError(
                    f"{name} temporary AV full-prefix token count differs"
                )

    def get(self, layer: str) -> AVKV:
        return self._layers[layer]

    def release(self) -> None:
        """Release the ephemeral full-prefix K/V after its final forward."""

        self._layers.clear()


class CleanAVKVCache:
    """Persistent post-RoPE media history committed only from clean AV states."""

    def __init__(
        self,
        contract: KVContract | None = None,
        *,
        segmented_fp8_history: bool = False,
    ) -> None:
        self.contract = contract or KVContract()
        self.segmented_fp8_history = bool(segmented_fp8_history)
        self._history: dict[str, AVKV] = {}
        self._history_blocks: dict[str, list[AVKV]] = {}
        self._staged: dict[str, AVKV] | None = None
        self._staged_commit_indices: Any | None = None
        self._staged_commit_tags: tuple[int, ...] | None = None
        self._block_index = 0
        self._commit_token_counts: list[int] = []
        self._commit_token_tags: list[tuple[int, ...]] = []
        self._attention_index_cache: dict[tuple[Any, ...], Any] = {}
        self._sink_video_only = False
        self._invalid_after_failed_commit = False

    def _require_valid(self) -> None:
        if self._invalid_after_failed_commit:
            raise RuntimeError("H3 cache is invalid after a failed clean commit")

    @property
    def committed_blocks(self) -> int:
        self._require_valid()
        return self._block_index

    @property
    def history_tokens(self) -> int:
        self._require_valid()
        return sum(self._commit_token_counts)

    @property
    def retained_blocks(self) -> int:
        return len(self._commit_token_counts)

    @property
    def history_audio_tokens(self) -> int:
        return sum(
            tag == AUDIO_TOKEN_TAG
            for tags in self._commit_token_tags
            for tag in tags
        )

    @property
    def history_video_tokens(self) -> int:
        return sum(
            tag == VIDEO_TOKEN_TAG
            for tags in self._commit_token_tags
            for tag in tags
        )

    @property
    def clean_commit_active(self) -> bool:
        return self._staged is not None

    @property
    def commit_token_counts(self) -> tuple[int, ...]:
        self._require_valid()
        return tuple(self._commit_token_counts)

    def history(self, layer: str) -> AVKV | None:
        """Return the complete retained causal prefix for attention.

        The default long-video policy retains a video-only block-zero sink plus
        the two most recent clean AV commits. Newly completed phases enter the
        same bounded prefix, so history does not grow with request count.
        """

        self._require_valid()
        if layer not in self.contract.layer_names:
            raise KeyError(f"unknown H3 attention layer: {layer}")
        pair = self._history.get(layer)
        if pair is None and self._history_blocks:
            import torch

            blocks = self._history_blocks.get(layer, ())
            if blocks:
                pair = AVKV(
                    key=torch.cat(tuple(item.key for item in blocks), dim=0),
                    value=torch.cat(tuple(item.value for item in blocks), dim=0),
                )
                self._history[layer] = pair
        if pair is None:
            return None
        if not self._commit_token_counts:
            raise RuntimeError("clean H3 history has no commit boundary ledger")
        return pair

    def recent_history(self, layer: str, count: int) -> AVKV | None:
        """Return a suffix view containing at most ``count`` committed blocks."""

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("recent H3 history count must be non-negative")
        pair = self.history(layer)
        if pair is None or count == 0:
            return None
        if count >= len(self._commit_token_counts):
            return pair
        tokens = sum(self._commit_token_counts[-count:])
        return AVKV(key=pair.key[-tokens:], value=pair.value[-tokens:])

    def sink_and_recent_history(
        self,
        layer: str,
        recent_count: int,
    ) -> AVKV | None:
        """Return block-zero sink plus the newest non-sink commits."""

        import torch

        selected = self.sink_and_recent_history_blocks(layer, recent_count)
        if not selected:
            return None
        if len(selected) == 1:
            return selected[0]
        return AVKV(
            key=torch.cat(tuple(pair.key for pair in selected), dim=0),
            value=torch.cat(tuple(pair.value for pair in selected), dim=0),
        )

    def sink_and_recent_history_blocks(
        self,
        layer: str,
        recent_count: int,
    ) -> tuple[AVKV, ...]:
        """Return block-zero plus ``recent_count`` newest block views."""

        if (
            isinstance(recent_count, bool)
            or not isinstance(recent_count, int)
            or recent_count < 0
        ):
            raise ValueError("recent H3 history count must be non-negative")
        blocks = self.history_blocks(layer)
        if not blocks:
            return ()
        recent_start = max(1, len(blocks) - recent_count)
        return (blocks[0], *blocks[recent_start:])

    def history_blocks(self, layer: str) -> tuple[AVKV, ...]:
        """Return retained history as zero-copy per-block views."""

        self._require_valid()
        if layer not in self.contract.layer_names:
            raise KeyError(f"unknown H3 attention layer: {layer}")
        segmented = self._history_blocks.get(layer)
        if segmented is not None:
            return tuple(segmented)
        pair = self.history(layer)
        if pair is None:
            return ()
        blocks: list[AVKV] = []
        offset = 0
        for count in self._commit_token_counts:
            blocks.append(
                AVKV(
                    key=pair.key.narrow(0, offset, count),
                    value=pair.value.narrow(0, offset, count),
                )
            )
            offset += count
        return tuple(blocks)

    def history_for_fused_current(
        self,
        layer: str,
        *,
        use_sink: bool = False,
        recent_blocks: int | None = None,
    ) -> AVKV | None:
        """Return the persistent part of a hypothetical post-commit view.

        The packed previous document supplies the newest clean block directly.
        This selector therefore reserves one recent slot without mutating the
        cache before the fused transformer forward succeeds.
        """

        if recent_blocks is None:
            return self.history(layer)
        persistent_recent = max(0, recent_blocks - 1)
        if use_sink:
            pair = self.sink_and_recent_history(layer, persistent_recent)
            if (
                pair is not None
                and self._sink_video_only
                and self.retained_blocks + 1 > recent_blocks
                and AUDIO_TOKEN_TAG in self._commit_token_tags[0]
            ):
                import torch

                # The pending document ages the sink out of the recent window.
                # Match post-commit trim without changing the previous document's
                # still-visible AV history during this fused forward.
                sink_tags = self._commit_token_tags[0]
                rows = [
                    row for row, tag in enumerate(sink_tags)
                    if tag == VIDEO_TOKEN_TAG
                ]
                rows.extend(range(len(sink_tags), int(pair.key.shape[0])))
                indices = torch.tensor(rows, dtype=torch.long, device=pair.key.device)
                return self._select_pair_rows(pair, indices)
            return pair
        return self.recent_history(layer, persistent_recent)

    def history_for_attention(
        self,
        layer: str,
        *,
        fused_current: bool,
        recent_full_blocks: int,
        older_video_stride: int,
        use_sink: bool = False,
        recent_blocks: int | None = None,
    ) -> AVKV | None:
        """Select direct history for dense-recent/sparse-older attention."""

        retained = self.retained_blocks
        if fused_current and recent_blocks is not None:
            retained_recent = max(0, recent_blocks - 1)
            count = min(
                retained,
                (1 if use_sink and retained else 0) + retained_recent,
            )
            recent_full_blocks = max(0, recent_full_blocks - 1)
        else:
            count = retained
        return self.attention_history(
            layer,
            count=count,
            recent_full_blocks=recent_full_blocks,
            older_video_stride=older_video_stride,
            preserve_sink=use_sink,
        )

    def history_modality_parts(
        self,
        layer: str,
        block_indices: tuple[int, ...],
    ) -> tuple[tuple[tuple[Any, Any], ...], tuple[tuple[Any, Any], ...]]:
        """Return selected history as exact audio and routable video K/V parts."""

        blocks = self.history_blocks(layer)
        audio_parts: list[tuple[Any, Any]] = []
        video_parts: list[tuple[Any, Any]] = []
        for block_index in block_indices:
            pair = blocks[block_index]
            tags = self._commit_token_tags[block_index]
            for target_tag, destination in (
                (AUDIO_TOKEN_TAG, audio_parts),
                (VIDEO_TOKEN_TAG, video_parts),
            ):
                rows = [
                    row for row, tag in enumerate(tags) if tag == target_tag
                ]
                if not rows:
                    continue
                if rows == list(range(rows[0], rows[-1] + 1)):
                    key = pair.key.narrow(0, rows[0], len(rows))
                    value = pair.value.narrow(0, rows[0], len(rows))
                else:
                    import torch

                    indices = torch.tensor(
                        rows,
                        dtype=torch.long,
                        device=pair.key.device,
                    )
                    key = pair.key.index_select(0, indices)
                    value = pair.value.index_select(0, indices)
                destination.append((key, value))
        return tuple(audio_parts), tuple(video_parts)

    def sol_history_modality_parts(
        self,
        layer: str,
        *,
        fused_current: bool,
        use_sink: bool = False,
        recent_blocks: int | None = None,
    ) -> tuple[tuple[tuple[Any, Any], ...], tuple[tuple[Any, Any], ...]]:
        """Split the same direct fused history view for SOL-Attn routing."""

        retained = self.retained_blocks
        if retained == 0:
            return (), ()
        if fused_current and recent_blocks is not None:
            recent = max(0, recent_blocks - 1)
            if use_sink:
                recent_start = max(1, retained - recent)
                block_indices = (0, *range(recent_start, retained))
            else:
                block_indices = tuple(range(max(0, retained - recent), retained))
        else:
            block_indices = tuple(range(retained))
        return self.history_modality_parts(layer, block_indices)

    def attention_history(
        self,
        layer: str,
        *,
        count: int,
        recent_full_blocks: int,
        older_video_stride: int,
        preserve_sink: bool = False,
    ) -> AVKV | None:
        """Select dense recent blocks and structured-sparse older video KV."""

        import torch

        if recent_full_blocks < 0 or older_video_stride < 1:
            raise ValueError("invalid H3 attention-history policy")
        pair = (
            self.sink_and_recent_history(layer, max(0, count - 1))
            if preserve_sink and count > 0
            else self.recent_history(layer, count)
        )
        if pair is None:
            return None
        if preserve_sink:
            recent_start = max(
                1,
                len(self._commit_token_tags) - max(0, count - 1),
            )
            selected_indices = (
                0,
                *range(recent_start, len(self._commit_token_tags)),
            )
            selected_tags = [
                self._commit_token_tags[index] for index in selected_indices
            ]
        else:
            selected_indices = tuple(
                range(
                    max(0, len(self._commit_token_tags) - count),
                    len(self._commit_token_tags),
                )
            )
            selected_tags = self._commit_token_tags[-count:]
        if older_video_stride == 1 or recent_full_blocks >= len(selected_tags):
            return pair
        cache_key = (
            self._block_index,
            selected_indices,
            recent_full_blocks,
            older_video_stride,
            pair.key.device,
        )
        indices = self._attention_index_cache.get(cache_key)
        if indices is None:
            dense_start = max(0, len(selected_tags) - recent_full_blocks)
            selected: list[int] = []
            offset = 0
            for block_index, tags in enumerate(selected_tags):
                if block_index >= dense_start:
                    selected.extend(range(offset, offset + len(tags)))
                else:
                    video_ordinal = 0
                    for local_index, tag in enumerate(tags):
                        if tag == AUDIO_TOKEN_TAG:
                            selected.append(offset + local_index)
                        elif tag == VIDEO_TOKEN_TAG:
                            if video_ordinal % older_video_stride == 0:
                                selected.append(offset + local_index)
                            video_ordinal += 1
                offset += len(tags)
            indices = torch.tensor(selected, dtype=torch.long, device=pair.key.device)
            self._attention_index_cache[cache_key] = indices
        return AVKV(
            key=pair.key.index_select(0, indices),
            value=pair.value.index_select(0, indices),
        )

    def begin_clean_commit(self, block_index: int) -> None:
        self._require_valid()
        if block_index != self._block_index:
            raise ValueError(
                f"clean block order mismatch: expected {self._block_index}, got {block_index}"
            )
        if self._staged is not None:
            raise RuntimeError("a clean commit is already active")
        self._staged = {}
        self._staged_commit_indices = None
        self._staged_commit_tags = None

    def stage(
        self,
        layer: str,
        key: Any,
        value: Any,
        token_tags: Any,
        commit_mask: Any,
        *,
        commit_indices: Any | None = None,
    ) -> None:
        import torch

        self._require_valid()
        if self._staged is None:
            raise RuntimeError("K/V may be staged only during a clean sigma=0 forward")
        if layer not in self.contract.layer_names or layer in self._staged:
            raise ValueError(f"invalid or duplicate clean K/V layer: {layer}")
        if (
            not torch.is_tensor(token_tags)
            or not torch.is_tensor(commit_mask)
            or token_tags.ndim != 1
            or commit_mask.ndim != 1
            or commit_mask.dtype != torch.bool
        ):
            raise TypeError("token_tags and commit_mask must be one-dimensional tensors")
        if (
            key.shape[0] != token_tags.shape[0]
            or value.shape[0] != token_tags.shape[0]
            or commit_mask.shape[0] != token_tags.shape[0]
        ):
            raise ValueError("token metadata and post-Ulysses K/V rows differ")
        if self._staged_commit_indices is None:
            if commit_indices is None:
                commit_indices = torch.nonzero(commit_mask, as_tuple=False).view(-1)
            if (
                not torch.is_tensor(commit_indices)
                or commit_indices.ndim != 1
                or commit_indices.dtype != torch.long
                or commit_indices.device != key.device
                or int(commit_indices.numel()) == 0
            ):
                raise ValueError("clean H3 block contains no valid audio/video rows")
            committed_tags = tuple(
                int(item)
                for item in token_tags.index_select(0, commit_indices)
                .detach()
                .cpu()
                .tolist()
            )
            _validate_committed_media_tags(committed_tags)
            self._staged_commit_indices = commit_indices.detach()
            self._staged_commit_tags = committed_tags
        indices = self._staged_commit_indices
        if indices is None:
            raise RuntimeError("clean H3 commit indices were not initialized")
        pair = AVKV(
            key=key.index_select(0, indices).detach(),
            value=value.index_select(0, indices).detach(),
        )
        _validate_av_pair(pair, self.contract, layer=layer)
        self._staged[layer] = pair

    def _validated_staged_commit(
        self,
    ) -> tuple[dict[str, AVKV], int, tuple[int, ...], Any]:
        """Validate the complete staged transaction before mutating history."""

        import torch

        self._require_valid()
        staged = self._staged
        if staged is None:
            raise RuntimeError("no clean commit is active")
        expected_layers = set(self.contract.layer_names)
        staged_layers = set(staged)
        missing = [
            name for name in self.contract.layer_names if name not in staged_layers
        ]
        if staged_layers != expected_layers:
            raise RuntimeError(f"clean commit is missing attention layers: {missing}")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1
            for count in self._commit_token_counts
        ):
            raise RuntimeError("H3 commit token accounting is malformed")
        if len(self._commit_token_tags) != len(self._commit_token_counts) or any(
            len(tags) != count
            for tags, count in zip(
                self._commit_token_tags,
                self._commit_token_counts,
                strict=True,
            )
        ):
            raise RuntimeError("H3 commit token tag accounting diverged")
        staged_tags = self._staged_commit_tags
        if staged_tags is None:
            raise RuntimeError("clean H3 commit tags were not initialized")

        current_layout = None
        for name in self.contract.layer_names:
            current = staged[name]
            if (
                not isinstance(current, AVKV)
                or not torch.is_tensor(current.key)
                or not torch.is_tensor(current.value)
                or current.key.ndim != 3
                or current.key.shape != current.value.shape
                or tuple(current.key.shape[1:])
                != (self.contract.local_heads, self.contract.head_dim)
                or current.key.dtype != current.value.dtype
                or current.key.device != current.value.device
            ):
                raise RuntimeError("per-layer H3 staged K/V layout diverged")
            layout = (current.key.shape, current.key.dtype, current.key.device)
            if current_layout is None:
                current_layout = layout
            elif layout != current_layout:
                raise RuntimeError("per-layer H3 staged K/V layout diverged")

        assert current_layout is not None
        current_tokens = int(current_layout[0][0])
        if current_tokens <= 0 or len(staged_tags) != current_tokens:
            raise RuntimeError("clean H3 commit token metadata diverged")
        previous_tokens = sum(self._commit_token_counts)
        if self.segmented_fp8_history:
            history_layers = set(self._history_blocks)
            if history_layers not in (set(), expected_layers):
                raise RuntimeError("segmented H3 history does not cover every layer")
            if bool(history_layers) != bool(self._commit_token_counts):
                raise RuntimeError("H3 history and commit token accounting diverged")
            for name in self.contract.layer_names:
                blocks = self._history_blocks.get(name, ())
                if len(blocks) != len(self._commit_token_counts) or sum(
                    int(pair.key.shape[0]) for pair in blocks
                ) != previous_tokens:
                    raise RuntimeError("segmented H3 history boundaries diverged")
        else:
            if self._history_blocks:
                raise RuntimeError("dense H3 cache unexpectedly contains segmented history")
            history_layers = set(self._history)
            if history_layers not in (set(), expected_layers):
                raise RuntimeError("H3 history does not cover every attention layer")
            if bool(history_layers) != bool(self._commit_token_counts):
                raise RuntimeError("H3 history and commit token accounting diverged")
            previous_layout = None
            for name in self.contract.layer_names:
                previous = self._history.get(name)
                if previous is None:
                    continue
                current = staged[name]
                if (
                    not isinstance(previous, AVKV)
                    or not torch.is_tensor(previous.key)
                    or not torch.is_tensor(previous.value)
                    or previous.key.ndim != 3
                    or previous.key.shape != previous.value.shape
                    or previous.key.shape[1:] != current.key.shape[1:]
                    or previous.key.dtype != current.key.dtype
                    or previous.key.device != current.key.device
                ):
                    raise RuntimeError("per-layer H3 history K/V layout diverged")
                layout = (
                    previous.key.shape,
                    previous.key.dtype,
                    previous.key.device,
                )
                if previous_layout is None:
                    previous_layout = layout
                elif layout != previous_layout:
                    raise RuntimeError("per-layer H3 history K/V layout diverged")
            if (
                (int(previous_layout[0][0]) if previous_layout else 0)
                != previous_tokens
            ):
                raise RuntimeError("H3 history and commit token accounting diverged")
        return staged, current_tokens, staged_tags, current_layout[2]

    def _invalidate_failed_commit(self) -> None:
        if self._staged is not None:
            self._staged.clear()
        self._staged = None
        self._staged_commit_indices = None
        self._staged_commit_tags = None
        self._history.clear()
        self._history_blocks.clear()
        self._block_index = 0
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._attention_index_cache.clear()
        self._invalid_after_failed_commit = True

    def commit(self, *, retain_recent_commit_count: int | None = None) -> None:
        """Publish staged K/V, optionally keeping only the newest commits."""

        import torch

        if retain_recent_commit_count is not None:
            self.commit_bounded(retain_recent_commit_count)
            return
        staged, current_tokens, _, current_device = self._validated_staged_commit()
        reclaim_cuda = current_device.type == "cuda"
        try:
            if self.segmented_fp8_history:
                for name in self.contract.layer_names:
                    current = self._quantize_history_block(staged.pop(name))
                    self._history_blocks.setdefault(name, []).append(current)
                self._history.clear()
            else:
                if reclaim_cuda:
                    torch.cuda.empty_cache()
                layer_count = len(self.contract.layer_names)
                for layer_number, name in enumerate(
                    self.contract.layer_names, start=1
                ):
                    current = staged.pop(name)
                    previous = self._history.pop(name, None)
                    if previous is None:
                        updated = current
                    else:
                        updated = AVKV(
                            key=torch.cat((previous.key, current.key), dim=0),
                            value=torch.cat((previous.value, current.value), dim=0),
                        )
                    self._history[name] = updated
                    current = previous = updated = None
                    if reclaim_cuda and (
                        layer_number % 8 == 0 or layer_number == layer_count
                    ):
                        torch.cuda.empty_cache()
            self._finish_commit(current_tokens)
        except BaseException:
            self._invalidate_failed_commit()
            raise

    def _finish_commit(self, current_tokens: int) -> None:
        staged_tags = self._staged_commit_tags
        if staged_tags is None:
            raise RuntimeError("clean H3 commit tags were not initialized")
        self._staged = None
        self._staged_commit_indices = None
        self._staged_commit_tags = None
        self._block_index += 1
        self._commit_token_counts.append(current_tokens)
        self._commit_token_tags.append(staged_tags)
        self._attention_index_cache.clear()

    @staticmethod
    def _quantize_history_block(pair: AVKV) -> AVKV:
        from digitalhuman_h3.inference.fused_kernels import (
            quantize_fp8_e4m3_per_head,
        )

        fp8_key, key_descale = quantize_fp8_e4m3_per_head(
            pair.key.unsqueeze(0)
        )
        fp8_value, value_descale = quantize_fp8_e4m3_per_head(
            pair.value.unsqueeze(0)
        )
        return AVKV(
            key=pair.key,
            value=pair.value,
            fp8_key=fp8_key.squeeze(0),
            fp8_value=fp8_value.squeeze(0),
            fp8_key_descale=key_descale,
            fp8_value_descale=value_descale,
            key_summary=pair.key.float().mean(dim=0),
        )

    def commit_bounded(self, count: int, *, advance_serial: bool = True) -> None:
        """Commit while keeping only the newest ``count`` block boundaries.

        HiAR-PP uses a fixed generated-history window.  Updating each layer in
        place avoids the old peak where a complete old window and a complete
        newly concatenated window stayed live together until all 50 layers had
        finished concatenating.
        """

        import torch

        self._require_valid()
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("bounded clean commit count must be positive")
        staged, current_tokens, staged_tags, current_device = (
            self._validated_staged_commit()
        )
        retained_counts = (
            self._commit_token_counts[-(count - 1) :] if count > 1 else []
        )
        retained_tags = self._commit_token_tags[-(count - 1) :] if count > 1 else []
        retained_tokens = sum(retained_counts)
        reclaim_cuda = current_device.type == "cuda"
        try:
            if self.segmented_fp8_history:
                keep = count - 1
                for name in self.contract.layer_names:
                    current = self._quantize_history_block(staged.pop(name))
                    previous = self._history_blocks.get(name, [])
                    retained = previous[-keep:] if keep else []
                    self._history_blocks[name] = [*retained, current]
                self._history.clear()
            else:
                if reclaim_cuda:
                    torch.cuda.empty_cache()
                layer_count = len(self.contract.layer_names)
                for layer_number, name in enumerate(
                    self.contract.layer_names, start=1
                ):
                    current = staged.pop(name)
                    previous = self._history.pop(name, None)
                    if previous is None or retained_tokens == 0:
                        updated = current
                    else:
                        updated = AVKV(
                            key=torch.cat(
                                (previous.key[-retained_tokens:], current.key), dim=0
                            ),
                            value=torch.cat(
                                (previous.value[-retained_tokens:], current.value),
                                dim=0,
                            ),
                        )
                    self._history[name] = updated
                    current = previous = updated = None
                    if reclaim_cuda and (
                        layer_number % 8 == 0 or layer_number == layer_count
                    ):
                        torch.cuda.empty_cache()
        except BaseException:
            self._invalidate_failed_commit()
            raise

        self._staged = None
        self._staged_commit_indices = None
        self._staged_commit_tags = None
        if advance_serial:
            self._block_index += 1
        self._commit_token_counts = [*retained_counts, current_tokens]
        self._commit_token_tags = [*retained_tags, staged_tags]
        self._attention_index_cache.clear()

    def retain_recent_commits(self, count: int) -> None:
        """Bound cross-request memory to the newest clean phase commits."""

        self._require_valid()
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("retained clean commit count must be a positive integer")
        if self._staged is not None:
            raise RuntimeError("cannot evict H3 history during a clean commit")
        if len(self._commit_token_counts) <= count:
            return
        retained_counts = self._commit_token_counts[-count:]
        retained_tokens = sum(retained_counts)
        if self.segmented_fp8_history:
            self._history_blocks = {
                name: blocks[-count:]
                for name, blocks in self._history_blocks.items()
            }
            self._history.clear()
        else:
            # Evict one layer at a time so old and retained forms of all 50
            # layers are never resident together.
            try:
                for name in self.contract.layer_names:
                    pair = self._history.pop(name)
                    self._history[name] = AVKV(
                        key=pair.key[-retained_tokens:].clone(),
                        value=pair.value[-retained_tokens:].clone(),
                    )
            except BaseException:
                self._invalidate_failed_commit()
                raise
        self._commit_token_counts = retained_counts
        self._commit_token_tags = self._commit_token_tags[-count:]
        self._attention_index_cache.clear()

    def retain_sink_and_recent_commits(
        self,
        recent_count: int,
        *,
        sink_video_only: bool = True,
    ) -> None:
        """Keep block-zero sink plus the newest complete AV blocks."""

        self._require_valid()
        if (
            isinstance(recent_count, bool)
            or not isinstance(recent_count, int)
            or recent_count < 0
        ):
            raise ValueError("recent clean commit count must be non-negative")
        if self._staged is not None:
            raise RuntimeError("cannot evict H3 history during a clean commit")
        self._sink_video_only = bool(sink_video_only)
        block_count = len(self._commit_token_counts)
        if block_count == 0:
            return
        recent_start = max(0, block_count - recent_count)
        selections = [
            (0, bool(sink_video_only and recent_start > 0)),
            *(
                (index, False)
                for index in range(max(1, recent_start), block_count)
            ),
        ]
        if selections == [(index, False) for index in range(block_count)]:
            return
        self._retain_commit_rows(selections)

    def drop_audio_history(self) -> int:
        """Remove retained audio KV while preserving every retained video row."""

        if self._staged is not None:
            raise RuntimeError("cannot reset H3 audio history during a clean commit")
        removed = self.history_audio_tokens
        if removed == 0:
            return 0
        self._retain_commit_rows(
            [(index, True) for index in range(len(self._commit_token_counts))]
        )
        return removed

    @staticmethod
    def _select_pair_rows(pair: AVKV, indices: Any) -> AVKV:
        key = pair.key.index_select(0, indices)
        value = pair.value.index_select(0, indices)
        return AVKV(
            key=key,
            value=value,
            fp8_key=(
                None
                if pair.fp8_key is None
                else pair.fp8_key.index_select(0, indices)
            ),
            fp8_value=(
                None
                if pair.fp8_value is None
                else pair.fp8_value.index_select(0, indices)
            ),
            fp8_key_descale=pair.fp8_key_descale,
            fp8_value_descale=pair.fp8_value_descale,
            key_summary=(
                None
                if pair.key_summary is None
                else key.float().mean(dim=0)
            ),
        )

    def _retain_commit_rows(
        self,
        selections: list[tuple[int, bool]],
    ) -> None:
        """Retain selected commits, optionally filtering each to video rows."""

        import torch

        original_counts = self._commit_token_counts
        original_tags = self._commit_token_tags
        retained_tags = [
            (
                tuple(tag for tag in original_tags[index] if tag == VIDEO_TOKEN_TAG)
                if video_only
                else original_tags[index]
            )
            for index, video_only in selections
        ]
        retained_counts = [len(tags) for tags in retained_tags]
        if self.segmented_fp8_history:
            video_rows_by_block_and_device: dict[tuple[int, Any], Any] = {}
            for name in self.contract.layer_names:
                blocks = self._history_blocks[name]
                retained_blocks: list[AVKV] = []
                for index, video_only in selections:
                    pair = blocks[index]
                    if video_only:
                        row_key = (index, pair.key.device)
                        rows = video_rows_by_block_and_device.get(row_key)
                        if rows is None:
                            rows = torch.tensor(
                                [
                                    row
                                    for row, tag in enumerate(original_tags[index])
                                    if tag == VIDEO_TOKEN_TAG
                                ],
                                dtype=torch.long,
                                device=pair.key.device,
                            )
                            video_rows_by_block_and_device[row_key] = rows
                        pair = self._select_pair_rows(pair, rows)
                    retained_blocks.append(pair)
                self._history_blocks[name] = retained_blocks
            self._history.clear()
        else:
            offsets: list[int] = []
            cursor = 0
            for count in original_counts:
                offsets.append(cursor)
                cursor += count
            selected_rows = [
                offsets[index] + row
                for index, video_only in selections
                for row, tag in enumerate(original_tags[index])
                if not video_only or tag == VIDEO_TOKEN_TAG
            ]
            indices_by_device: dict[Any, Any] = {}
            for name in self.contract.layer_names:
                pair = self._history.pop(name)
                indices = indices_by_device.get(pair.key.device)
                if indices is None:
                    indices = torch.tensor(
                        selected_rows,
                        dtype=torch.long,
                        device=pair.key.device,
                    )
                    indices_by_device[pair.key.device] = indices
                self._history[name] = self._select_pair_rows(pair, indices)
        self._commit_token_counts = retained_counts
        self._commit_token_tags = retained_tags
        self._attention_index_cache.clear()

    def rollback(self) -> None:
        self._staged = None
        self._staged_commit_indices = None
        self._staged_commit_tags = None

    def advance_serial_without_history_commit(self, block_index: int) -> None:
        """Close one generated block after rebuilding its next-window prefix."""

        if self._staged is not None:
            raise RuntimeError("cannot advance H3 serial during a clean commit")
        if block_index != self._block_index:
            raise ValueError(
                f"clean block order mismatch: expected {self._block_index}, "
                f"got {block_index}"
            )
        self._block_index += 1

    def discard_retained_history_preserve_serial(self) -> int:
        """Drop retained AV-KV without rewinding the logical commit serial."""

        if self._staged is not None:
            raise RuntimeError("cannot discard H3 history during a clean commit")
        removed = self.history_tokens
        logical_serial = self._block_index
        self._history.clear()
        self._history_blocks.clear()
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._attention_index_cache.clear()
        if self._block_index != logical_serial:
            raise RuntimeError("H3 boundary reset rewound the logical commit serial")
        return removed

    def clear(self) -> None:
        if self._staged is not None:
            raise RuntimeError("cannot clear H3 history during a clean commit")
        self._history.clear()
        self._history_blocks.clear()
        self._block_index = 0
        self._commit_token_counts.clear()
        self._commit_token_tags.clear()
        self._attention_index_cache.clear()
        self._invalid_after_failed_commit = False
        self._sink_video_only = False


__all__ = [
    "AUDIO_TOKEN_TAG",
    "AVKV",
    "AV_TOKEN_TAGS",
    "CleanAVKVCache",
    "KVContract",
    "MAIN_LAYER_NAMES",
    "TemporaryAVPrefix",
    "TemporaryVideoPrefix",
    "VIDEO_TOKEN_TAG",
    "VideoKV",
    "kv_contract_from_model",
]
