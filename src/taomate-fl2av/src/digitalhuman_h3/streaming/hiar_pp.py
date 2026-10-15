from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterator, Sequence


HIAR_STAGE_COUNT = 3


@dataclass(frozen=True)
class HiARFullPrefixMemoryUpdate:
    """One canonical high-clean AV latent update awaiting lagged visibility."""

    canonical_latents: Any
    block_seconds: float

    @property
    def canonical_rows(self) -> Any:
        """Compatibility alias; the value is the complete AV pair, not video."""

        return self.canonical_latents


def hiar_full_prefix_update_seconds(plan: Any, block_index: int) -> float | None:
    """Return the EMA duration when a block closes ten high-res latent units.

    The formal two-group layout contributes one update per block.  The
    one-group blockmix layout contributes one update after each adjacent pair,
    matching the Stage3 training contract exactly.
    """

    try:
        phase = plan.phases[block_index]
    except (AttributeError, IndexError) as exc:
        raise ValueError(f"unknown HiAR full-prefix block {block_index}") from exc
    if int(phase.group_count) == 2:
        return float(phase.duration)
    if (
        int(phase.group_count) == 1
        and block_index > 0
        and block_index % 2 == 1
        and int(plan.phases[block_index - 1].group_count) == 1
    ):
        return float(plan.phases[block_index - 1].duration + phase.duration)
    return None


class HiARDelayedFullPrefixMemory:
    """Canonical high-clean EMA whose updates become visible after a block lag."""

    def __init__(self, *, half_life_seconds: float, lag_blocks: int) -> None:
        from digitalhuman_h3.streaming.full_prefix import FullPrefixAVEMA

        if (
            isinstance(lag_blocks, bool)
            or not isinstance(lag_blocks, int)
            or lag_blocks < 0
        ):
            raise ValueError("HiAR full-prefix lag must be a non-negative integer")
        self.lag_blocks = lag_blocks
        self._memory = FullPrefixAVEMA(half_life_seconds=half_life_seconds)
        self._pending: dict[int, HiARFullPrefixMemoryUpdate] = {}
        self._visible_source_block_index: int | None = None

    @property
    def initialized(self) -> bool:
        return self._memory.initialized

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def visible_source_block_index(self) -> int | None:
        return self._visible_source_block_index

    def enqueue(
        self,
        source_block_index: int,
        canonical_latents: Any,
        *,
        block_seconds: float,
    ) -> None:
        from digitalhuman_h3.streaming.full_prefix import FullPrefixAVLatents

        if (
            isinstance(source_block_index, bool)
            or not isinstance(source_block_index, int)
            or source_block_index < 0
        ):
            raise ValueError("HiAR full-prefix source block must be non-negative")
        if source_block_index in self._pending or (
            self._visible_source_block_index is not None
            and source_block_index <= self._visible_source_block_index
        ):
            raise ValueError("HiAR full-prefix source block was already enqueued")
        if not isinstance(canonical_latents, FullPrefixAVLatents):
            raise TypeError("HiAR full-prefix update must contain video and audio latents")
        self._pending[source_block_index] = HiARFullPrefixMemoryUpdate(
            canonical_latents=canonical_latents,
            block_seconds=float(block_seconds),
        )

    def advance(self, block_index: int) -> tuple[int, ...]:
        """Expose all high-clean updates old enough for ``block_index``."""

        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise TypeError("HiAR full-prefix block index must be an integer")
        cutoff = block_index - self.lag_blocks
        due = tuple(sorted(source for source in self._pending if source <= cutoff))
        for source in due:
            update = self._pending.pop(source)
            self._memory.update(
                update.canonical_latents,
                block_seconds=update.block_seconds,
            )
            self._visible_source_block_index = source
        return due

    def view(self) -> Any | None:
        return self._memory.view()

    @property
    def half_life_seconds(self) -> float:
        return float(self._memory.half_life_seconds)

    def reset(self) -> None:
        self._memory.reset()
        self._pending.clear()
        self._visible_source_block_index = None


class HiARProgressAVKVCache:
    """Bounded AV-KV history captured at one HiAR denoising stage.

    Stage 0 and stage 1 intentionally commit intermediate noisy states.  The
    final stage commits the last distilled state.  Storage and validation are
    delegated to the native AV cache, while this wrapper exposes the HiAR
    progress semantics and the number of retained block boundaries.
    """

    def __init__(
        self,
        *,
        contract: Any,
        max_history_blocks: int,
        segmented_fp8_history: bool = False,
        sink_recent_blocks: int | None = None,
        sink_video_only: bool = True,
    ) -> None:
        from digitalhuman_h3.streaming.cache import CleanAVKVCache

        if (
            isinstance(max_history_blocks, bool)
            or not isinstance(max_history_blocks, int)
            or max_history_blocks < 1
        ):
            raise ValueError("HiAR max history blocks must be a positive integer")
        self._storage = CleanAVKVCache(
            contract,
            segmented_fp8_history=segmented_fp8_history,
        )
        self.max_history_blocks = max_history_blocks
        if sink_recent_blocks is not None and sink_recent_blocks < 0:
            raise ValueError("HiAR sink recent blocks must be non-negative")
        self.sink_recent_blocks = sink_recent_blocks
        self.sink_video_only = bool(sink_video_only)
        self._retained_blocks = 0

    @property
    def contract(self) -> Any:
        return self._storage.contract

    @property
    def committed_blocks(self) -> int:
        return self._storage.committed_blocks

    @property
    def history_tokens(self) -> int:
        return self._storage.history_tokens

    @property
    def history_audio_tokens(self) -> int:
        return self._storage.history_audio_tokens

    @property
    def history_video_tokens(self) -> int:
        return self._storage.history_video_tokens

    @property
    def clean_commit_active(self) -> bool:
        return self._storage.clean_commit_active

    @property
    def retained_blocks(self) -> int:
        return self._retained_blocks

    def history(self, layer: str) -> Any:
        return self._storage.history(layer)

    def history_blocks(self, layer: str) -> Any:
        return self._storage.history_blocks(layer)

    def history_for_fused_current(self, layer: str) -> Any:
        """Select persistent history before appending fused pending AV-KV.

        E4 keeps block zero plus ``recent-1`` already committed chunks; the
        packed previous document supplies the final recent chunk.
        """

        if self.sink_recent_blocks is not None:
            return self._storage.sink_and_recent_history(
                layer,
                max(0, self.sink_recent_blocks - 1),
            )
        persistent_count = self.max_history_blocks - 1
        return self._storage.recent_history(layer, persistent_count)

    def history_for_attention(
        self,
        layer: str,
        *,
        fused_current: bool,
        recent_full_blocks: int,
        older_video_stride: int,
    ) -> Any:
        preserve_sink = self.sink_recent_blocks is not None
        if preserve_sink:
            recent_count = max(
                0,
                self.sink_recent_blocks - (1 if fused_current else 0),
            )
            persistent_count = min(
                self.retained_blocks,
                1 + recent_count,
            )
        else:
            persistent_count = min(
                self.retained_blocks,
                self.max_history_blocks - (1 if fused_current else 0),
            )
        persistent_recent_full = max(
            0,
            recent_full_blocks - (1 if fused_current else 0),
        )
        return self._storage.attention_history(
            layer,
            count=persistent_count,
            recent_full_blocks=persistent_recent_full,
            older_video_stride=older_video_stride,
            preserve_sink=preserve_sink,
        )

    def sol_history_modality_parts(
        self,
        layer: str,
        *,
        fused_current: bool,
    ) -> tuple[tuple[tuple[Any, Any], ...], tuple[tuple[Any, Any], ...]]:
        """Select the same bounded history, split into audio and video parts."""

        retained = self._storage.retained_blocks
        if retained == 0:
            return (), ()
        if self.sink_recent_blocks is not None:
            recent_count = max(
                0,
                self.sink_recent_blocks - (1 if fused_current else 0),
            )
            recent_start = max(1, retained - recent_count)
            block_indices = (0, *range(recent_start, retained))
        else:
            count = min(
                retained,
                self.max_history_blocks - (1 if fused_current else 0),
            )
            block_indices = tuple(range(retained - count, retained))
        return self._storage.history_modality_parts(layer, block_indices)

    def begin_progress_commit(self, block_index: int) -> None:
        self._storage.begin_clean_commit(block_index)

    def stage(self, *args: Any, **kwargs: Any) -> None:
        self._storage.stage(*args, **kwargs)

    def commit_progress(self) -> None:
        if self.sink_recent_blocks is None:
            self._storage.commit_bounded(self.max_history_blocks)
        else:
            self._storage.commit()
            self._storage.retain_sink_and_recent_commits(
                self.sink_recent_blocks,
                sink_video_only=self.sink_video_only,
            )
        self._retained_blocks = self._storage.retained_blocks

    def begin_window_prefix_commit(self) -> None:
        """Start one clean prefix block without advancing generation serial."""

        self._storage.begin_clean_commit(self.committed_blocks)

    def commit_window_prefix(self) -> None:
        self._storage.commit_bounded(
            self.max_history_blocks,
            advance_serial=False,
        )
        self._retained_blocks = self._storage.retained_blocks

    def advance_serial_without_history_commit(self, block_index: int) -> None:
        self._storage.advance_serial_without_history_commit(block_index)

    def rollback(self) -> None:
        self._storage.rollback()

    def discard_retained_history_preserve_serial(self) -> int:
        removed = self._storage.discard_retained_history_preserve_serial()
        self._retained_blocks = 0
        return removed

    def drop_audio_history(self) -> int:
        return self._storage.drop_audio_history()

    def retain_sink_and_recent_commits(
        self,
        recent_count: int,
        *,
        sink_video_only: bool | None = None,
    ) -> None:
        self._storage.retain_sink_and_recent_commits(
            recent_count,
            sink_video_only=(
                self.sink_video_only
                if sink_video_only is None
                else bool(sink_video_only)
            ),
        )
        self._retained_blocks = self._storage.retained_blocks

    def clear(self) -> None:
        self._storage.clear()
        self._retained_blocks = 0


class HiARProgressSession:
    """Native H3 attention session for one stage of HiAR-PP.

    Every noisy forward reads up to ``max_history_blocks`` outputs previously
    produced by the same denoising stage.  The stage-output AV-KV is captured
    either by a separate no-update forward or by the next block's fused
    previous-commit/current-denoise forward.  Thus early stages retain their
    own progress state rather than incorrectly substituting final clean KV.
    """

    def __init__(
        self,
        plan: Any,
        *,
        model: Any,
        max_history_blocks: int,
        attention_backend: str = "sdpa",
        history_kv_recent_full_blocks: int | None = None,
        history_kv_older_video_stride: int = 1,
        history_kv_dynamic_density: float = 0.5,
        current_video_kv_stride: int = 1,
        dense_current_document_when_empty: bool = False,
        sol_attn_tau: float = 1.0,
        sol_attn_thresh_type: str = "diag",
        sol_attn_dense_layers: int = 2,
        sink_recent_blocks: int | None = None,
        sink_video_only: bool = True,
    ) -> None:
        from digitalhuman_h3.streaming.attention_hook import H3StreamingAttentionHook
        from digitalhuman_h3.streaming.cache import kv_contract_from_model
        from digitalhuman_h3.streaming.capability import require_streaming_capability

        self.plan = plan
        self.cache = HiARProgressAVKVCache(
            contract=kv_contract_from_model(model),
            max_history_blocks=max_history_blocks,
            segmented_fp8_history=(
                attention_backend in {
                    "flash_attn3_fp8_segmented",
                    "flash_attn3_fp8_segmented_topk",
                }
            ),
            sink_recent_blocks=sink_recent_blocks,
            sink_video_only=sink_video_only,
        )
        self.hook = H3StreamingAttentionHook(
            self.cache,
            attention_backend=attention_backend,
            history_kv_recent_full_blocks=(
                max_history_blocks
                if history_kv_recent_full_blocks is None
                else history_kv_recent_full_blocks
            ),
            history_kv_older_video_stride=history_kv_older_video_stride,
            history_kv_dynamic_density=history_kv_dynamic_density,
            current_video_kv_stride=current_video_kv_stride,
            dense_current_document_when_empty=dense_current_document_when_empty,
            sol_attn_tau=sol_attn_tau,
            sol_attn_thresh_type=sol_attn_thresh_type,
            sol_attn_dense_layers=sol_attn_dense_layers,
        )
        self.capability = require_streaming_capability(model=model)
        self._installed = False
        self._denoised_blocks: set[int] = set()

    def __enter__(self) -> "HiARProgressSession":
        from digitalhuman_h3.streaming.capability import set_official_attention_hook

        if self._installed:
            raise RuntimeError("HiAR progress session is already installed")
        set_official_attention_hook(self.hook)
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        from digitalhuman_h3.streaming.capability import set_official_attention_hook

        if self.cache.clean_commit_active:
            self.cache.rollback()
        self.hook.deactivate()
        if self._installed:
            set_official_attention_hook(None)
            self._installed = False

    def _phase(self, block_index: int) -> Any:
        if not self._installed:
            raise RuntimeError("use HiARProgressSession as a context manager")
        if block_index != self.cache.committed_blocks:
            raise ValueError("HiAR blocks must be processed and committed in order")
        try:
            return self.plan.phases[block_index]
        except IndexError as exc:
            raise ValueError(f"unknown HiAR block {block_index}") from exc

    @contextmanager
    def denoise_step(
        self,
        block_index: int,
        *,
        full_prefix: Any | None = None,
    ) -> Iterator[Any]:
        from digitalhuman_h3.streaming.attention_hook import HookMode

        phase = self._phase(block_index)
        self.hook.activate(
            mode=HookMode.NOISY,
            block_index=block_index,
            full_prefix=full_prefix,
        )
        try:
            yield phase
        finally:
            self.hook.deactivate()
        self._denoised_blocks.add(block_index)

    @contextmanager
    def denoise_and_commit_input(
        self,
        block_index: int,
        *,
        full_prefix: Any | None = None,
    ) -> Iterator[Any]:
        """Denoise once and retain that same forward's stage-input AV-KV."""

        from digitalhuman_h3.streaming.attention_hook import HookMode

        phase = self._phase(block_index)
        self.cache.begin_progress_commit(block_index)
        self.hook.activate(
            mode=HookMode.PROGRESS_COMMIT,
            block_index=block_index,
            full_prefix=full_prefix,
        )
        try:
            yield phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit_progress()
            except BaseException:
                self.cache.rollback()
                raise
        finally:
            self.hook.deactivate()

    @contextmanager
    def progress_commit(
        self,
        block_index: int,
        *,
        full_prefix: Any | None = None,
    ) -> Iterator[Any]:
        from digitalhuman_h3.streaming.attention_hook import HookMode

        phase = self._phase(block_index)
        if block_index not in self._denoised_blocks:
            raise RuntimeError("HiAR progress commit requires a denoise forward")
        self.cache.begin_progress_commit(block_index)
        self.hook.activate(
            mode=HookMode.PROGRESS_COMMIT,
            block_index=block_index,
            full_prefix=full_prefix,
        )
        try:
            yield phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit_progress()
            except BaseException:
                self.cache.rollback()
                raise
            self._denoised_blocks.remove(block_index)
        finally:
            self.hook.deactivate()

    @contextmanager
    def replace_history_with_clean_window(
        self,
        block_index: int,
    ) -> Iterator[Any]:
        """Replace stage-local progress KV with selected clean window chunks."""

        phase = self._phase(block_index)
        if block_index not in self._denoised_blocks:
            raise RuntimeError(
                "clean-window replacement requires the window's final denoise"
            )
        self.cache.discard_retained_history_preserve_serial()
        try:
            yield phase
            if self.cache.retained_blocks <= 0:
                raise RuntimeError("clean-window replacement committed no prefix")
            self.cache.advance_serial_without_history_commit(block_index)
            self._denoised_blocks.remove(block_index)
        except BaseException:
            if self.cache.clean_commit_active:
                self.cache.rollback()
            raise

    @contextmanager
    def clean_window_prefix_commit(self, block_index: int) -> Iterator[None]:
        """Commit one selected clean chunk while rebuilding a window prefix."""

        from digitalhuman_h3.streaming.attention_hook import HookMode

        if block_index not in self._denoised_blocks:
            raise RuntimeError("clean-window prefix commit is outside a rebuild")
        self.cache.begin_window_prefix_commit()
        self.hook.activate(
            mode=HookMode.PROGRESS_COMMIT,
            block_index=block_index,
            full_prefix=None,
        )
        try:
            yield
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit_window_prefix()
            except BaseException:
                self.cache.rollback()
                raise
        finally:
            self.hook.deactivate()

    @contextmanager
    def fused_progress_denoise(
        self,
        previous_block_index: int,
        current_block_index: int,
        *,
        previous_full_prefix: Any | None = None,
        current_full_prefix: Any | None = None,
        previous_video_only: bool = False,
    ) -> Iterator[tuple[Any, Any]]:
        """Commit the previous stage output while denoising the next block.

        The fused packed forward contains two independent prompt documents.
        The current document reads the previous document's AV-KV, while the
        previous document cannot read current rows.  This is the H3 equivalent
        of LTX HiAR-PP's fused cache-update/next-block forward.  Separate
        previous/current full-prefix objects preserve lag-zero ordering across
        the fused boundary and across prompt changes.  A caller may explicitly
        mark a request boundary as ``previous_video_only`` so the new request
        can continue visual motion without attending to the old request's
        audio.  The fused commit itself remains AV; callers that reset audio at
        request boundaries must drop persisted audio immediately after commit.
        """

        from digitalhuman_h3.streaming.attention_hook import HookMode

        previous_phase = self._phase(previous_block_index)
        if current_block_index != previous_block_index + 1:
            raise ValueError("fused HiAR blocks must be adjacent")
        try:
            current_phase = self.plan.phases[current_block_index]
        except IndexError as exc:
            raise ValueError(f"unknown HiAR block {current_block_index}") from exc
        if previous_block_index not in self._denoised_blocks:
            raise RuntimeError("fused HiAR commit requires the previous denoise output")
        self.cache.begin_progress_commit(previous_block_index)
        self.hook.activate(
            mode=HookMode.FUSED_PROGRESS_DENOISE,
            block_index=previous_block_index,
            full_prefix=current_full_prefix,
            fused_previous_full_prefix=previous_full_prefix,
            fused_previous_video_only=previous_video_only,
        )
        try:
            yield previous_phase, current_phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit_progress()
            except BaseException:
                self.cache.rollback()
                raise
            self._denoised_blocks.remove(previous_block_index)
            self._denoised_blocks.add(current_block_index)
        finally:
            self.hook.deactivate()


@dataclass(frozen=True)
class HiARTopology:
    """Rank layout for one model replica per distilled denoising transition."""

    world_size: int
    rank: int
    tp_size: int = 2
    stage_count: int = HIAR_STAGE_COUNT

    def __post_init__(self) -> None:
        if self.stage_count != HIAR_STAGE_COUNT:
            raise ValueError("Stage1 HiAR requires exactly three pipeline stages")
        if self.tp_size not in {1, 2}:
            raise ValueError("MiniMax H3 HiAR supports TP=1 or TP=2")
        if self.world_size <= 0 or self.world_size % self.stage_count:
            raise ValueError("HiAR world size must be divisible by three stages")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("HiAR rank is outside the global world")
        if self.ranks_per_stage % self.tp_size:
            raise ValueError("HiAR ranks per stage must be divisible by TP size")
        if self.ulysses_degree not in {1, 2, 4, 8}:
            raise ValueError("HiAR supports Ulysses degree 1, 2, 4, or 8")

    @property
    def ranks_per_stage(self) -> int:
        return self.world_size // self.stage_count

    @property
    def ulysses_degree(self) -> int:
        return self.ranks_per_stage // self.tp_size

    @property
    def stage_index(self) -> int:
        return self.rank // self.ranks_per_stage

    @property
    def stage_local_rank(self) -> int:
        return self.rank % self.ranks_per_stage

    @property
    def tp_rank(self) -> int:
        return self.stage_local_rank % self.tp_size

    @property
    def ulysses_rank(self) -> int:
        return self.stage_local_rank // self.tp_size

    @property
    def stage_leader(self) -> int:
        return self.stage_index * self.ranks_per_stage

    @property
    def output_leader(self) -> int:
        return (self.stage_count - 1) * self.ranks_per_stage

    @property
    def checkpoint_source_rank(self) -> int:
        return self.tp_rank

    def stage_ranks(self, stage: int) -> tuple[int, ...]:
        if not 0 <= stage < self.stage_count:
            raise ValueError("HiAR stage is outside the pipeline")
        start = stage * self.ranks_per_stage
        return tuple(range(start, start + self.ranks_per_stage))

    def tp_ranks(self, stage: int, ulysses_rank: int) -> tuple[int, ...]:
        start = stage * self.ranks_per_stage + ulysses_rank * self.tp_size
        return tuple(range(start, start + self.tp_size))

    def ulysses_ranks(self, stage: int, tp_rank: int) -> tuple[int, ...]:
        start = stage * self.ranks_per_stage
        return tuple(
            start + ulysses_rank * self.tp_size + tp_rank
            for ulysses_rank in range(self.ulysses_degree)
        )

    def checkpoint_replica_ranks(self, tp_rank: int) -> tuple[int, ...]:
        return tuple(
            stage * self.ranks_per_stage + ulysses_rank * self.tp_size + tp_rank
            for stage in range(self.stage_count)
            for ulysses_rank in range(self.ulysses_degree)
        )

    def corresponding_stage_rank(self, stage: int) -> int:
        """Return the rank with the same TP/Ulysses lane in another stage."""

        if not 0 <= stage < self.stage_count:
            raise ValueError("HiAR stage is outside the pipeline")
        return stage * self.ranks_per_stage + self.stage_local_rank

    @property
    def full_prefix_source_peer(self) -> int:
        return self.corresponding_stage_rank(self.stage_count - 1)

    @property
    def final_clean_source_peer(self) -> int:
        """Final-stage rank in this TP/Ulysses lane."""

        return self.corresponding_stage_rank(self.stage_count - 1)

    @property
    def full_prefix_consumer_peers(self) -> tuple[int, ...]:
        if self.stage_index != self.stage_count - 1:
            return ()
        return tuple(
            self.corresponding_stage_rank(stage)
            for stage in range(self.stage_count - 1)
        )

    @property
    def previous_peer(self) -> int | None:
        if self.stage_index == 0:
            return None
        return self.rank - self.ranks_per_stage

    @property
    def next_peer(self) -> int | None:
        if self.stage_index == self.stage_count - 1:
            return None
        return self.rank + self.ranks_per_stage


@dataclass(frozen=True)
class HiARProcessGroups:
    stage_group: Any
    checkpoint_replica_group: Any
    full_prefix_group: Any
    fp8_scale_sync_group: Any
    clean_feedback_group: Any
    parallel_context: Any


def build_hiar_process_groups(torch: Any, topology: HiARTopology) -> HiARProcessGroups:
    """Create every NCCL group in one deterministic global order."""

    from digitalhuman_h3.distributed.parallel import ParallelContext

    distributed = torch.distributed
    stage_groups: list[Any] = []
    for stage in range(topology.stage_count):
        stage_groups.append(
            distributed.new_group(
                ranks=list(topology.stage_ranks(stage)), backend="nccl"
            )
        )

    tp_groups: dict[tuple[int, int], Any] = {}
    for stage in range(topology.stage_count):
        for ulysses_rank in range(topology.ulysses_degree):
            tp_groups[(stage, ulysses_rank)] = distributed.new_group(
                ranks=list(topology.tp_ranks(stage, ulysses_rank)), backend="nccl"
            )

    ulysses_groups: dict[tuple[int, int], Any] = {}
    if topology.ulysses_degree > 1:
        for stage in range(topology.stage_count):
            for tp_rank in range(topology.tp_size):
                ulysses_groups[(stage, tp_rank)] = distributed.new_group(
                    ranks=list(topology.ulysses_ranks(stage, tp_rank)),
                    backend="nccl",
                )

    checkpoint_groups: list[Any] = []
    for tp_rank in range(topology.tp_size):
        checkpoint_groups.append(
            distributed.new_group(
                ranks=list(topology.checkpoint_replica_ranks(tp_rank)),
                backend="nccl",
            )
        )

    # Full-prefix traffic runs opposite to the activation pipeline.  Give each
    # TP/Ulysses lane its own three-stage communicator so delayed stage2->0/1
    # sends cannot be ordered against stage1->2 activation sends on WORLD.
    full_prefix_groups: list[Any] = []
    for stage_local_rank in range(topology.ranks_per_stage):
        full_prefix_groups.append(
            distributed.new_group(
                ranks=[
                    stage * topology.ranks_per_stage + stage_local_rank
                    for stage in range(topology.stage_count)
                ],
                backend="nccl",
            )
        )

    # Each corresponding TP/Ulysses lane publishes one shared clean-prefix
    # activation scale.  Keep this blocking reduction off the communicator
    # used by delayed full-prefix point-to-point traffic.
    fp8_scale_sync_groups: list[Any] = []
    for stage_local_rank in range(topology.ranks_per_stage):
        fp8_scale_sync_groups.append(
            distributed.new_group(
                ranks=[
                    stage * topology.ranks_per_stage + stage_local_rank
                    for stage in range(topology.stage_count)
                ],
                backend="nccl",
            )
        )

    # Clean-window feedback has the same lane membership as full-prefix
    # transport, but needs its own communicator so blocking broadcasts cannot
    # be ordered against outstanding asynchronous prefix sends.
    clean_feedback_groups: list[Any] = []
    for stage_local_rank in range(topology.ranks_per_stage):
        clean_feedback_groups.append(
            distributed.new_group(
                ranks=[
                    stage * topology.ranks_per_stage + stage_local_rank
                    for stage in range(topology.stage_count)
                ],
                backend="nccl",
            )
        )

    context = ParallelContext(
        tp_world_size=topology.tp_size,
        tp_rank=topology.tp_rank,
        ulysses_world_size=topology.ulysses_degree,
        ulysses_rank=topology.ulysses_rank,
        ring_world_size=1,
        ring_rank=0,
        tp_group=tp_groups[(topology.stage_index, topology.ulysses_rank)],
        ulysses_group=(
            None
            if topology.ulysses_degree == 1
            else ulysses_groups[(topology.stage_index, topology.tp_rank)]
        ),
    )
    return HiARProcessGroups(
        stage_group=stage_groups[topology.stage_index],
        checkpoint_replica_group=checkpoint_groups[topology.tp_rank],
        full_prefix_group=full_prefix_groups[topology.stage_local_rank],
        fp8_scale_sync_group=fp8_scale_sync_groups[topology.stage_local_rank],
        clean_feedback_group=clean_feedback_groups[topology.stage_local_rank],
        parallel_context=context,
    )


def broadcast_checkpoint_replicas(
    torch: Any,
    *,
    transformer: Any,
    topology: HiARTopology,
    groups: HiARProcessGroups,
) -> int:
    """Replicate one loaded TP shard to every Ulysses and pipeline replica."""

    state = transformer.state_dict(keep_vars=True)
    for value in state.values():
        if not torch.is_tensor(value):
            raise TypeError("H3 transformer state contains a non-tensor value")
        torch.distributed.broadcast(
            value,
            src=topology.checkpoint_source_rank,
            group=groups.checkpoint_replica_group,
        )
    return len(state)


def summarize_hiar_throughput(
    completion_seconds: Sequence[float],
    frame_counts: Sequence[int],
    *,
    steady_start_block_index: int = 1,
) -> dict[str, float | int]:
    if len(completion_seconds) != len(frame_counts) or not completion_seconds:
        raise ValueError("HiAR completions and frame counts must be non-empty and aligned")
    if any(value <= 0 for value in frame_counts):
        raise ValueError("HiAR frame counts must be positive")
    if any(
        right <= left
        for left, right in zip(completion_seconds, completion_seconds[1:])
    ):
        raise ValueError("HiAR completion times must be strictly increasing")
    if not 1 <= steady_start_block_index < len(completion_seconds):
        raise ValueError("steady start block must follow block zero and exist")
    result: dict[str, float | int] = {
        "block_count": len(frame_counts),
        "native_frame_count": sum(frame_counts),
        "first_block_completion_seconds": float(completion_seconds[0]),
    }
    anchor = steady_start_block_index - 1
    intervals = [
        right - left
        for left, right in zip(
            completion_seconds[anchor:-1],
            completion_seconds[steady_start_block_index:],
        )
    ]
    steady_frames = sum(frame_counts[steady_start_block_index:])
    steady_seconds = completion_seconds[-1] - completion_seconds[anchor]
    result.update(
        steady_start_block_index=steady_start_block_index,
        steady_block_count=len(frame_counts) - steady_start_block_index,
        steady_native_frame_count=steady_frames,
        steady_wall_seconds=float(steady_seconds),
        steady_stream_fps=float(steady_frames / steady_seconds),
        steady_output_interval_seconds_mean=float(mean(intervals)),
        steady_output_interval_seconds_max=float(max(intervals)),
        steady_output_interval_seconds_min=float(min(intervals)),
    )
    return result


__all__ = [
    "HIAR_STAGE_COUNT",
    "HiARDelayedFullPrefixMemory",
    "HiARFullPrefixMemoryUpdate",
    "HiARProgressAVKVCache",
    "HiARProgressSession",
    "HiARProcessGroups",
    "HiARTopology",
    "broadcast_checkpoint_replicas",
    "build_hiar_process_groups",
    "hiar_full_prefix_update_seconds",
    "summarize_hiar_throughput",
]
