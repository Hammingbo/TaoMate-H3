from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from .attention_hook import H3StreamingAttentionHook, HookMode
from .cache import (
    CleanAVKVCache,
    KVContract,
    TemporaryAVPrefix,
    TemporaryVideoPrefix,
    kv_contract_from_model,
)
from .capability import require_streaming_capability, set_official_attention_hook
from .geometry import StreamPhase, StreamPlan


class H3StreamingSession:
    """State machine around real official H3 transformer forwards.

    The caller performs each low-level transformer forward. Noisy denoise
    forwards can only read committed history. A separate successful sigma=0
    forward stages all 50 layer K/V tensors and atomically commits the block.
    """

    def __init__(
        self,
        plan: StreamPlan,
        *,
        model: Any | None = None,
        server_args: Any | None = None,
        contract: KVContract | None = None,
        cache: CleanAVKVCache | None = None,
        attention_backend: str = "sdpa",
        sol_attn_tau: float = 1.0,
        sol_attn_dense_layers: int = 2,
        sol_attn_dense_denoise_steps: int = 1,
        sol_attn_route_current_video: bool = False,
        direct_sage_kv_workspace: bool = False,
        sol_attn_static_summary_cache: bool = False,
        fused_history_use_sink: bool = False,
        fused_history_recent_blocks: int | None = None,
    ) -> None:
        if not isinstance(plan, StreamPlan):
            raise TypeError("plan must be a StreamPlan")
        self.plan = plan
        if cache is not None and contract is not None and cache.contract != contract:
            raise ValueError("injected clean cache and KV contract differ")
        resolved_contract = contract
        if resolved_contract is None and cache is None and model is not None:
            resolved_contract = kv_contract_from_model(model)
        segmented_fp8_history = attention_backend in {
            "flash_attn3_fp8_segmented",
        }
        if cache is not None and cache.segmented_fp8_history != segmented_fp8_history:
            raise ValueError("streaming cache format differs from attention backend")
        self.cache = cache or CleanAVKVCache(
            resolved_contract,
            segmented_fp8_history=segmented_fp8_history,
        )
        self._block_index_base = self.cache.committed_blocks
        self.hook = H3StreamingAttentionHook(
            self.cache,
            attention_backend=attention_backend,
            sol_attn_tau=sol_attn_tau,
            sol_attn_dense_layers=sol_attn_dense_layers,
            sol_attn_dense_denoise_steps=sol_attn_dense_denoise_steps,
            sol_attn_route_current_video=sol_attn_route_current_video,
            direct_sage_kv_workspace=direct_sage_kv_workspace,
            sol_attn_static_summary_cache=sol_attn_static_summary_cache,
        )
        self.capability = require_streaming_capability(
            model=model,
            server_args=server_args,
        )
        self._installed = False
        self._noisy_blocks: set[int] = set()
        self._fused_history_use_sink = bool(fused_history_use_sink)
        self._fused_history_recent_blocks = fused_history_recent_blocks

    def __enter__(self) -> H3StreamingSession:
        if self._installed:
            raise RuntimeError("H3 streaming session is already installed")
        set_official_attention_hook(self.hook)
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.cache.clean_commit_active:
            self.cache.rollback()
        self.hook.deactivate()
        if self._installed:
            set_official_attention_hook(None)
            self._installed = False

    def phase(self, block_index: int) -> StreamPhase:
        if isinstance(block_index, bool) or not isinstance(block_index, int):
            raise TypeError("block_index must be an integer")
        try:
            return self.plan.phases[block_index]
        except IndexError as exc:
            raise ValueError(f"unknown streaming block {block_index}") from exc

    def _require_installed(self, block_index: int) -> StreamPhase:
        if not self._installed:
            raise RuntimeError("use H3StreamingSession as a context manager")
        phase = self.phase(block_index)
        if self._block_index_base + block_index != self.cache.committed_blocks:
            raise ValueError("streaming blocks must be processed and committed in order")
        return phase

    @contextmanager
    def noisy_step(
        self,
        block_index: int,
        *,
        full_prefix: TemporaryAVPrefix | TemporaryVideoPrefix | None = None,
        denoise_step_index: int | None = None,
    ) -> Iterator[StreamPhase]:
        """Wrap exactly one noisy transformer forward for the current block.

        The low-level caller must build video/audio RoPE coordinates from the
        yielded phase's absolute latent starts. Reusing standalone-block
        zero-based positions is rejected by the integration contract.
        """

        phase = self._require_installed(block_index)
        global_block_index = self._block_index_base + block_index
        self.hook.activate(
            mode=HookMode.NOISY,
            block_index=global_block_index,
            full_prefix=full_prefix,
            denoise_step_index=denoise_step_index,
            phase_index=phase.index,
        )
        try:
            yield phase
        finally:
            self.hook.deactivate()
        self._noisy_blocks.add(global_block_index)

    @contextmanager
    def fused_clean_commit_denoise(
        self,
        previous_block_index: int,
        current_block_index: int,
        *,
        previous_full_prefix: (
            TemporaryAVPrefix | TemporaryVideoPrefix | None
        ) = None,
        current_full_prefix: (
            TemporaryAVPrefix | TemporaryVideoPrefix | None
        ) = None,
        denoise_step_index: int = 0,
    ) -> Iterator[tuple[StreamPhase, StreamPhase]]:
        """Commit clean block N while running noisy step zero for block N+1."""

        previous_phase = self._require_installed(previous_block_index)
        if current_block_index != previous_block_index + 1:
            raise ValueError("fused direct H3 blocks must be adjacent")
        current_phase = self.phase(current_block_index)
        previous_global = self._block_index_base + previous_block_index
        current_global = self._block_index_base + current_block_index
        if previous_global not in self._noisy_blocks:
            raise RuntimeError(
                "fused direct commit requires the previous denoise output"
            )
        self.cache.begin_clean_commit(previous_global)
        self.hook.activate(
            mode=HookMode.FUSED_PROGRESS_DENOISE,
            block_index=previous_global,
            full_prefix=current_full_prefix,
            fused_previous_full_prefix=previous_full_prefix,
            fused_current_use_sink=self._fused_history_use_sink,
            fused_current_recent_blocks=self._fused_history_recent_blocks,
            denoise_step_index=denoise_step_index,
            phase_index=current_phase.index,
        )
        try:
            yield previous_phase, current_phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit()
            except BaseException:
                self.cache.rollback()
                raise
            self._noisy_blocks.remove(previous_global)
            self._noisy_blocks.add(current_global)
        finally:
            self.hook.deactivate()

    @contextmanager
    def clean_commit(
        self,
        block_index: int,
        *,
        full_prefix: TemporaryAVPrefix | TemporaryVideoPrefix | None = None,
        retain_recent_commit_count: int | None = None,
    ) -> Iterator[StreamPhase]:
        """Wrap the dedicated clean sigma=0 forward and commit on success."""

        phase = self._require_installed(block_index)
        global_block_index = self._block_index_base + block_index
        if global_block_index not in self._noisy_blocks:
            raise RuntimeError("a clean commit requires at least one noisy denoise forward")
        self.cache.begin_clean_commit(global_block_index)
        self.hook.activate(
            mode=HookMode.CLEAN_COMMIT,
            block_index=global_block_index,
            full_prefix=full_prefix,
            phase_index=phase.index,
        )
        try:
            yield phase
        except BaseException:
            self.cache.rollback()
            raise
        else:
            try:
                self.cache.commit(
                    retain_recent_commit_count=retain_recent_commit_count
                )
            except BaseException:
                self.cache.rollback()
                raise
            else:
                self._noisy_blocks.remove(global_block_index)
        finally:
            self.hook.deactivate()

    def close(self) -> None:
        self.__exit__(None, None, None)


__all__ = ["H3StreamingSession"]
