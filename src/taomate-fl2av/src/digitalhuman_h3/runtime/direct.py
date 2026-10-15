# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import threading
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, TypeVar

from ..inference.acceleration import (
    InferenceAccelerationConfig,
    InferenceComponentTimer,
    apply_dit_inference_acceleration,
)
from ..denoise_schedule import DISTILLED_STATE_INDICES
from ..paths import H3Paths
from ..streaming.full_prefix import (
    FULL_PREFIX_AUDIO_VIDEO,
    FULL_PREFIX_MODALITIES,
    FULL_PREFIX_VIDEO_ONLY,
)
from .ppu_torch import execution_torch_facts, load_execution_torch


_TP_SIZE = 2
_DEFAULT_ULYSSES_DEGREE = 2
_SUPPORTED_ULYSSES_DEGREES = (1, 2, 4, 8)
DENOISE_SCHEDULE_CHOICES = ("checkpoint_default", "distilled4", "teacher50")
_T = TypeVar("_T")


class H3RuntimeError(RuntimeError):
    pass


def resolve_inference_sigma_state_indices(
    denoise_schedule: str,
    *,
    checkpoint_loaded: bool,
) -> tuple[int, ...] | None:
    if denoise_schedule not in DENOISE_SCHEDULE_CHOICES:
        raise ValueError(
            "denoise_schedule must be checkpoint_default, distilled4, or teacher50"
        )
    if denoise_schedule == "teacher50":
        return None
    if denoise_schedule == "distilled4" or checkpoint_loaded:
        return DISTILLED_STATE_INDICES
    return None


def _inside(path: Path, root: Path) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output path must stay inside {root}") from exc
    return path


def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _stage3_extended_streaming_contract(
    checkpoint: Path,
) -> tuple[float | None, str, str | None, str]:
    if (checkpoint / "adapter_model.safetensors").is_file():
        from ..inference.lora_checkpoint import read_h3_adapter_config

        contract = read_h3_adapter_config(checkpoint)
    else:
        marker = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
        contract = marker.get("contract", {})
    if contract.get("stage") != "stage3":
        return None, "off", None, "first_block"
    runtime = contract.get("runtime_contract", {})
    prefix = runtime.get("full_prefix", {})
    self_forcing = runtime.get("self_forcing", {})
    keyframe_policy = self_forcing.get("keyframe_condition_policy", "first_block")
    if keyframe_policy not in {"first_block", "clip_absolute_persistent"}:
        raise H3RuntimeError("Stage3 checkpoint has an invalid keyframe condition policy")
    handoff = self_forcing.get("phase_latent_handoff", {})
    handoff_mode = handoff.get("mode", "off")
    if handoff_mode not in {"off", "first_transition", "all_transitions"}:
        raise H3RuntimeError(
            "Stage3 checkpoint has an invalid phase latent handoff mode"
        )
    if prefix.get("enabled") is False:
        if prefix.get("schema") != (
            "digitalhuman_h3.training.stage3_full_prefix.disabled.v1"
        ):
            raise H3RuntimeError(
                "Stage3 checkpoint disabled full-prefix schema differs"
            )
        return None, str(handoff_mode), None, str(keyframe_policy)
    value = prefix.get("half_life_seconds")
    modality = prefix.get("modality", FULL_PREFIX_VIDEO_ONLY)
    if (
        prefix.get("enabled") is not True
        or modality not in FULL_PREFIX_MODALITIES
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise H3RuntimeError(
            "Stage3 checkpoint has no valid enabled full-prefix inference contract"
        )
    schema = prefix.get("schema")
    if schema is not None:
        lag_blocks = prefix.get("lag_blocks", 0)
        expected_schema = (
            "digitalhuman_h3.training.stage3_full_prefix"
            if modality == FULL_PREFIX_VIDEO_ONLY
            else "digitalhuman_h3.training.stage3_full_prefix_av"
        ) + f".v{2 if lag_blocks else 1}"
        if schema != expected_schema:
            raise H3RuntimeError(
                "Stage3 checkpoint full-prefix schema and modality differ"
            )
    return float(value), str(handoff_mode), str(modality), str(keyframe_policy)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _native_environment(torch_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "digitalhuman_h3.runtime.native.v1",
        "backend": "digitalhuman_h3_native_torch",
        "execution_torch": dict(torch_facts),
        "distributions": {
            name: _distribution_version(name)
            for name in (
                "diffusers",
                "imageio",
                "imageio-ffmpeg",
                "safetensors",
                "scipy",
                "transformers",
            )
        },
        "external_service_required": False,
    }


class H3Runtime:
    """Native MiniMax H3 runtime with TP2 and Ulysses1/2/4/8."""

    def __init__(
        self,
        *,
        model_root: Path | None = None,
        variant: str = "fl2va",
        tp_size: int = _TP_SIZE,
        ulysses_degree: int = _DEFAULT_ULYSSES_DEGREE,
        adapter_checkpoint: Path | None = None,
        adapter_role: str = "generator",
        full_checkpoint: Path | None = None,
        allow_intermediate_checkpoint: bool = False,
        inference_acceleration: InferenceAccelerationConfig | None = None,
        denoise_schedule: str = "checkpoint_default",
    ) -> None:
        if variant not in {"fl2va", "ref2va"}:
            raise H3RuntimeError("native runtime supports FL2VA and Ref2VA partitions")
        if variant == "ref2va" and (adapter_checkpoint is not None or full_checkpoint is not None):
            raise H3RuntimeError("image Ref2VA currently uses the released base weights only")
        single_gpu_low_stage = (
            (tp_size, ulysses_degree) == (1, 1)
            and os.environ.get("H3_PROGRESSIVE_PP_STAGE") in {"0", "1"}
        )
        if not single_gpu_low_stage and (tp_size != _TP_SIZE or ulysses_degree not in _SUPPORTED_ULYSSES_DEGREES):
            raise ValueError(
                "native H3 runtime requires tp_size=2 and ulysses_degree=1, 2, 4 or 8; "
                "progressive PP LOW stages also support tp_size=1 and ulysses_degree=1"
            )
        if full_checkpoint is not None:
            raise ValueError("this inference release supports generator LoRA checkpoints only")
        self.inference_acceleration = (
            inference_acceleration or InferenceAccelerationConfig()
        )
        load_time_w8a8_policy = (
            self.inference_acceleration.dit_w8a8_policy
            if full_checkpoint is None
            else None
        )
        self.conditioned_text_policy = (
            "text_only"
            if adapter_checkpoint is not None or full_checkpoint is not None
            else "trajectory_multimodal"
        )
        self.inference_sigma_state_indices = resolve_inference_sigma_state_indices(
            denoise_schedule,
            checkpoint_loaded=(
                adapter_checkpoint is not None or full_checkpoint is not None
            ),
        )
        self.denoise_schedule = (
            "distilled4"
            if self.inference_sigma_state_indices is not None
            else "teacher50"
        )
        self.requested_denoise_schedule = denoise_schedule

        paths = H3Paths.from_env()
        self.source_model_root = (model_root or paths.model_root).resolve(strict=True)
        self.output_root = paths.output_root.resolve()
        self.variant = variant
        self.tp_size = tp_size
        self.ulysses_degree = ulysses_degree
        self.expected_world_size = tp_size * ulysses_degree
        self.server_args = SimpleNamespace(
            backend="native",
            model_variant=variant,
            num_gpus=self.expected_world_size,
            tp_size=self.tp_size,
            sp_degree=self.ulysses_degree,
            ulysses_degree=self.ulysses_degree,
            ring_degree=1,
            dp_size=1,
            enable_cfg_parallel=False,
            cfg_parallel_degree=1,
            enable_torch_compile=False,
            warmup_mode="off",
            quantization=None,
        )

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        if self.world_size != self.expected_world_size:
            raise ValueError(
                "native H3 torchrun world size must equal "
                f"tp_size * ulysses_degree = {self.expected_world_size}, "
                f"got {self.world_size}"
            )
        if not 0 <= self.rank < self.world_size or not 0 <= self.local_rank < self.world_size:
            raise ValueError("invalid torchrun rank environment")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            self.torch = load_execution_torch()
            torch_facts = execution_torch_facts(self.torch)
        except (ImportError, RuntimeError, OSError) as exc:
            raise H3RuntimeError(f"PyTorch is unavailable: {exc}") from exc
        if not torch_facts["cuda_available"] or self.local_rank >= torch_facts[
            "cuda_device_count"
        ]:
            raise H3RuntimeError(
                f"CUDA device for LOCAL_RANK={self.local_rank} is unavailable; "
                f"cuda_device_count={torch_facts['cuda_device_count']} "
                f"CUDA_VISIBLE_DEVICES={torch_facts['cuda_visible_devices']!r}"
            )
        self.runtime_environment = _native_environment(torch_facts)
        self.torch.cuda.set_device(self.local_rank)
        self.device = self.torch.device("cuda", self.local_rank)
        if not self.torch.distributed.is_initialized():
            self.torch.distributed.init_process_group(
                backend="nccl", init_method="env://", timeout=timedelta(hours=3)
            )
        if (
            self.torch.distributed.get_world_size() != self.world_size
            or self.torch.distributed.get_rank() != self.rank
        ):
            raise H3RuntimeError("torchrun process group differs from rank environment")

        self.parallel_context = self._build_parallel_context()
        self.model_root = self.source_model_root
        self.model_resolution = {
            "variant": variant,
            "runtime_backend": "native",
            "source_model_root": str(self.source_model_root),
            "effective_model_root": str(self.model_root),
            "runtime_environment": self.runtime_environment,
        }
        if self.is_main_process:
            self.output_root.mkdir(mode=0o755, parents=True, exist_ok=True)
        self.barrier()

        pipeline_error: str | None = None
        self.pipeline: Any | None = None
        self._precomputed_text_path = os.environ.get("H3_PRECOMPUTED_TEXT_PATH")
        try:
            from ..model.pipeline import MiniMaxH3NativePipeline

            self.pipeline = MiniMaxH3NativePipeline(
                model_root=self.model_root,
                variant=variant,
                device=self.device,
                parallel_context=self.parallel_context,
                rank=self.rank,
                world_size=self.world_size,
                defer_text_encoder=bool(self._precomputed_text_path),
                transformer_load_w8a8_policy=load_time_w8a8_policy,
            )
        except Exception as exc:
            pipeline_error = _exception_text(exc)
        pipeline_error = self._collective_error("native H3 pipeline load", pipeline_error)
        if pipeline_error:
            raise H3RuntimeError(pipeline_error)
        assert self.pipeline is not None
        self._last_clean_video_latents: Any | None = None
        self._last_clean_audio_latents: Any | None = None
        self._output_decoder_preload_thread: threading.Thread | None = None
        self._output_decoder_preload_error: str | None = None
        self._output_decoder_preload_seconds: float | None = None
        self._text_precompute_receipt: dict[str, Any] | None = None
        self.full_checkpoint_resolution: dict[str, Any] | None = None
        self.full_prefix_half_life_seconds: float | None = None
        self.full_prefix_modality: str | None = None
        self.phase_latent_handoff_mode = "off"
        self.keyframe_condition_policy = "first_block"
        self.adapter_resolution: dict[str, Any] | None = None
        if adapter_checkpoint is not None:
            try:
                from ..inference.lora_checkpoint import apply_h3_lora_checkpoint

                receipt = apply_h3_lora_checkpoint(
                    self.pipeline.transformer,
                    adapter_checkpoint,
                    role=adapter_role,
                    allow_intermediate=allow_intermediate_checkpoint,
                )
                self.adapter_resolution = dict(receipt)
                self.model_resolution["adapter"] = self.adapter_resolution
                adapter_streaming_contract = _stage3_extended_streaming_contract(
                    adapter_checkpoint.expanduser().resolve(strict=True)
                )
                if (
                    self.full_prefix_modality is not None
                    and adapter_streaming_contract[2]
                    != self.full_prefix_modality
                ):
                    raise H3RuntimeError(
                        "full and adapter checkpoints select different "
                        "full-prefix modalities"
                    )
                (
                    self.full_prefix_half_life_seconds,
                    self.phase_latent_handoff_mode,
                    self.full_prefix_modality,
                    self.keyframe_condition_policy,
                ) = adapter_streaming_contract
            except Exception as exc:
                raise H3RuntimeError(
                    f"native H3 LoRA load failed: {_exception_text(exc)}"
                ) from exc
        acceleration_error: str | None = None
        acceleration_receipt: dict[str, Any] | None = None
        try:
            acceleration_receipt = apply_dit_inference_acceleration(
                self.torch,
                self.pipeline.transformer,
                self.inference_acceleration,
            )
            self.pipeline.configure_inference_video_vae(
                fp8_linears=self.inference_acceleration.video_vae_fp8_linears,
                fp8_attention=self.inference_acceleration.video_vae_fp8_attention,
                exact_fusions=self.inference_acceleration.video_vae_exact_fusions,
                batched_tiles=self.inference_acceleration.video_vae_batched_tiles,
                balanced_cpu_output=(
                    self.inference_acceleration.video_vae_balanced_cpu_output
                ),
                temporal_window_dp=(
                    self.inference_acceleration.video_vae_temporal_window_dp
                ),
                temporal_window_microbatch_size=(
                    self.inference_acceleration.video_vae_temporal_window_microbatch_size
                ),
                gpu_rgb24_output=(
                    self.inference_acceleration.video_vae_gpu_rgb24_output
                ),
                keep_on_device_after_decode=(
                    self.inference_acceleration.video_vae_keep_on_device_after_decode
                ),
                decoder_tile_size=(
                    self.inference_acceleration.video_vae_decoder_tile_size
                ),
                decoder_tile_overlap=(
                    self.inference_acceleration.video_vae_decoder_tile_overlap
                ),
            )
        except Exception as exc:
            acceleration_error = _exception_text(exc)
        acceleration_error = self._collective_error(
            "native H3 inference acceleration",
            acceleration_error,
        )
        if acceleration_error:
            raise H3RuntimeError(acceleration_error)
        self.model_resolution["inference_acceleration"] = acceleration_receipt
        self._component_timer: InferenceComponentTimer | None = None
        if self.inference_acceleration.record_component_timings:
            self._component_timer = InferenceComponentTimer(
                self.torch,
                rank=self.rank,
                world_size=self.world_size,
            )
            self._component_timer.attach_transformer(self.pipeline.transformer)
            self.pipeline.set_component_timer(self._component_timer)
        sigma_points = (
            len(self.inference_sigma_state_indices)
            if self.inference_sigma_state_indices is not None
            else 50
        )
        self.model_resolution["denoise_schedule"] = {
            "mode": self.denoise_schedule,
            "requested_mode": self.requested_denoise_schedule,
            "teacher_sigma_points": 50,
            "retained_state_indices": (
                list(self.inference_sigma_state_indices)
                if self.inference_sigma_state_indices is not None
                else None
            ),
            "sigma_points": sigma_points,
            "model_forwards": sigma_points - 1,
        }
        self.model_resolution["conditioned_text_policy"] = (
            self.conditioned_text_policy
        )
        self.model_resolution["keyframe_condition_policy"] = self.keyframe_condition_policy


    def _build_parallel_context(self) -> Any:
        from ..distributed import ParallelContext

        tp_groups: list[Any] = []
        for ulysses_rank in range(self.ulysses_degree):
            ranks = [
                ulysses_rank * self.tp_size + tp_rank
                for tp_rank in range(self.tp_size)
            ]
            tp_groups.append(
                self.torch.distributed.new_group(ranks=ranks, backend="nccl")
            )
        ulysses_groups: list[Any] = []
        for tp_rank in range(self.tp_size):
            ranks = [
                ulysses_rank * self.tp_size + tp_rank
                for ulysses_rank in range(self.ulysses_degree)
            ]
            ulysses_groups.append(
                self.torch.distributed.new_group(ranks=ranks, backend="nccl")
            )
        return ParallelContext(
            tp_world_size=self.tp_size,
            tp_rank=self.rank % self.tp_size,
            ulysses_world_size=self.ulysses_degree,
            ulysses_rank=self.rank // self.tp_size,
            ring_world_size=1,
            ring_rank=0,
            tp_group=tp_groups[self.rank // self.tp_size],
            ulysses_group=ulysses_groups[self.rank % self.tp_size],
        )

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def modules(self) -> dict[str, Any]:
        assert self.pipeline is not None
        return self.pipeline.modules

    @property
    def last_clean_video_latents(self) -> Any | None:
        return self._last_clean_video_latents

    @property
    def last_clean_audio_latents(self) -> Any | None:
        return self._last_clean_audio_latents

    @property
    def last_video_decode_receipt(self) -> Mapping[str, Any] | None:
        assert self.pipeline is not None
        return self.pipeline.last_video_decode_receipt

    def decode_video_latents(
        self,
        video_latents: Any,
        *,
        output_path: Path,
        video_filter: str | None = None,
        published_video_frames: int | None = None,
        video_crf: int = 12,
        verify_video_decode: bool = True,
    ) -> Path:
        """Collectively decode one complete CPU video-latent timeline."""

        assert self.pipeline is not None
        self.wait_output_decoder_preload()
        resolved_output = _inside(output_path, self.output_root)
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return self.pipeline._decode_and_save_parallel(
            video_latents=video_latents,
            audio_latents=None,
            output_path=resolved_output,
            fps=24,
            error_prefix="full-sequence Video VAE decode failed",
            decode_audio=False,
            video_filter=video_filter,
            published_video_frames=published_video_frames,
            video_crf=video_crf,
            verify_video_decode=verify_video_decode,
        )

    def preload_output_decoder_async(self) -> None:
        """Load CPU AV-VAE weights while the DiT generates the latent timeline."""

        assert self.pipeline is not None
        if self.pipeline.output_decoder is not None:
            return
        if self._output_decoder_preload_thread is not None:
            return

        def preload() -> None:
            started = time.perf_counter()
            try:
                self.pipeline._load_output_decoder()
            except Exception as exc:
                self._output_decoder_preload_error = _exception_text(exc)
            finally:
                self._output_decoder_preload_seconds = time.perf_counter() - started

        self._output_decoder_preload_thread = threading.Thread(
            target=preload,
            name=f"h3-output-decoder-preload-rank{self.rank}",
            daemon=True,
        )
        self._output_decoder_preload_thread.start()

    def wait_output_decoder_preload(self) -> None:
        thread = self._output_decoder_preload_thread
        if thread is None:
            return
        thread.join()
        self._output_decoder_preload_thread = None
        collective_error = self._collective_error(
            "output decoder preload",
            self._output_decoder_preload_error,
        )
        if collective_error:
            raise H3RuntimeError(collective_error)
        preload_seconds: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(
            preload_seconds,
            self._output_decoder_preload_seconds,
        )
        if self.is_main_process:
            print(
                json.dumps(
                    {
                        "status": "output_decoder_preload_complete",
                        "seconds_max": max(float(value) for value in preload_seconds),
                        "seconds_by_rank": [
                            float(value) for value in preload_seconds
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    def decode_audio_latents(self, audio_latents: Any) -> tuple[Any, int]:
        """Decode one complete CPU audio-latent timeline on rank zero."""

        if not self.is_main_process:
            raise H3RuntimeError("only rank zero decodes the published audio timeline")
        assert self.pipeline is not None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        decoder = self.pipeline._load_output_decoder()
        timer_context = (
            nullcontext()
            if self._component_timer is None
            else self._component_timer.region("audio_vae")
        )
        with timer_context:
            return decoder.decode_audio(
                audio_latents=audio_latents,
                device=self.device,
                offload_after_decode=True,
            )

    def component_timing_receipt(self) -> dict[str, Any] | None:
        if self._component_timer is None:
            return None
        return self._component_timer.receipt()

    def inference_acceleration_receipt(self) -> dict[str, Any]:
        receipt = dict(self.model_resolution["inference_acceleration"] or {})
        if self.inference_acceleration.static_timestep_adaln_cache:
            assert self.pipeline is not None
            receipt["static_timestep_adaln_cache_stats"] = (
                self.pipeline.transformer.inference_static_timestep_cache_stats()
            )
        return receipt

    def barrier(self) -> None:
        self.torch.distributed.barrier()

    def broadcast(self, value: Any) -> Any:
        return self._rank_zero_payload(value if self.is_main_process else None)

    def _rank_zero_payload(self, value: _T | None) -> _T:
        payload = [value if self.is_main_process else None]
        self.torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    def _collective_error(self, label: str, local_error: str | None) -> str | None:
        gathered: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(
            gathered, {"rank": self.rank, "error": local_error}
        )
        failures = [item for item in gathered if item.get("error")]
        if not failures:
            return None
        return f"{label} failed collectively (" + "; ".join(
            f"rank {item['rank']}: {item['error']}" for item in failures
        ) + ")"

    def _collective_request(self, request: Mapping[str, Any]) -> None:
        gathered: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(gathered, dict(request))
        if any(item != request for item in gathered):
            raise H3RuntimeError("generation request differs across torchrun ranks")

    def precompute_text_prompts(
        self,
        prompts: Sequence[str],
        *,
        parallel: bool = False,
    ) -> tuple[tuple[Any, Any], ...]:
        normalized_prompts = tuple(prompts)
        if not normalized_prompts or any(
            not isinstance(prompt, str) or not prompt.strip()
            for prompt in normalized_prompts
        ):
            raise ValueError("precomputed prompts must be non-empty strings")
        self._collective_request(
            {
                "operation": "precompute_text_prompts",
                "prompts": list(normalized_prompts),
                "parallel": parallel,
            }
        )
        assert self.pipeline is not None
        from ..distributed import use_parallel_context

        started = time.perf_counter()
        if self._precomputed_text_path:
            cache_path = Path(self._precomputed_text_path)
            deadline = started + 600.0
            while not cache_path.is_file():
                if time.perf_counter() >= deadline:
                    raise H3RuntimeError(
                        f"timed out waiting for shared text features: {cache_path}"
                    )
                time.sleep(0.25)
            payload = self.torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
            if tuple(payload["prompts"]) != normalized_prompts:
                raise H3RuntimeError("shared text features use different prompts")
            encoded = tuple(payload["encoded"])
            if len(encoded) != len(normalized_prompts):
                raise H3RuntimeError("shared text feature count differs")
            self._text_precompute_receipt = {
                "schema": "digitalhuman_h3.inference.text_precompute.v1",
                "mode": "shared_precomputed",
                "prompt_count": len(normalized_prompts),
                "tp_size": self.tp_size,
                "ulysses_degree": self.ulysses_degree,
                "seconds_max": time.perf_counter() - started,
                "source_path": str(cache_path),
                "producer": payload["producer"],
            }
            return encoded
        encoded: list[tuple[Any, Any]] = []
        local_encoded: dict[int, tuple[Any, Any]] = {}
        try:
            with use_parallel_context(self.parallel_context):
                for index, prompt in enumerate(normalized_prompts):
                    if parallel and (
                        index % self.ulysses_degree
                        != self.parallel_context.ulysses_rank
                    ):
                        continue
                    hidden, tags = self.pipeline._encode_text(
                        prompt=prompt,
                        images=(),
                    )
                    local_encoded[index] = (
                        hidden.detach().to(device="cpu").contiguous(),
                        tags.detach().to(device="cpu").contiguous(),
                    )
        finally:
            if os.environ.get("H3_RELEASE_PRECOMPUTED_TEXT_ENCODER") == "1":
                self.pipeline.release_text_encoder()
            else:
                self.pipeline._offload_text()

        if parallel:
            for index in range(len(normalized_prompts)):
                owner = index % self.ulysses_degree
                source_rank = owner * self.tp_size + self.parallel_context.tp_rank
                payload = [
                    local_encoded.get(index) if self.rank == source_rank else None
                ]
                self.torch.distributed.broadcast_object_list(
                    payload,
                    src=source_rank,
                    group=self.parallel_context.ulysses_group,
                )
                item = payload[0]
                if not isinstance(item, tuple) or len(item) != 2:
                    raise H3RuntimeError(
                        f"parallel text prompt {index} broadcast is invalid"
                    )
                encoded.append(item)
        else:
            encoded.extend(
                local_encoded[index] for index in range(len(normalized_prompts))
            )

        local_timing = {
            "rank": self.rank,
            "tp_rank": self.parallel_context.tp_rank,
            "ulysses_rank": self.parallel_context.ulysses_rank,
            "encoded_prompt_count": len(local_encoded),
            "seconds": time.perf_counter() - started,
        }
        timings: list[Any] = [None] * self.world_size
        self.torch.distributed.all_gather_object(timings, local_timing)
        self._text_precompute_receipt = {
            "schema": "digitalhuman_h3.inference.text_precompute.v1",
            "mode": "ulysses_sharded" if parallel else "replicated",
            "prompt_count": len(normalized_prompts),
            "tp_size": self.tp_size,
            "ulysses_degree": self.ulysses_degree,
            "encoded_prompt_count_by_rank": timings,
            "seconds_max": max(float(item["seconds"]) for item in timings),
            "prompt_sources": (
                [
                    {
                        "prompt_index": index,
                        "owner_ulysses_rank": index % self.ulysses_degree,
                        "source_global_rank_by_tp_rank": [
                            (index % self.ulysses_degree) * self.tp_size + tp_rank
                            for tp_rank in range(self.tp_size)
                        ],
                    }
                    for index in range(len(normalized_prompts))
                ]
                if parallel
                else None
            ),
        }
        export_path_value = os.environ.get("H3_PRECOMPUTED_TEXT_EXPORT_PATH")
        if export_path_value and self.is_main_process:
            export_path = Path(export_path_value)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = export_path.with_name(export_path.name + ".tmp")
            self.torch.save(
                {
                    "prompts": normalized_prompts,
                    "encoded": tuple(encoded),
                    "producer": dict(self._text_precompute_receipt),
                },
                pending_path,
            )
            pending_path.replace(export_path)
        return tuple(encoded)

    def text_precompute_receipt(self) -> dict[str, Any] | None:
        receipt = getattr(self, "_text_precompute_receipt", None)
        return None if receipt is None else dict(receipt)



__all__ = ["H3Runtime", "H3RuntimeError"]
