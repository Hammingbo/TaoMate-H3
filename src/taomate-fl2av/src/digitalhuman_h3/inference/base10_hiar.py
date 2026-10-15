# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import json

import os

import time

from dataclasses import dataclass, replace

from pathlib import Path

from typing import Any, Mapping

from digitalhuman_h3.streaming.geometry import (
    StreamPhase,
    StreamPlan,
    canonical_continuation_plan,
    plan_duration,
)

BASE10_HIAR_REQUEST_COUNT = 12

BASE10_HIAR_REQUEST_SECONDS = 5.0

BASE10_HIAR_CHUNKS_PER_REQUEST = 4

BASE10_HIAR_STAGE_COUNT = 3

BASE10_HIAR_TP_SIZE = 2

BASE10_HIAR_DIT_GPU_COUNT = 6

BASE10_HIAR_TEACHER_GPU_COUNT = 2

BASE10_TEACHER_STATE_NUMBERS = (3, 6, 9)

BASE10_STAGE3_TARGET_STATE_INDICES = (16, 33, 49)

BASE10_VIDEO_RECENT_CHUNKS = 2

BASE10_TEACHER_SCHEMA = "digitalhuman_h3.inference.base10_hiar_teacher.v1"

BASE10_ROLLOVER_REFERENCE_LATENTS = 40

BASE10_KEYFRAME_STRATEGY = "joint_av_first_last_keyframes"

def base10_teacher_producer_topology() -> dict[str, Any]:
    """Describe the fixed topology of ``generate_base10_audio_teacher.py``."""

    return {
        "entrypoint": "scripts/inference/generate_base10_audio_teacher.py",
        "world_size": BASE10_HIAR_TEACHER_GPU_COUNT,
        "tensor_parallel_size": BASE10_HIAR_TP_SIZE,
        "ulysses_degree": 1,
        "teacher_gpu_count": BASE10_HIAR_TEACHER_GPU_COUNT,
    }

def base10_rollover_teacher_producer_topology(
    world_size: int | None = None,
) -> dict[str, Any]:
    """Describe the configured TP2 FL2VA audio-rollover producer."""

    if world_size is None:
        world_size = int(os.environ.get("H3_LOW_TEACHER_NPROC", "8"))

    return {
        "entrypoint": "scripts/inference/generate_base10_audio_rollover_teacher.py",
        "world_size": world_size,
        "tensor_parallel_size": 2,
        "ulysses_degree": world_size // 2,
        "teacher_gpu_count": world_size,
    }

def base10_teacher_geometry(
    *,
    width: int,
    height: int,
    video_latent_h: int,
    video_latent_w: int,
) -> dict[str, int]:
    return {
        "width": int(width),
        "height": int(height),
        "video_latent_h": int(video_latent_h),
        "video_latent_w": int(video_latent_w),
    }

def base10_keyframe_teacher_producer_topology() -> dict[str, Any]:
    return {**base10_rollover_teacher_producer_topology(8),
            "entrypoint": "scripts/inference/generate_base10_keyframe_teacher.py"}

@dataclass(frozen=True)
class Base10HiARRequest:
    index: int
    prompt: str
    seed: int
    phase_start: int
    phase_stop: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def video_latent_count(self) -> int:
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self) -> int:
        return self.audio_latent_stop - self.audio_latent_start

@dataclass(frozen=True)
class Base10HiARSchedule:
    plan: StreamPlan
    requests: tuple[Base10HiARRequest, ...]

    @property
    def phase_count(self) -> int:
        return len(self.plan.phases)

    @property
    def video_latent_count(self) -> int:
        return self.requests[-1].video_latent_stop

    @property
    def audio_latent_count(self) -> int:
        return self.requests[-1].audio_latent_stop

@dataclass(frozen=True)
class CompletedBase10TeacherArtifact:
    """A completed BF16 Base10 teacher set consumed request by request.

    Geometry is part of the artifact contract because the current-request
    audio noise is sampled after video noise from the same generator. A
    teacher produced at another spatial size therefore cannot be reused even
    though only its audio milestones are persisted.
    """

    root: Path
    case_id: str
    schedule: Base10HiARSchedule
    completion: Mapping[str, Any]
    producer_topology: Mapping[str, Any]
    producer_geometry: Mapping[str, int]

    def audio_noise_seed(self, request_index: int) -> int:
        """Return the producer's audio-only RNG seed for one request."""

        sequence = self.completion.get("audio_noise_seed_sequence")
        if isinstance(sequence, list) and len(sequence) == len(self.schedule.requests):
            return int(sequence[request_index])
        if self.completion.get("strategy") == (
            "previous_clean_audio_tail_reference_then_new_noise"
        ):
            raise RuntimeError("rollover Base10 teacher has no audio seed sequence")
        return int(self.schedule.requests[request_index].seed)

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        case_id: str,
        schedule: Base10HiARSchedule,
        consumer_geometry: Mapping[str, Any],
        visual_conditioning: Mapping[str, Any] | None = None,
    ) -> "CompletedBase10TeacherArtifact":
        resolved = root.expanduser().resolve(strict=True)
        marker_path = resolved / "complete.json"
        try:
            completion = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read completed Base10 teacher marker: {exc}"
            ) from exc
        expected = {
            "schema": "digitalhuman_h3.inference.base10_hiar_teacher_set.v1",
            "case_id": case_id,
            "request_count": len(schedule.requests),
        }
        if not isinstance(completion, dict) or any(
            completion.get(key) != value for key, value in expected.items()
        ):
            raise RuntimeError("completed Base10 teacher marker differs")
        base_precision = completion.get("base_precision", "bf16")
        if base_precision != "bf16":
            raise RuntimeError("external Base10 teacher artifact must be BF16")
        expected_geometry_keys = (
            "width",
            "height",
            "video_latent_h",
            "video_latent_w",
        )
        producer_geometry = completion.get("producer_geometry")
        if not isinstance(producer_geometry, dict):
            raise RuntimeError(
                "completed Base10 teacher marker has no producer geometry"
            )
        expected_geometry = {
            key: int(consumer_geometry[key]) for key in expected_geometry_keys
        }
        if producer_geometry != expected_geometry:
            raise RuntimeError(
                "external Base10 teacher producer geometry differs from consumer"
            )
        keyframe_teacher = completion.get("strategy") == BASE10_KEYFRAME_STRATEGY
        if keyframe_teacher or visual_conditioning is not None:
            if (not keyframe_teacher or visual_conditioning is None
                    or completion.get("visual_conditioning") != dict(visual_conditioning)):
                raise RuntimeError("Base10 teacher visual conditions or endpoint differ")
        expected_topology = base10_keyframe_teacher_producer_topology() if keyframe_teacher else (
            base10_rollover_teacher_producer_topology()
            if completion.get("strategy")
            == "previous_clean_audio_tail_reference_then_new_noise"
            else base10_teacher_producer_topology()
        )
        if keyframe_teacher or completion.get("strategy") == "previous_clean_audio_tail_reference_then_new_noise":
            if completion.get("partition") != "fl2va":
                raise RuntimeError(
                    "rollover Base10 teacher must use the FL2VA partition"
                )
            audio_seed_sequence = completion.get("audio_noise_seed_sequence")
            if (
                not isinstance(audio_seed_sequence, list)
                or len(audio_seed_sequence) != len(schedule.requests)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < 2**63
                    for value in audio_seed_sequence
                )
            ):
                raise RuntimeError(
                    "rollover Base10 teacher audio seed sequence differs"
                )
        topology = completion.get("producer_topology", expected_topology)
        if not isinstance(topology, dict) or any(
            topology.get(key) != value
            for key, value in expected_topology.items()
        ):
            raise RuntimeError("external Base10 teacher topology differs")
        return cls(
            root=resolved,
            case_id=case_id,
            schedule=schedule,
            completion=completion,
            producer_topology=dict(topology),
            producer_geometry=dict(producer_geometry),
        )

    def load_request(
        self,
        torch: Any,
        *,
        request_index: int,
        device: Any,
    ) -> dict[str, Any]:
        if not 0 <= request_index < len(self.schedule.requests):
            raise RuntimeError("external Base10 teacher request index is outside set")
        request = self.schedule.requests[request_index]
        path = base10_teacher_path(self.root, request_index)
        if not path.is_file():
            raise RuntimeError(
                f"completed Base10 teacher request is absent: {path}"
            )
        request_receipt_path = base10_teacher_receipt_path(
            self.root, request_index
        )
        try:
            request_receipt = json.loads(
                request_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read Base10 teacher request receipt: {exc}"
            ) from exc
        request_expected = {
            "schema": BASE10_TEACHER_SCHEMA,
            "request_index": request.index,
            "prompt": request.prompt,
            "seed": request.seed,
            "audio_latent_count": request.audio_latent_count,
            "producer_geometry": dict(self.producer_geometry),
        }
        if not isinstance(request_receipt, dict) or any(
            request_receipt.get(key) != value
            for key, value in request_expected.items()
        ):
            raise RuntimeError("Base10 teacher request receipt differs")
        milestones = wait_for_base10_teacher_request(
            torch,
            root=self.root,
            request=request,
        )
        device_milestones = {
            index: value.to(device=device, dtype=torch.float32)
            for index, value in enumerate(milestones)
        }
        receipt = {
            "schema": "digitalhuman_h3.streaming.base_audio_teacher.v1",
            "mode": "base10_milestones",
            "model": "base_h3_with_inference_lora_bypassed",
            "packed_video_rows": 0,
            "packed_audio_rows": 2 * request.audio_latent_count,
            "audio_latents_per_channel": request.audio_latent_count,
            "schedule": "base_10_state_prefix",
            "full_state_count": 10,
            "executed_forwards": 9,
            "captured_base_state_numbers": list(BASE10_TEACHER_STATE_NUMBERS),
            "stage3_target_state_indices": list(
                BASE10_STAGE3_TARGET_STATE_INDICES
            ),
            "reused_current_request_initial_audio_noise": True,
            "reused_by_every_streaming_chunk": True,
            "video_projection_executed": False,
            "video_output_head_executed": False,
            "source": "external_bf16_artifact",
            "base_precision": "bf16",
            "w8a8_inherited": False,
            "producer_topology": dict(self.producer_topology),
            "producer_geometry": dict(self.producer_geometry),
            "consumer_teacher_model_resident": False,
            "artifact_root": str(self.root),
            "artifact_case_id": self.case_id,
            "artifact_request_index": request_index,
            "artifact_completion_schema": self.completion["schema"],
        }
        producer_teacher = request_receipt.get("teacher")
        if self.completion.get("strategy") == BASE10_KEYFRAME_STRATEGY:
            if (not isinstance(producer_teacher, dict)
                    or producer_teacher.get("visual_conditioning") != self.completion["visual_conditioning"]
                    or producer_teacher.get("conditioned_forward_count") != 9
                    or producer_teacher.get("audio_noise_seed") != self.audio_noise_seed(request_index)):
                raise RuntimeError("Base10 keyframe teacher request conditioning differs")
            receipt.update(producer_teacher)
            receipt.update(mode="base10_fl2va_joint_keyframes", source="external_bf16_artifact")
        if self.completion.get("strategy") == (
            "previous_clean_audio_tail_reference_then_new_noise"
        ):
            if not isinstance(producer_teacher, dict):
                raise RuntimeError("rollover Base10 teacher receipt is absent")
            if producer_teacher.get("partition") != "fl2va":
                raise RuntimeError(
                    "rollover Base10 teacher request did not use FL2VA"
                )
            audio_noise_seed = self.audio_noise_seed(request_index)
            if producer_teacher.get("audio_noise_seed") != audio_noise_seed:
                raise RuntimeError(
                    "rollover Base10 teacher request audio seed differs"
                )
            receipt.update(
                {
                    "mode": "base10_fl2va_tail40_rollover",
                    "active_milestones_are_exact_canonical_target": True,
                    "target_geometry": "canonical_active",
                    "audio_noise_seed": audio_noise_seed,
                    "reference_latents_per_channel": producer_teacher.get(
                        "reference_latents_per_channel"
                    ),
                    "reference_duration_seconds": producer_teacher.get(
                        "reference_duration_seconds"
                    ),
                    "reference_rope_interval": producer_teacher.get(
                        "reference_rope_interval"
                    ),
                    "target_rope_interval": producer_teacher.get(
                        "target_rope_interval"
                    ),
                    "prefix_max_abs_change": producer_teacher.get(
                        "prefix_max_abs_change"
                    ),
                    "continuation_partition": producer_teacher.get(
                        "partition"
                    ),
                    "prefix_source_request": producer_teacher.get(
                        "prefix_source_request"
                    ),
                    "prefix_source_tail_interval": producer_teacher.get(
                        "prefix_source_tail_interval"
                    ),
                    "packed_text_rows": producer_teacher.get(
                        "packed_text_rows"
                    ),
                    "packed_audio_rows": producer_teacher.get(
                        "packed_audio_rows"
                    ),
                    "packed_live_rows": producer_teacher.get(
                        "packed_live_rows"
                    ),
                    "packed_padded_rows": producer_teacher.get(
                        "packed_padded_rows"
                    ),
                    "official_rng_audio_latents_per_channel": producer_teacher.get(
                        "official_rng_audio_latents_per_channel"
                    ),
                    "noise_suffix_slice_per_channel": producer_teacher.get(
                        "noise_suffix_slice_per_channel"
                    ),
                    "discarded_video_noise_elements": producer_teacher.get(
                        "discarded_video_noise_elements"
                    ),
                    "teacher_persistent_kv": False,
                }
            )
        return {"milestones": device_milestones, "receipt": receipt}

@dataclass(frozen=True)
class Base10VideoPrefixUpdate:
    canonical_rows: Any
    block_seconds: float

class Base10DelayedVideoFullPrefixMemory:
    """Lagged video-only canonical EMA matching the Stage3 Base10 contract."""

    def __init__(self, *, half_life_seconds: float, lag_blocks: int) -> None:
        from digitalhuman_h3.streaming.full_prefix import FullPrefixEMA

        if isinstance(lag_blocks, bool) or not isinstance(lag_blocks, int) or lag_blocks < 0:
            raise ValueError("Base10 full-prefix lag must be non-negative")
        self.lag_blocks = lag_blocks
        self._memory = FullPrefixEMA(half_life_seconds=half_life_seconds)
        self._pending: dict[int, Base10VideoPrefixUpdate] = {}
        self._visible_source_block_index: int | None = None

    @property
    def initialized(self) -> bool:
        return self._memory.initialized

    @property
    def visible_source_block_index(self) -> int | None:
        return self._visible_source_block_index

    def enqueue(
        self,
        source_block_index: int,
        canonical_rows: Any,
        *,
        block_seconds: float,
    ) -> None:
        if source_block_index in self._pending:
            raise ValueError("Base10 full-prefix source was already enqueued")
        self._pending[source_block_index] = Base10VideoPrefixUpdate(
            canonical_rows=canonical_rows,
            block_seconds=float(block_seconds),
        )

    def advance(self, block_index: int) -> tuple[int, ...]:
        cutoff = block_index - self.lag_blocks
        due = tuple(sorted(source for source in self._pending if source <= cutoff))
        for source in due:
            update = self._pending.pop(source)
            self._memory.update(
                update.canonical_rows,
                block_seconds=update.block_seconds,
            )
            self._visible_source_block_index = source
        return due

    def view(self) -> Any | None:
        return self._memory.view()

    def reset(self) -> None:
        """Release the stage-local EMA tensors after its final consumer."""

        self._memory.reset()
        self._pending.clear()
        self._visible_source_block_index = None

def build_base10_hiar_schedule(
    prompts: tuple[str, ...],
    seeds: tuple[int, ...],
    *, request_count: int = BASE10_HIAR_REQUEST_COUNT,
    request_seconds: float = BASE10_HIAR_REQUEST_SECONDS,
) -> Base10HiARSchedule:
    """Build the same canonical 5-second target geometry as direct inference."""

    if len(prompts) != BASE10_HIAR_REQUEST_COUNT or len(seeds) != len(prompts):
        raise ValueError("Base10 HiAR requires twelve aligned prompts and seeds")
    if not 1 <= request_count <= BASE10_HIAR_REQUEST_COUNT:
        raise ValueError("Base10 request_count must select an authored prefix")
    if request_seconds != BASE10_HIAR_REQUEST_SECONDS and request_count != 1:
        raise ValueError("custom Base10 duration requires a single request")
    base_plan = plan_duration(request_seconds)
    expected_phases = (
        5 if request_count == 1 and request_seconds == 175 / 24
        else 6 if request_count == 1 and request_seconds == 209 / 24
        else 7 if request_count == 1 and request_seconds == 243 / 24
        else BASE10_HIAR_CHUNKS_PER_REQUEST
    )
    if len(base_plan.phases) != expected_phases:
        raise RuntimeError("Base10 request geometry has an unexpected chunk count")

    phases: list[StreamPhase] = []
    requests: list[Base10HiARRequest] = []
    frame_offset = 0
    video_offset = 0
    audio_offset = 0
    sample_offset = 0
    for request_index, (prompt, seed) in enumerate(
        zip(prompts[:request_count], seeds[:request_count], strict=True)
    ):
        request_plan = (
            base_plan
            if request_index == 0
            else canonical_continuation_plan(base_plan, request_index=request_index)
        )
        phase_start = len(phases)
        for local in request_plan.phases:
            phases.append(
                replace(
                    local,
                    index=len(phases),
                    frame_start=local.frame_start + frame_offset,
                    frame_stop=local.frame_stop + frame_offset,
                    video_latent_start=local.video_latent_start + video_offset,
                    video_latent_stop=local.video_latent_stop + video_offset,
                    audio_latent_start=local.audio_latent_start + audio_offset,
                    audio_latent_stop=local.audio_latent_stop + audio_offset,
                    audio_sample_start=local.audio_sample_start + sample_offset,
                    audio_sample_stop=local.audio_sample_stop + sample_offset,
                )
            )
        requests.append(
            Base10HiARRequest(
                index=request_index,
                prompt=prompt,
                seed=seed,
                phase_start=phase_start,
                phase_stop=len(phases),
                frame_start=frame_offset,
                frame_stop=frame_offset + request_plan.native_frame_count,
                video_latent_start=video_offset,
                video_latent_stop=(
                    video_offset + request_plan.phases[-1].video_latent_stop
                ),
                audio_latent_start=audio_offset,
                audio_latent_stop=(
                    audio_offset + request_plan.phases[-1].audio_latent_stop
                ),
            )
        )
        frame_offset += request_plan.native_frame_count
        video_offset += request_plan.phases[-1].video_latent_stop
        audio_offset += request_plan.phases[-1].audio_latent_stop
        sample_offset += request_plan.phases[-1].audio_sample_stop

    return Base10HiARSchedule(
        plan=StreamPlan(native_frame_count=frame_offset, phases=tuple(phases)),
        requests=tuple(requests),
    )


def base10_teacher_path(root: Path, request_index: int) -> Path:
    return root / f"request_{request_index:02d}.pt"

def base10_teacher_receipt_path(root: Path, request_index: int) -> Path:
    return root / f"request_{request_index:02d}.json"

def write_base10_teacher_request(
    torch: Any,
    *,
    root: Path,
    request: Base10HiARRequest,
    milestones: Mapping[int, Any],
    teacher_receipt: Mapping[str, Any],
    producer_geometry: Mapping[str, int],
) -> Path:
    """Publish one request atomically so the six-GPU consumer can pipeline it."""

    expected_rows = 2 * request.audio_latent_count
    ordered = []
    for stage_index in range(BASE10_HIAR_STAGE_COUNT):
        value = milestones.get(stage_index)
        if value is None or tuple(value.shape) != (expected_rows, 32):
            raise RuntimeError("Base10 teacher milestone shape differs from request")
        ordered.append(value.detach().to(device="cpu", dtype=torch.float32))
    root.mkdir(parents=True, exist_ok=True)
    output = base10_teacher_path(root, request.index)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema": BASE10_TEACHER_SCHEMA,
            "request_index": request.index,
            "prompt": request.prompt,
            "seed": request.seed,
            "audio_latent_count": request.audio_latent_count,
            "teacher_state_numbers": list(BASE10_TEACHER_STATE_NUMBERS),
            "stage3_target_state_indices": list(
                BASE10_STAGE3_TARGET_STATE_INDICES
            ),
            "milestones": ordered,
        },
        temporary,
    )
    os.replace(temporary, output)
    receipt = base10_teacher_receipt_path(root, request.index)
    receipt_temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    receipt_temporary.write_text(
        json.dumps(
            {
                "schema": BASE10_TEACHER_SCHEMA,
                "request_index": request.index,
                "prompt": request.prompt,
                "seed": request.seed,
                "audio_latent_count": request.audio_latent_count,
                "producer_geometry": dict(producer_geometry),
                "teacher": dict(teacher_receipt),
                "published_at": time.time(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(receipt_temporary, receipt)
    return output

def wait_for_base10_teacher_request(
    torch: Any,
    *,
    root: Path,
    request: Base10HiARRequest,
) -> tuple[Any, Any, Any]:
    """Wait for and load the three request-level Base10 audio milestones."""

    path = base10_teacher_path(root, request.index)
    while not path.is_file():
        time.sleep(0.01)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema": BASE10_TEACHER_SCHEMA,
        "request_index": request.index,
        "prompt": request.prompt,
        "seed": request.seed,
        "audio_latent_count": request.audio_latent_count,
        "teacher_state_numbers": list(BASE10_TEACHER_STATE_NUMBERS),
        "stage3_target_state_indices": list(BASE10_STAGE3_TARGET_STATE_INDICES),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Base10 teacher request metadata differs")
    milestones = payload.get("milestones")
    expected_shape = (2 * request.audio_latent_count, 32)
    if (
        not isinstance(milestones, list)
        or len(milestones) != BASE10_HIAR_STAGE_COUNT
        or any(tuple(value.shape) != expected_shape for value in milestones)
    ):
        raise RuntimeError("Base10 teacher request milestones differ")
    return tuple(milestones)


def base10_teacher_phase_rows(
    teacher_rows: Any,
    *,
    request: Base10HiARRequest,
    phase: StreamPhase,
) -> Any:
    """Slice one channel-major request teacher state for a native chunk."""

    local_start = phase.audio_latent_start - request.audio_latent_start
    local_stop = phase.audio_latent_stop - request.audio_latent_start
    if not 0 <= local_start < local_stop <= request.audio_latent_count:
        raise ValueError("Base10 phase is outside its request audio geometry")
    return (
        teacher_rows.view(2, request.audio_latent_count, -1)[
            :, local_start:local_stop
        ]
        .contiguous()
        .view(-1, int(teacher_rows.shape[-1]))
    )

__all__ = [
    "BASE10_HIAR_CHUNKS_PER_REQUEST",
    "BASE10_HIAR_DIT_GPU_COUNT",
    "BASE10_HIAR_REQUEST_COUNT",
    "BASE10_HIAR_REQUEST_SECONDS",
    "BASE10_HIAR_STAGE_COUNT",
    "BASE10_HIAR_TEACHER_GPU_COUNT",
    "BASE10_HIAR_TP_SIZE",
    "BASE10_ROLLOVER_REFERENCE_LATENTS",
    "BASE10_STAGE3_TARGET_STATE_INDICES",
    "BASE10_TEACHER_SCHEMA",
    "BASE10_TEACHER_STATE_NUMBERS",
    "BASE10_VIDEO_RECENT_CHUNKS",
    "Base10DelayedVideoFullPrefixMemory",
    "Base10HiARRequest",
    "Base10HiARSchedule",
    "Base10HiARStageTeacher",
    "Base10VideoPrefixUpdate",
    "CompletedBase10TeacherArtifact",
    "base10_high_clean_video_prefix_update",
    "base10_hiar_contract",
    "base10_rollover_teacher_producer_topology",
    "base10_teacher_geometry",
    "base10_teacher_producer_topology",
    "base10_teacher_path",
    "base10_teacher_phase_rows",
    "build_base10_hiar_schedule",
    "load_base10_case",
    "wait_for_base10_teacher_request",
    "write_base10_teacher_request",
]
