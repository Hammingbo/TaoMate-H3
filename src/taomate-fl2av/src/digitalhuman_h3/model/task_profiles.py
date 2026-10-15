# SPDX-License-Identifier: Apache-2.0
"""Native MiniMax H3 request profiles without a serving-framework dependency."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


MINIMAX_H3_TASK_T2VA = "t2va"
MINIMAX_H3_TASK_FL2VA = "fl2va"
MINIMAX_H3_TASK_REF2VA = "ref2va"
MINIMAX_H3_CONDITION_ROLE_KEYFRAME = "keyframe"
MINIMAX_H3_CONDITION_ROLE_REFERENCE = "reference"
MINIMAX_H3_DEFAULT_BRANCHES: tuple[dict[str, str], ...] = ({"name": "cond_1"},)
MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES: tuple[tuple[int, ...], ...] = (
    (0,),
    (-1,),
    (0, -1),
)
MINIMAX_H3_TASK_PARTITIONS = MappingProxyType(
    {"t2va": "fl2va", "fl2va": "fl2va", "ref2va": "ref2va"}
)


@dataclass(frozen=True)
class MiniMaxH3ConditionRule:
    role: str
    condition_type: str
    material_chain: str
    requires_frame_index: bool = False
    visual_tokenizer_encode: bool = False
    audio_tokenizer_encode: bool = False


@dataclass(frozen=True)
class MiniMaxH3TaskProfile:
    task: str
    conditions_required: bool
    condition_rules: tuple[MiniMaxH3ConditionRule, ...]
    branches: tuple[dict[str, str], ...]
    default_flow_shift: float
    default_audio_flow_shift: float
    required_components: tuple[str, ...]
    geometry_source: str
    duration_from_audio_reference: bool = False
    min_condition_count: int | None = None
    max_condition_count: int | None = None
    auto_aspect_ratio: str | None = None
    auto_geometry_source: str | None = None

    def rule_for(self, *, role: str, condition_type: str) -> MiniMaxH3ConditionRule:
        for rule in self.condition_rules:
            if rule.role == role and rule.condition_type == condition_type:
                return rule
        raise ValueError(
            f"task {self.task!r} does not allow condition "
            f"role={role!r} type={condition_type!r}"
        )


_BASE_COMPONENTS = (
    "processor",
    "text_encoder",
    "tokenizer",
    "transformer",
    "video_vae",
    "audio_vae",
)
_BRANCHES = tuple(dict(branch) for branch in MINIMAX_H3_DEFAULT_BRANCHES)
MINIMAX_H3_TASK_PROFILES: dict[str, MiniMaxH3TaskProfile] = {
    MINIMAX_H3_TASK_T2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_T2VA,
        conditions_required=False,
        condition_rules=(),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        geometry_source="explicit_target",
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
    ),
    MINIMAX_H3_TASK_FL2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_FL2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_KEYFRAME,
                condition_type="image",
                material_chain="image.target_canvas",
                requires_frame_index=True,
                visual_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        geometry_source="explicit_target",
        min_condition_count=1,
        max_condition_count=2,
        auto_geometry_source="first_keyframe",
    ),
    MINIMAX_H3_TASK_REF2VA: MiniMaxH3TaskProfile(
        task=MINIMAX_H3_TASK_REF2VA,
        conditions_required=True,
        condition_rules=(
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="image",
                material_chain="image.reference_preserve",
                visual_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video",
                material_chain="video.reference_preserve",
                visual_tokenizer_encode=True,
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="video_audio",
                material_chain="video_audio.reference_preserve",
                visual_tokenizer_encode=True,
                audio_tokenizer_encode=True,
            ),
            MiniMaxH3ConditionRule(
                role=MINIMAX_H3_CONDITION_ROLE_REFERENCE,
                condition_type="audio",
                material_chain="audio",
                audio_tokenizer_encode=True,
            ),
        ),
        branches=_BRANCHES,
        default_flow_shift=12.0,
        default_audio_flow_shift=3.0,
        required_components=_BASE_COMPONENTS,
        geometry_source="explicit_target",
        duration_from_audio_reference=True,
        auto_aspect_ratio="16:9",
        auto_geometry_source="policy_default",
    ),
}


def partition_for_task(task: str) -> str:
    normalized = task.strip().lower() if isinstance(task, str) else ""
    if not normalized:
        raise ValueError("MiniMax H3 task must be a non-empty string")
    try:
        return MINIMAX_H3_TASK_PARTITIONS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax H3 task {task!r}") from exc


def minimax_h3_task_profile(task: str) -> MiniMaxH3TaskProfile:
    try:
        return MINIMAX_H3_TASK_PROFILES[task]
    except KeyError as exc:
        raise ValueError(
            f"unknown minimax_h3 task {task!r}; supported: "
            f"{sorted(MINIMAX_H3_TASK_PROFILES)}"
        ) from exc


__all__ = [
    "MINIMAX_H3_CONDITION_ROLE_KEYFRAME",
    "MINIMAX_H3_CONDITION_ROLE_REFERENCE",
    "MINIMAX_H3_DEFAULT_BRANCHES",
    "MINIMAX_H3_FL2VA_KEYFRAME_SIGNATURES",
    "MINIMAX_H3_TASK_FL2VA",
    "MINIMAX_H3_TASK_PARTITIONS",
    "MINIMAX_H3_TASK_PROFILES",
    "MINIMAX_H3_TASK_REF2VA",
    "MINIMAX_H3_TASK_T2VA",
    "MiniMaxH3ConditionRule",
    "MiniMaxH3TaskProfile",
    "minimax_h3_task_profile",
    "partition_for_task",
]
