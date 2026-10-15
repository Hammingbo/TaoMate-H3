"""Long-horizon H3 inference policy aligned with the proven LTX defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class StreamingInferencePolicy:
    """Semantic streaming controls shared by CLI inference and training eval."""

    causal_rope_type: str = "split"
    max_prefix_blocks: int = 2
    cross_chunk_kv_mode: str = "hybrid"
    cross_chunk_kv_recent_blocks: int = 2
    hybrid_prefix_prompt_mode: str = "current"
    hybrid_prefix_audio_prompt_mode: str = "current"
    cross_chunk_audio_kv_mode: str = "match_video"
    cross_chunk_audio_kv_reset_windows: int = 12
    base_audio_teacher: bool = False
    prefix_renorm: bool = True
    prefix_renorm_alpha: float = 1.0
    prefix_cond_source: str = "current"
    dedupe_segment_speech: bool = False
    use_sink_block: bool = True
    cross_request_visual_handoff: bool = False
    cross_request_kv_boundary_reset: bool = False
    canonical_continuation_geometry: bool = False
    phase_latent_handoff_mode: str = "off"
    full_prefix_enabled: bool = True
    direct_request_full_prefix_reuse: bool = False
    progressive_history_kv_mode: str = "clean_commit"
    progressive_history_kv_max_blocks: int = 5
    full_prefix_lag_blocks: int = 0
    keyframe_condition_policy: str = "first_block"

    def __post_init__(self) -> None:
        if self.keyframe_condition_policy not in {
            "first_block", "clip_absolute_persistent"
        }:
            raise ValueError("unsupported keyframe_condition_policy")
        if self.keyframe_condition_policy == "clip_absolute_persistent" and (
            self.cross_request_visual_handoff or self.phase_latent_handoff_mode != "off"
        ):
            raise ValueError("persistent clip keyframes cannot use latent handoff")
        if self.causal_rope_type != "split":
            raise ValueError("MiniMax-H3 streaming uses its native split t/h/w RoPE")
        if self.cross_chunk_kv_mode not in {"full", "sliding", "hybrid"}:
            raise ValueError("cross_chunk_kv_mode must be full, sliding, or hybrid")
        for label, value in (
            ("max_prefix_blocks", self.max_prefix_blocks),
            ("cross_chunk_kv_recent_blocks", self.cross_chunk_kv_recent_blocks),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.hybrid_prefix_prompt_mode != "current":
            raise ValueError("H3 hybrid history binds the current visual prompt")
        if self.hybrid_prefix_audio_prompt_mode != "current":
            raise ValueError("H3 hybrid history binds the current audio prompt")
        if self.cross_chunk_audio_kv_mode != "match_video":
            raise ValueError("cross_chunk_audio_kv_mode must be match_video")
        if (
            isinstance(self.cross_chunk_audio_kv_reset_windows, bool)
            or not isinstance(self.cross_chunk_audio_kv_reset_windows, int)
            or self.cross_chunk_audio_kv_reset_windows <= 0
        ):
            raise ValueError("cross_chunk_audio_kv_reset_windows must be positive")
        if not 0.0 <= float(self.prefix_renorm_alpha) <= 1.0:
            raise ValueError("prefix_renorm_alpha must be in [0, 1]")
        if self.prefix_cond_source != "current":
            raise ValueError("H3 streaming conditions each new chunk from its current prompt")
        if self.dedupe_segment_speech:
            raise ValueError("H3 keeps benchmark speech text unchanged between segments")
        if not self.use_sink_block and self.cross_chunk_kv_mode == "hybrid":
            raise ValueError("hybrid cross-chunk KV requires the persistent sink block")
        if self.cross_request_kv_boundary_reset:
            if not self.cross_request_visual_handoff:
                raise ValueError(
                    "cross-request KV boundary reset requires visual handoff"
                )
            if self.full_prefix_enabled:
                raise ValueError(
                    "cross-request KV boundary reset requires full-prefix off"
                )
        if self.phase_latent_handoff_mode not in {
            "off",
            "first_transition",
            "all_transitions",
        }:
            raise ValueError(
                "phase_latent_handoff_mode must be off, first_transition, or "
                "all_transitions"
            )
        if self.progressive_history_kv_mode not in {
            "clean_commit",
            "stage_input",
            "dual_clean",
        }:
            raise ValueError(
                "progressive_history_kv_mode must be clean_commit, stage_input, or dual_clean"
            )
        if (
            isinstance(self.progressive_history_kv_max_blocks, bool)
            or not isinstance(self.progressive_history_kv_max_blocks, int)
            or self.progressive_history_kv_max_blocks < 1
        ):
            raise ValueError(
                "progressive_history_kv_max_blocks must be a positive integer"
            )
        if (
            isinstance(self.full_prefix_lag_blocks, bool)
            or not isinstance(self.full_prefix_lag_blocks, int)
            or self.full_prefix_lag_blocks < 0
        ):
            raise ValueError("full_prefix_lag_blocks must be a non-negative integer")
        if self.progressive_history_kv_mode != "stage_input":
            if self.full_prefix_lag_blocks:
                raise ValueError(
                    "full_prefix_lag_blocks requires progressive stage-input KV"
                )
            if self.progressive_history_kv_mode == "dual_clean" and (
                self.full_prefix_enabled or self.phase_latent_handoff_mode != "off"
            ):
                raise ValueError("dual_clean requires full-prefix and latent handoff off")
        else:
            if self.phase_latent_handoff_mode != "off":
                raise ValueError(
                    "progressive stage-input KV requires phase latent handoff off"
                )
            if self.prefix_renorm:
                raise ValueError(
                    "progressive stage-input KV requires prefix renormalization off"
                )
            if not self.full_prefix_enabled:
                raise ValueError(
                    "progressive stage-input KV requires full-prefix memory"
                )

    @property
    def recent_prefix_blocks(self) -> int:
        return min(self.max_prefix_blocks, self.cross_chunk_kv_recent_blocks)

    def uses_phase_latent_handoff(self, phase_index: int) -> bool:
        if phase_index <= 0 or self.phase_latent_handoff_mode == "off":
            return False
        if self.phase_latent_handoff_mode == "first_transition":
            return phase_index == 1
        return True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def add_streaming_policy_arguments(parser: Any) -> None:
    """Expose LTX-compatible spellings on every H3 streaming entrypoint."""

    parser.add_argument(
        "--keyframe-condition-policy",
        choices=("first_block", "clip_absolute_persistent"),
        default=None,
        help="Keep first/last II2VA references visible at fixed whole-clip times.",
    )
    parser.add_argument(
        "--causal_rope_type",
        "--causal-rope-type",
        choices=("split",),
        default="split",
    )
    parser.add_argument(
        "--max_prefix_blocks", "--max-prefix-blocks", type=int, default=2
    )
    parser.add_argument(
        "--cross_chunk_kv_mode",
        "--cross-chunk-kv-mode",
        choices=("full", "sliding", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--cross_chunk_kv_recent_blocks",
        "--cross-chunk-kv-recent-blocks",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--hybrid_prefix_prompt_mode",
        "--hybrid-prefix-prompt-mode",
        choices=("current",),
        default="current",
    )
    parser.add_argument(
        "--hybrid_prefix_audio_prompt_mode",
        "--hybrid-prefix-audio-prompt-mode",
        choices=("current",),
        default="current",
    )
    parser.add_argument(
        "--cross_chunk_audio_kv_mode",
        "--cross-chunk-audio-kv-mode",
        choices=("match_video",),
        default="match_video",
        help="Keep audio and video history under the same sink-plus-recent policy.",
    )
    parser.add_argument(
        "--cross_chunk_audio_kv_reset_windows",
        "--cross-chunk-audio-kv-reset-windows",
        type=int,
        default=12,
    )
    parser.set_defaults(base_audio_teacher=False)
    parser.add_argument(
        "--base_audio_teacher",
        "--base-audio-teacher",
        dest="base_audio_teacher",
        action="store_true",
        help=(
            "Generate one whole-request Base H3 audio trajectory and use its "
            "Base10 state 3/6/9 latents to supervise every streaming chunk."
        ),
    )
    parser.add_argument(
        "--no_base_audio_teacher",
        "--no-base-audio-teacher",
        dest="base_audio_teacher",
        action="store_false",
    )
    parser.set_defaults(prefix_renorm=True)
    parser.add_argument(
        "--prefix_renorm",
        "--prefix-renorm",
        dest="prefix_renorm",
        action="store_true",
    )
    parser.add_argument(
        "--no_prefix_renorm",
        "--no-prefix-renorm",
        dest="prefix_renorm",
        action="store_false",
    )
    parser.add_argument(
        "--prefix_renorm_alpha",
        "--prefix-renorm-alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--prefix_cond_source",
        "--prefix-cond-source",
        choices=("current",),
        default="current",
    )
    parser.set_defaults(dedupe_segment_speech=False)
    parser.add_argument(
        "--no_dedupe_segment_speech",
        "--no-dedupe-segment-speech",
        dest="dedupe_segment_speech",
        action="store_false",
    )
    parser.set_defaults(use_sink_block=True)
    parser.add_argument(
        "--use_sink_block",
        "--use-sink-block",
        dest="use_sink_block",
        action="store_true",
    )
    parser.add_argument(
        "--no_use_sink_block",
        "--no-use-sink-block",
        dest="use_sink_block",
        action="store_false",
    )
    parser.set_defaults(cross_request_visual_handoff=False)
    parser.add_argument(
        "--cross_request_visual_handoff",
        "--cross-request-visual-handoff",
        dest="cross_request_visual_handoff",
        action="store_true",
        help=(
            "Condition each continuation request on the previous request's "
            "terminal clean video latent while retaining persistent AV-KV."
        ),
    )
    parser.add_argument(
        "--no_cross_request_visual_handoff",
        "--no-cross-request-visual-handoff",
        dest="cross_request_visual_handoff",
        action="store_false",
    )
    parser.set_defaults(cross_request_kv_boundary_reset=False)
    parser.add_argument(
        "--cross_request_kv_boundary_reset",
        "--cross-request-kv-boundary-reset",
        dest="cross_request_kv_boundary_reset",
        action="store_true",
        help=(
            "At each visual-handoff request boundary, discard prior AV-KV while "
            "keeping the terminal clean video latent and global media timeline."
        ),
    )
    parser.add_argument(
        "--no_cross_request_kv_boundary_reset",
        "--no-cross-request-kv-boundary-reset",
        dest="cross_request_kv_boundary_reset",
        action="store_false",
    )
    parser.set_defaults(canonical_continuation_geometry=False)
    parser.add_argument(
        "--canonical_continuation_geometry",
        "--canonical-continuation-geometry",
        dest="canonical_continuation_geometry",
        action="store_true",
        help=(
            "Generate only steady-state continuation targets after request zero; "
            "requires persistent clean AV-KV and one-shot AV publication."
        ),
    )
    parser.add_argument(
        "--no_canonical_continuation_geometry",
        "--no-canonical-continuation-geometry",
        dest="canonical_continuation_geometry",
        action="store_false",
    )
    parser.add_argument(
        "--phase_latent_handoff_mode",
        "--phase-latent-handoff-mode",
        choices=("off", "first_transition", "all_transitions"),
        default=None,
    )
    parser.add_argument(
        "--spatial_anchor",
        "--spatial-anchor",
        dest="phase_latent_handoff_mode",
        action="store_const",
        const="all_transitions",
    )
    parser.add_argument(
        "--no_spatial_anchor",
        "--no-spatial-anchor",
        dest="phase_latent_handoff_mode",
        action="store_const",
        const="off",
    )
    parser.set_defaults(full_prefix_enabled=True)
    parser.add_argument(
        "--full_prefix",
        "--full-prefix",
        dest="full_prefix_enabled",
        action="store_true",
    )
    parser.add_argument(
        "--no_full_prefix",
        "--no-full-prefix",
        dest="full_prefix_enabled",
        action="store_false",
    )
    parser.set_defaults(direct_request_full_prefix_reuse=False)
    parser.add_argument(
        "--direct_request_full_prefix_reuse",
        "--direct-request-full-prefix-reuse",
        dest="direct_request_full_prefix_reuse",
        action="store_true",
        help=(
            "Reuse one encoded full-prefix K/V inside each direct-resolution "
            "request instead of re-encoding the slowly changing latent EMA for "
            "every internal chunk."
        ),
    )
    parser.add_argument(
        "--progressive_history_kv_mode",
        "--progressive-history-kv-mode",
        choices=("clean_commit", "stage_input", "dual_clean"),
        default="clean_commit",
    )
    parser.add_argument(
        "--progressive_history_kv_max_blocks",
        "--progressive-history-kv-max-blocks",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--full_prefix_lag_blocks",
        "--full-prefix-lag-blocks",
        type=int,
        default=0,
    )


def streaming_policy_from_args(
    args: Any,
    *,
    phase_latent_handoff_default: str = "off",
    keyframe_condition_default: str = "first_block",
) -> StreamingInferencePolicy:
    phase_latent_handoff_mode = getattr(args, "phase_latent_handoff_mode", None)
    if phase_latent_handoff_mode is None:
        phase_latent_handoff_mode = phase_latent_handoff_default
    return StreamingInferencePolicy(
        causal_rope_type=args.causal_rope_type,
        max_prefix_blocks=args.max_prefix_blocks,
        cross_chunk_kv_mode=args.cross_chunk_kv_mode,
        cross_chunk_kv_recent_blocks=args.cross_chunk_kv_recent_blocks,
        hybrid_prefix_prompt_mode=args.hybrid_prefix_prompt_mode,
        hybrid_prefix_audio_prompt_mode=args.hybrid_prefix_audio_prompt_mode,
        cross_chunk_audio_kv_mode=args.cross_chunk_audio_kv_mode,
        cross_chunk_audio_kv_reset_windows=(
            args.cross_chunk_audio_kv_reset_windows
        ),
        base_audio_teacher=args.base_audio_teacher,
        prefix_renorm=args.prefix_renorm,
        prefix_renorm_alpha=args.prefix_renorm_alpha,
        prefix_cond_source=args.prefix_cond_source,
        dedupe_segment_speech=args.dedupe_segment_speech,
        use_sink_block=args.use_sink_block,
        cross_request_visual_handoff=args.cross_request_visual_handoff,
        cross_request_kv_boundary_reset=getattr(
            args, "cross_request_kv_boundary_reset", False
        ),
        canonical_continuation_geometry=args.canonical_continuation_geometry,
        phase_latent_handoff_mode=phase_latent_handoff_mode,
        full_prefix_enabled=args.full_prefix_enabled,
        direct_request_full_prefix_reuse=(
            args.direct_request_full_prefix_reuse
        ),
        progressive_history_kv_mode=args.progressive_history_kv_mode,
        progressive_history_kv_max_blocks=(
            args.progressive_history_kv_max_blocks
        ),
        full_prefix_lag_blocks=args.full_prefix_lag_blocks,
        keyframe_condition_policy=(
            getattr(args, "keyframe_condition_policy", None) or keyframe_condition_default
        ),
    )


def phase_latent_handoff_default_for_resolution(
    checkpoint_default: str,
    *,
    is_progressive: bool,
) -> str:
    """Use checkpoint phase handoff only on progressive-resolution requests."""

    return checkpoint_default if is_progressive else "off"


__all__ = [
    "StreamingInferencePolicy",
    "add_streaming_policy_arguments",
    "phase_latent_handoff_default_for_resolution",
    "streaming_policy_from_args",
]
