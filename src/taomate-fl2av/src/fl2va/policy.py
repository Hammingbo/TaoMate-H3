"""Recent-seven clean AV-KV with a frozen tail40 audio prefix."""
from digitalhuman_h3.streaming.policy import StreamingInferencePolicy


def streaming_policy():
    return StreamingInferencePolicy(
        max_prefix_blocks=7,
        cross_chunk_kv_mode='hybrid',
        cross_chunk_kv_recent_blocks=7,
        hybrid_prefix_prompt_mode='current',
        hybrid_prefix_audio_prompt_mode='current',
        cross_chunk_audio_kv_mode='match_video',
        cross_chunk_audio_kv_reset_windows=12,
        prefix_renorm=True,
        prefix_renorm_alpha=1.0,
        prefix_cond_source='current',
        dedupe_segment_speech=False,
        use_sink_block=True,
        cross_request_visual_handoff=False,
        cross_request_kv_boundary_reset=False,
        canonical_continuation_geometry=True,
        phase_latent_handoff_mode='off',
        full_prefix_enabled=False,
        base_audio_teacher=True,
        keyframe_condition_policy='clip_absolute_persistent',
    )
