"""Online generator LoRA with persistent clean AV-KV and whole-clip decode."""
from __future__ import annotations
import gc
import json
from fl2va.teacher import visual_contract

def run_student(args,case,schedule,active,transport,*,runtime=None,features=None,
                close_runtime=True,keep_transformer=False):
    import torch
    from digitalhuman_h3.runtime import H3Runtime
    from digitalhuman_h3.streaming.runtime import install_streaming_runtime
    import digitalhuman_h3.streaming.runtime as streaming_module
    from fl2va.policy import streaming_policy
    from digitalhuman_h3.inference.base10_hiar import CompletedBase10TeacherArtifact,base10_teacher_geometry
    from digitalhuman_h3.inference.acceleration import InferenceAccelerationConfig
    from fl2va.decode import generate_long_video_lean
    teacher = CompletedBase10TeacherArtifact.open(args.teacher_root,case_id=case['source_case_id'],schedule=schedule,
        consumer_geometry=base10_teacher_geometry(width=768,height=1376,video_latent_h=86,video_latent_w=48),
        visual_conditioning=visual_contract(case,schedule))
    runtime=runtime or H3Runtime(model_root=args.model_root,variant='fl2va',tp_size=2,ulysses_degree=4,
        adapter_checkpoint=args.checkpoint,allow_intermediate_checkpoint=True,denoise_schedule='distilled4',
        inference_acceleration=InferenceAccelerationConfig(streaming_attention_backend='sdpa'))
    assert runtime.keyframe_condition_policy == 'clip_absolute_persistent'
    runtime.model_resolution['generator_weights']={'generator_weights':'online'}
    policy=streaming_policy()
    original_decode=runtime.decode_video_latents
    def decode(latents,*,output_path):
        if not keep_transformer:
            runtime.pipeline.transformer.to('meta'); gc.collect(); torch.cuda.empty_cache()
        return original_decode(latents,output_path=output_path)
    if not keep_transformer:
        runtime.decode_video_latents=decode
    streaming_module.canonical_continuation_plan=lambda base_plan,*,request_index: active[request_index]
    metadata=dict(case_id=case['source_case_id'],source_manifest=case.get('source_manifest'),
        text_policy=case.get('text_policy'),checkpoint=str(args.checkpoint),generator_weights='online',task=case['task'],
        topology='tp2_sp4',audio_source='frozen_base10_joint_global_keyframes_tail40_not_student_audio',
        requested_seconds=args.seconds,native_frames=schedule.plan.native_frame_count,
        native_seconds=schedule.plan.native_frame_count/24,block_frames=[p.frame_count for p in schedule.plan.phases],
        request_count=len(schedule.requests),streaming_policy=policy.to_dict(),post_generation_crop=False,
        tail_visibility='last_request_only' if len(case['initial_conditions']) == 2 else 'none',
        tail_start_request_index=len(schedule.requests)-1 if len(case['initial_conditions']) == 2 else None,
        initial_conditions=case['initial_conditions'],teacher_root=str(args.teacher_root))
    with install_streaming_runtime(plan=transport[0],server_args=runtime.server_args,reuse_across_requests=True,
            full_prefix_half_life_seconds=None,policy=policy,attention_backend='sdpa',external_base_audio_teacher=teacher) as overlay:
        overlay.bind_clip_keyframe_deadline(end_frame=schedule.plan.native_frame_count,end_video_latent=schedule.video_latent_count)
        overlay.bind_clip_tail_start_request(len(schedule.requests)-1)
        result=generate_long_video_lean(runtime=runtime,run_dir=args.output_dir,
            segment_prompts=[r.prompt for r in schedule.requests],segment_seeds=[r.seed for r in schedule.requests],
            segment_audio_noise_seeds=[teacher.audio_noise_seed(i) for i in range(len(schedule.requests))],
            initial_conditions=case['initial_conditions'],aspect_ratio='9:16',run_metadata=metadata,
            expected_output_width=768,expected_output_height=1376,streaming_overlay=overlay,
            transport_plans=transport,
            clip_frames=schedule.plan.native_frame_count,precomputed_features=features)
        executions=[e.to_dict() for e in overlay.executions]
    if runtime.is_main_process:
        frames=[p['frame_stop']-p['frame_start'] for e in executions for p in e['phases']]
        assert frames == metadata['block_frames']
        if len(case['initial_conditions']) == 2:
            assert [e['clip_keyframe_visible_frame_indices'] for e in executions] == (
                [[0]]*(len(schedule.requests)-1)+[[0,-1]])
        (args.output_dir/'no_repeat_inference.json').write_text(json.dumps(dict(executions=executions,metadata=metadata),indent=2))
        (args.output_dir/'probe.complete.json').write_text(json.dumps(dict(**metadata,video=result['video']),indent=2))
    runtime.barrier()
    if close_runtime:
        torch.distributed.destroy_process_group()
