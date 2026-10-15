"""Frozen Base10 joint audio-video teacher."""
from __future__ import annotations
import gc
import json
from digitalhuman_h3.streaming.geometry import video_temporal_positions

def visual_contract(case, schedule):
    return dict(task='ii2va', initial_conditions=case['initial_conditions'],
        end_frame=schedule.plan.native_frame_count, end_video_latent=schedule.video_latent_count,
        prompt_encoding='text_only', video_context='joint_native_fl2va_global_keyframes_tail40',
        image_noise_aug=0.999,tail_visibility='last_request_only',
        tail_start_request_index=len(schedule.requests)-1)

def teacher_packed(torch, *, text_len, latent_h, latent_w, request, schedule):
    """Native joint AV layout plus the established frozen 40-latent audio prefix."""
    from digitalhuman_h3.model.packed_sequence import (
        minimax_h3_packed_sequence, minimax_h3_audio_only_frozen_prefix_packed_sequence)
    from digitalhuman_h3.model.architecture import MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
    packed = minimax_h3_packed_sequence(text_len=text_len, latent_t=request.video_latent_count,
        latent_h=latent_h, latent_w=latent_w, audio_t=request.audio_latent_count,
        include_keyframe_cond=True,
        keyframe_frame_indices=[0,-1] if request.index==len(schedule.requests)-1 else [0],
        frame_count=request.frame_count)
    frame_rows = (latent_h//2)*(latent_w//2)
    grid = packed['img_position_ids']
    cond_pos = packed['img_pos'][~packed['update_mask']]
    grid[cond_pos[:frame_rows],0] = text_len
    end_time = float(video_temporal_positions(schedule.video_latent_count, 1)[0]) - 5/3
    grid[cond_pos[frame_rows:],0] = text_len + end_time
    times = torch.tensor([float(t)+text_len for t in video_temporal_positions(
        request.video_latent_start, request.video_latent_count)], dtype=torch.float64)
    grid[packed['img_pos'][packed['update_mask']],0] = times.repeat_interleave(frame_rows)
    grid[packed['audio_pos'],0] += request.audio_latent_start
    if request.index:
        ref = minimax_h3_audio_only_frozen_prefix_packed_sequence(text_len=text_len,
            ref_audio_t=40, audio_t=request.audio_latent_count, latent_h=latent_h, latent_w=latent_w,
            reference_time_start=text_len+request.audio_latent_start-40,
            target_time_start=text_len+request.audio_latent_start)
        at = int(packed['audio_pos'][0]); used = int(packed['cu_seqlens'][1]); extra = 80
        size = ((used+extra+MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT-1)//MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT)*MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT
        new_grid = torch.zeros(size,3,dtype=torch.float64)
        new_grid[:at] = grid[:at]; new_grid[at:at+extra] = ref['img_position_ids'][ref['audio_pos'][:extra]]
        new_grid[at+extra:used+extra] = grid[at:used]
        tags = torch.full((size,),-1,dtype=torch.long)
        tags[:at] = packed['token_tags'][:at]; tags[at:at+extra] = 2
        tags[at+extra:used+extra] = packed['token_tags'][at:used]
        packed.update(seq_len=size, img_position_ids=new_grid, token_tags=tags,
            img_pos=packed['img_pos']+(packed['img_pos']>=at)*extra,
            audio_pos=torch.arange(at,at+extra+2*request.audio_latent_count),
            audio_update_mask=torch.cat((torch.zeros(extra,dtype=torch.bool),packed['audio_update_mask'])),
            cu_seqlens=torch.tensor([0,used+extra,size],dtype=torch.int32))
    return packed

def run_teacher(args, case, schedule, transport):
    import torch
    from digitalhuman_h3.runtime import H3Runtime
    from digitalhuman_h3.distributed import use_parallel_context
    from digitalhuman_h3.model.pipeline import _prepare_keyframes
    from digitalhuman_h3.model.geometry import resolve_geometry
    from digitalhuman_h3.model.denoise import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
    from digitalhuman_h3.denoise_schedule import select_time_shift_sigmas
    from digitalhuman_h3.inference.base10_hiar import (
        BASE10_KEYFRAME_STRATEGY, base10_keyframe_teacher_producer_topology,
        base10_teacher_geometry, write_base10_teacher_request)
    runtime = H3Runtime(model_root=args.model_root, variant='fl2va',tp_size=2,ulysses_degree=4)
    pipe = runtime.pipeline
    output = args.output_dir
    if runtime.is_main_process:
        output.mkdir(parents=True,exist_ok=False)
    runtime.barrier()
    features = runtime.precompute_text_prompts(tuple(r.prompt for r in schedule.requests))
    previous = None
    visual = visual_contract(case,schedule)
    with torch.inference_mode(), use_parallel_context(runtime.parallel_context):
        for request, base, feature in zip(schedule.requests,transport,features,strict=True):
            geom = resolve_geometry(duration_seconds=base.native_frame_count/24,width=768,height=1376)
            images, _ = _prepare_keyframes(case['initial_conditions'],geometry=geom)
            rng = torch.Generator(device='cpu').manual_seed(request.seed)
            cond = pipe._encode_keyframe_rows(images,geometry=geom,generator=rng)
            video, _ = pipe._initial_noise(geom,generator=rng)
            audio_rng = torch.Generator(device='cpu').manual_seed(request.seed)
            pipe._initial_video_noise(geom,generator=audio_rng)
            audio = pipe._initial_audio_noise(geom,generator=audio_rng)
            rows = (geom.video_latent_h//2)*(geom.video_latent_w//2)
            # Encode both images first to preserve the baseline RNG sequence.
            tail_visible = request.index == len(schedule.requests)-1
            if not tail_visible:
                cond = cond[:rows]
            video = video[(geom.video_latent_t-request.video_latent_count)*rows:]
            audio = audio.view(2,geom.audio_latent_t,-1)[:,-request.audio_latent_count:].contiguous().view(-1,32)
            ref = None if previous is None else previous.view(2,-1,32)[:,-40:].contiguous().view(-1,32)
            packed = teacher_packed(torch,text_len=feature[0].shape[0],latent_h=geom.video_latent_h,
                latent_w=geom.video_latent_w,request=request,schedule=schedule)
            packed['token_tags'][packed['text_pos']] = feature[1].view(-1)
            branch = MiniMaxH3DenoiseBranch(packed=packed,text_embeddings=feature[0],
                token_tags=packed['token_tags'],device=runtime.device,parallel_context=runtime.parallel_context)
            refined = pipe.transformer.refine_prompt_embeds(branch.static_kwargs['prompt_embeds'],
                branch.static_kwargs['refiner_packed_seq_params']['cu_seqlens_q'],device=runtime.device)
            branch.static_kwargs.update(prompt_embeds=refined,refined_prompt_embeds_length=int(refined.shape[0]))
            branch.static_kwargs['rope_cache'] = pipe.transformer.build_rope_cache(branch.static_kwargs['img_position_ids'],device=runtime.device)
            milestones = {}; forwards = 0
            def forward(model,kwargs,step):
                nonlocal forwards
                actual = kwargs['x'][0].index_select(0,branch.img_cond_seq_idx)
                if not torch.equal(actual,cond.to(device=actual.device,dtype=actual.dtype)):
                    raise RuntimeError('teacher keyframe rows missing at model forward')
                forwards += 1
                return model(**kwargs)
            def capture(step,_video,audio_rows):
                if step in (2,5,8):
                    milestones[(2,5,8).index(step)] = audio_rows[branch.audio_target_slice].detach().cpu().clone()
            _, final_audio = minimax_h3_denoise_loop(model=pipe.transformer,model_forward=forward,positive=branch,
                initial_video_rows=torch.cat((cond.to(video),video)),
                initial_audio_rows=audio if ref is None else torch.cat((ref.to(audio),audio)),
                keyframe_cond_rows=cond,audio_ref_rows=ref,
                sigmas_video=select_time_shift_sigmas(num_steps=10,shift_scale=12.0),
                sigmas_audio=select_time_shift_sigmas(num_steps=10,shift_scale=3.0),
                device=runtime.device,on_step=capture)
            previous = final_audio[branch.audio_target_slice].detach().cpu().clone()
            assert forwards == 9 and torch.equal(previous,milestones[2])
            assert all(torch.isfinite(t).all() for t in milestones.values())
            prefix_change = 0.0 if ref is None else (final_audio[:80].cpu()-ref).abs().max().item()
            assert prefix_change == 0
            receipt = dict(visual_conditioning=visual, conditioned_forward_count=forwards,
                keyframe_frame_indices=[0,-1] if tail_visible else [0],
                tail_reference_visible=tail_visible,
                tail_start_request_index=len(schedule.requests)-1,
                audio_noise_seed=request.seed,partition='fl2va',reference_latents_per_channel=0 if ref is None else 40,
                prefix_max_abs_change=prefix_change,active_milestones_are_exact_canonical_target=True,
                target_geometry='canonical_active',teacher_persistent_kv=False,
                packed_video_rows=int(branch.img_pos.numel()),packed_audio_rows=int(branch.audio_pos.numel()),
                video_projection_executed=True,video_output_head_executed=True,
                keyframe_rope_times=torch.unique_consecutive(packed['img_position_ids'][packed['img_pos'][~packed['update_mask']],0]).tolist(),
                frame_start=request.frame_start,frame_stop=request.frame_stop)
            producer_geometry=base10_teacher_geometry(width=768,height=1376,video_latent_h=86,video_latent_w=48)
            if runtime.is_main_process:
                write_base10_teacher_request(torch,root=output,request=request,milestones=milestones,
                    teacher_receipt=receipt,producer_geometry=producer_geometry)
                print(f'[teacher] {args.seconds}s request {request.index+1}/{len(schedule.requests)} complete',flush=True)
            del branch,packed,final_audio,video,audio,cond,refined,milestones
            gc.collect(); torch.cuda.empty_cache(); runtime.barrier()
    if runtime.is_main_process:
        (output/'complete.json').write_text(json.dumps(dict(schema='digitalhuman_h3.inference.base10_hiar_teacher_set.v1',
            case_id=case['source_case_id'],request_count=len(schedule.requests),base_precision='bf16',
            producer_topology=base10_keyframe_teacher_producer_topology(),producer_geometry=producer_geometry,
            strategy=BASE10_KEYFRAME_STRATEGY,partition='fl2va',visual_conditioning=visual,
            audio_noise_seed_sequence=[r.seed for r in schedule.requests],persistent_kv=False),indent=2))
    runtime.barrier(); torch.distributed.destroy_process_group()
