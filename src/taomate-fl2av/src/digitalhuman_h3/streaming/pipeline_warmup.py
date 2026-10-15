"""Model-resident first-shape warmup with a private, discarded streaming cache."""
import json
import os
from pathlib import Path
import time


def warm_stage(*, owner, model, branch, phase, active_plan, video, audio,
               video_sigmas, audio_sigmas, original_loop, loop_module,
               model_forward, directory):
    import torch
    from .runtime import _native_attention_metadata
    from .session import H3StreamingSession

    stage = int(os.environ['H3_PROGRESSIVE_PP_STAGE'])
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    records = []
    try:
        for repetition in range(2):
            # A fresh session owns all warmup KV. Neither history nor generated
            # latents can enter the real request's streaming overlay.
            session = H3StreamingSession(
                active_plan, model=model, server_args=owner.server_args,
                attention_backend=owner.attention_backend,
                sol_attn_tau=owner.sol_attn_tau,
                sol_attn_dense_layers=owner.sol_attn_dense_layers,
                sol_attn_dense_denoise_steps=owner.sol_attn_dense_denoise_steps,
                sol_attn_route_current_video=owner.sol_attn_route_current_video,
            )
            def forward(active_model, call_kwargs, step_index):
                with session.noisy_step(phase.index, denoise_step_index=stage), _native_attention_metadata(branch):
                    return active_model(**call_kwargs) if model_forward is None else model_forward(active_model, call_kwargs, stage)
            torch.cuda.synchronize()
            started = time.time()
            with session, torch.no_grad():
                result_video, result_audio = original_loop(
                    model=model, model_forward=forward, positive=branch,
                    initial_video_rows=video.clone(), initial_audio_rows=audio.clone(),
                    keyframe_cond_rows=None, audio_ref_rows=None,
                    sigmas_video=video_sigmas, sigmas_audio=audio_sigmas,
                    device=torch.device('cuda'),
                )
                if stage == 2:
                    timestep = branch.prepare_timestep_plan(
                        video_timesteps=[1.0], audio_timesteps=[1.0],
                        imgvid_cond_noise_aug=1.0, audio_ref_cond_noise_aug=1.0,
                    )[0]
                    call_kwargs = branch.forward_kwargs(video_rows=result_video, audio_rows=result_audio, step_timesteps=timestep)
                    call_kwargs['streaming_cache_only'] = True
                    with session.clean_commit(phase.index), _native_attention_metadata(branch):
                        model(**call_kwargs) if model_forward is None else model_forward(model, call_kwargs, 3)
            torch.cuda.synchronize()
            records.append(dict(repetition=repetition, start_epoch=started, end_epoch=time.time()))
            session.cache.clear()
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    rank = torch.distributed.get_rank()
    (path / f'warmup_stage{stage}_rank{rank}.json').write_text(json.dumps(records, indent=2))
    if rank == 0:
        print(json.dumps(dict(status='dit_warmup_complete', stage=stage,
                              seconds=[r['end_epoch']-r['start_epoch'] for r in records])), flush=True)
