"""Stage-local predicted-clean KV, fused with the following LOW denoise."""
from contextlib import contextmanager
import os

from .cache import CleanAVKVCache, kv_contract_from_model
from .capability import set_official_attention_hook
from .session import H3StreamingSession


class ProgressivePredictedHistory:
    def __init__(self, model, *, owned_stage=None):
        self.owned_stages = (0, 1) if owned_stage is None else ((owned_stage,) if owned_stage < 2 else ())
        self.caches = tuple(CleanAVKVCache(kv_contract_from_model(model)) for _ in range(2))
        self.sessions = ()
        self.video_rows = None
        self.audio_rows = None
        self.pending = {}
        self.pending_audio_callbacks = {}
        self.deferred = {}
        self.early_send = (
            owned_stage in (0, 1)
            and os.environ.get("H3_PROGRESSIVE_LOW_EARLY_SEND") == "1"
        )
        self.fused = os.environ.get("H3_PROGRESSIVE_FUSED_PREDICTED_KV") == "1"
        self.fused_forwards = 0
        self.separate_clean_forwards = 0

    def begin_request(self, *, owner, plan, model, total_video_latents):
        for cache in self.caches:
            owner._apply_request_boundary_audio_policy(cache, total_video_latents=total_video_latents)
        self.sessions = tuple(
            H3StreamingSession(
                plan, model=model, cache=cache, attention_backend=owner.attention_backend,
                fused_history_use_sink=owner.policy.use_sink_block,
                fused_history_recent_blocks=owner.policy.cross_chunk_kv_recent_blocks,
            )
            for cache in self.caches
        )
        self.pending.clear()
        self.pending_audio_callbacks.clear()

    def forward(self, stage, phase, *, owner, branch, model, call_kwargs,
                model_forward, metadata, restore_hook, full_prefix,
                video_sigmas, audio_sigmas, condition_timestep):
        previous = self.deferred.get(stage)
        if previous is None:
            with self.noisy(stage, phase, restore_hook=restore_hook, full_prefix=full_prefix), metadata(branch):
                return model(**call_kwargs) if model_forward is None else model_forward(model, call_kwargs, stage)

        import torch
        from .fused_documents import (
            build_fused_denoise_branch, current_document_velocities,
            fused_document_media_rows, fused_document_timestep_plan,
        )

        fused = build_fused_denoise_branch(
            torch=torch, previous=previous["branch"], current=branch,
            model=model, device=self.video_rows.device,
            static_runtime_plans=os.environ.get("H3_PROGRESSIVE_FUSED_STATIC_PLANS") == "1",
        )
        video, audio = fused_document_media_rows(
            torch, branch=fused,
            previous_video_rows=previous["video"], current_video_rows=self.video_rows,
            previous_audio_rows=previous["audio"], current_audio_rows=self.audio_rows,
        )
        timestep = fused_document_timestep_plan(
            torch, branch=fused, previous_video_timestep=1.0, previous_audio_timestep=1.0,
            current_video_timestep=1.0-float(video_sigmas[stage]),
            current_audio_timestep=1.0-float(audio_sigmas[stage]),
            previous_condition_timestep=previous["condition_timestep"],
            current_condition_timestep=condition_timestep,
        )
        fused_kwargs = fused.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=timestep)
        session = self.sessions[stage]
        set_official_attention_hook(None)
        try:
            with session, session.fused_clean_commit_denoise(
                previous["phase"], phase,
                previous_full_prefix=previous["full_prefix"], current_full_prefix=full_prefix,
                denoise_step_index=stage,
            ), metadata(fused, live_documents_hint=(0, 2)):
                velocity = model(**fused_kwargs) if model_forward is None else model_forward(model, fused_kwargs, stage)
            owner._apply_cross_chunk_cache_policy(session.cache)
        finally:
            set_official_attention_hook(restore_hook)
        self.deferred.pop(stage)
        self.fused_forwards += 1
        return current_document_velocities(fused, *velocity)

    @contextmanager
    def noisy(self, stage, phase, *, restore_hook, full_prefix=None):
        session = self.sessions[stage]
        set_official_attention_hook(None)
        try:
            with session, session.noisy_step(phase, full_prefix=full_prefix, denoise_step_index=stage):
                yield
        finally:
            set_official_attention_hook(restore_hook)

    def initial(self, video, audio):
        self.video_rows, self.audio_rows = video, audio

    def capture(self, stage, video_velocity, audio_velocity, *, branch, video_sigmas, audio_sigmas, clean_audio_callback, step_offset=0):
        stage += step_offset
        # Native H3 velocity has sign x0 - noise: x0 = x_sigma + sigma*v.
        video = self.video_rows.detach().float().clone()
        audio = self.audio_rows.detach().float().clone()
        video[branch.video_target_slice].add_(video_velocity.float(), alpha=float(video_sigmas[stage]))
        audio[branch.audio_target_slice].add_(audio_velocity[branch.audio_target_slice].float(), alpha=float(audio_sigmas[stage]))
        if clean_audio_callback is not None:
            if self.early_send:
                # This audio is only for this stage's future history, not the
                # Euler video sent downstream. Resolve state9 after that send.
                self.pending_audio_callbacks[stage] = clean_audio_callback
            else:
                clean_audio_callback(2, video, audio)
        self.pending[stage] = video, audio

    def commit(self, *, owner, phase, branch, model, restore_hook, full_prefix, condition_timestep, model_forward, metadata):
        import torch

        timestep = branch.prepare_timestep_plan(
            video_timesteps=[1.0], audio_timesteps=[1.0],
            imgvid_cond_noise_aug=condition_timestep, audio_ref_cond_noise_aug=1.0,
        )[0]
        for stage in self.owned_stages:
            session = self.sessions[stage]
            video, audio = self.pending.pop(stage)
            clean_audio_callback = self.pending_audio_callbacks.pop(stage, None)
            if clean_audio_callback is not None:
                clean_audio_callback(2, video, audio)
            if self.fused and phase < len(session.plan.phases)-1:
                self.deferred[stage] = {
                    "video": video, "audio": audio, "branch": branch, "phase": phase,
                    "condition_timestep": condition_timestep, "full_prefix": full_prefix,
                }
                continue
            kwargs = branch.forward_kwargs(video_rows=video, audio_rows=audio, step_timesteps=timestep)
            set_official_attention_hook(None)
            try:
                with session, session.clean_commit(phase, full_prefix=full_prefix), metadata(branch), torch.no_grad():
                    if model_forward is None:
                        model(**kwargs)
                    else:
                        model_forward(model, kwargs, 3)
                owner._apply_cross_chunk_cache_policy(session.cache)
                self.separate_clean_forwards += 1
            finally:
                set_official_attention_hook(restore_hook)

    def separate_clean_count(self, phase, last_phase):
        if self.fused and phase < last_phase:
            return 0
        return len(self.owned_stages)

    def release(self):
        for cache in self.caches:
            cache.clear()
        self.pending.clear()
        self.pending_audio_callbacks.clear()
        self.deferred.clear()
        self.sessions = ()
        self.video_rows = self.audio_rows = None
