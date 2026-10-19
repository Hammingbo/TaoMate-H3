"""Continuous FL2VA requests with whole-clip audio and video decoding."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def native_mux_output_args(final_path: Path, frames: int) -> list[str]:
    from digitalhuman_h3.inference.long_video import _final_mux_output_args
    args = list(_final_mux_output_args(final_path, expected_seconds=frames / 24))
    args[1] = str(frames)
    # Represent both 24 fps and 32 kHz exactly instead of rounding to milliseconds.
    args[-1:-1] = ['-movie_timescale', '96000']
    return args


def generate_long_video_lean(
    *,
    runtime: Any,
    run_dir: Path,
    segment_prompts: Sequence[str],
    segment_seeds: Sequence[int],
    segment_audio_noise_seeds: Sequence[int],
    initial_conditions: Sequence[Mapping[str, Any]],
    aspect_ratio: str,
    run_metadata: Mapping[str, Any],
    expected_output_width: int,
    expected_output_height: int,
    streaming_overlay: Any,
    transport_plans: Sequence[Any],
    clip_frames: int,
    precomputed_features=None,
) -> dict[str, Any]:
    import torch

    from digitalhuman_h3.inference.long_video import (
        _audio_sample_count,
        _binary,
        _probe,
        _rank_zero_action,
        _rank_zero_value,
        _run,
    )
    from digitalhuman_h3.inference.media_timing import exact_audio_delivery_filter

    segment_count = len(transport_plans)
    if (
        len(segment_prompts) != segment_count
        or len(segment_seeds) != segment_count
        or len(initial_conditions) > 2
        or len(segment_audio_noise_seeds) != segment_count
        or not streaming_overlay.policy.canonical_continuation_geometry
    ):
        raise ValueError("long-tail inference requires explicit transport plans and canonical one-shot AV decode")

    if len(initial_conditions) == 2:
        if streaming_overlay.policy.keyframe_condition_policy != "clip_absolute_persistent":
            raise ValueError("II2VA requires the whole-clip persistent keyframe policy")
        # The caller binds the exact variable-length whole-clip endpoint.

    started = time.perf_counter()
    run_dir = Path(run_dir)
    segments_dir = run_dir / "segments"

    def setup() -> None:
        run_dir.mkdir(parents=True, exist_ok=False)
        segments_dir.mkdir()

    _rank_zero_action(runtime, "lean output setup", setup)
    features = (runtime.precompute_text_prompts(tuple(segment_prompts))
                if precomputed_features is None else precomputed_features)
    if len(features) != segment_count:
        raise ValueError('precomputed text feature count differs from segment count')
    videos: list[Any] = []
    audios: list[Any] = []
    requests: list[dict[str, Any]] = []
    for index, (segment_prompt, segment_seed, feature) in enumerate(
        zip(segment_prompts, segment_seeds, features, strict=True)
    ):
        streaming_overlay.plan = transport_plans[index]
        request_seconds = transport_plans[index].native_frame_count / 24
        conditions = list(initial_conditions) if index == 0 else []
        task = ("t2va", "i2va", "ii2va")[len(conditions)]
        native_task = "fl2va" if conditions else "t2va"
        request = {
            "request_index": index,
            "prompt": segment_prompt,
            "seed": int(segment_seed),
            "task": task,
            "native_task": native_task,
            "conditions": conditions,
            "target": {
                "duration_seconds": request_seconds,
                "effective_model_width": expected_output_width,
                "effective_model_height": expected_output_height,
                "aspect_ratio": aspect_ratio,
            },
            "run_metadata": dict(run_metadata),
        }
        _rank_zero_action(
            runtime,
            f"request {index} record",
            lambda: _write_json(segments_dir / f"segment_{index:02d}.request.json", request),
        )
        request_started = time.perf_counter()
        result = runtime.pipeline.generate(
            prompt=segment_prompt,
            seed=int(segment_seed),
            task=native_task,
            conditions=conditions,
            duration_seconds=float(request_seconds),
            aspect_ratio=aspect_ratio,
            width=expected_output_width,
            height=expected_output_height,
            num_inference_steps=50,
            sigma_state_indices=runtime.inference_sigma_state_indices,
            flow_shift=12.0,
            audio_flow_shift=3.0,
            trajectory_only=True,
            output_path=None,
            precomputed_text=feature,
            clean_keyframe_rows=None,
            decode_audio=False,
            audio_noise_seed=int(segment_audio_noise_seeds[index]),
        )
        video = result.video_latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
        audio = result.audio_latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
        runtime._last_clean_video_latents = video
        runtime._last_clean_audio_latents = audio
        execution = streaming_overlay.last_execution
        if execution.request_index != index or not execution.canonical_continuation_geometry:
            raise RuntimeError("canonical execution receipt is absent or out of order")
        video_transport = int(execution.video_transport_prefix_latents)
        audio_transport = int(execution.audio_transport_prefix_latents_per_channel)
        if (
            video_transport + int(execution.published_video_latents) != video.shape[2]
            or audio_transport + int(execution.published_audio_latents_per_channel) != audio.shape[2]
        ):
            raise RuntimeError("canonical transport and published latent counts differ")
        request.update(
            raw_video_latent_shape=list(video.shape),
            raw_audio_latent_shape=list(audio.shape),
            video_transport_prefix_latents=video_transport,
            audio_transport_prefix_latents_per_channel=audio_transport,
        )
        videos.append(video[:, :, video_transport:, ...].contiguous())
        audios.append(audio[:, :, audio_transport:].contiguous())
        request.update(
            published_video_latent_shape=list(videos[-1].shape),
            published_audio_latent_shape=list(audios[-1].shape),
            elapsed_seconds=time.perf_counter() - request_started,
            streaming_execution=execution.to_dict(),
        )
        requests.append(request)
        _rank_zero_action(
            runtime,
            f"request {index} completion record",
            lambda: _write_json(segments_dir / f"segment_{index:02d}.complete.json", request),
        )
        if runtime.is_main_process:
            print(f"[lean] request {index + 1}/{segment_count} complete in {request['elapsed_seconds']:.1f}s", flush=True)
        del result

    streaming_overlay.release_retained_state()
    total_seconds = clip_frames / 24
    published_samples = round(total_seconds * 32_000)
    audio_path = run_dir / "audio.float.wav"

    def publish_audio() -> dict[str, Any]:
        full_audio = torch.cat(audios, dim=2).contiguous()
        waveform, sample_rate = runtime.decode_audio_latents(full_audio)
        if sample_rate != 32_000 or waveform.shape[0] != 1 or waveform.shape[1] != 2:
            raise RuntimeError("one-shot audio geometry differs from canonical stereo 32kHz")
        native_samples = int(waveform.shape[2])
        pcm = waveform[0].detach().float().transpose(0, 1).contiguous().cpu().numpy()
        with tempfile.TemporaryDirectory(prefix="h3-lean-audio-") as temporary:
            pcm_path = Path(temporary) / "audio.f32le"
            pcm_path.write_bytes(pcm.tobytes())
            _run([
                _binary("ffmpeg"), "-nostdin", "-v", "error", "-xerror",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "2", "-i", str(pcm_path),
                "-af", exact_audio_delivery_filter(
                    native_samples=native_samples,
                    published_samples=published_samples,
                    sample_rate=sample_rate,
                ),
                "-c:a", "pcm_f32le", "-n", str(audio_path),
            ])
        probe = _probe(audio_path)
        audio_stream = next(item for item in probe["streams"] if item["codec_type"] == "audio")
        if _audio_sample_count(audio_stream, label="lean canonical WAV") != published_samples:
            raise RuntimeError("published canonical WAV sample count differs")
        receipt = {
            "audio": str(audio_path),
            "audio_vae_decode_count": 1,
            "native_samples": native_samples,
            "published_samples": published_samples,
            "sample_rate": sample_rate,
            "channels": 2,
            "latent_shape": list(full_audio.shape),
            "audio_timing_filter": exact_audio_delivery_filter(
                native_samples=native_samples,
                published_samples=published_samples,
                sample_rate=sample_rate,
            ),
        }
        _write_json(run_dir / "audio.complete.json", receipt)
        print(f"[lean] audio ready: {audio_path}", flush=True)
        return receipt

    audio_receipt = _rank_zero_value(runtime, "canonical audio publication", publish_audio)
    audios.clear()
    full_video = torch.cat(videos, dim=2).contiguous()
    videos.clear()
    _rank_zero_action(runtime, "save final decode recovery latents", lambda:
        torch.save(full_video, run_dir / "generated_video_latents.pt"))
    return publish_complete_video(runtime=runtime,full_video=full_video,run_dir=run_dir,
        requests=requests,run_metadata=run_metadata,audio_receipt=audio_receipt,
        expected_output_width=expected_output_width,expected_output_height=expected_output_height,
        clip_frames=clip_frames,started=started)


def publish_complete_video(*,runtime,full_video,run_dir,requests,run_metadata,audio_receipt,
                           expected_output_width,expected_output_height,clip_frames,started):
    """Decode and mux the complete clip, including when resuming from saved latents."""
    from digitalhuman_h3.inference.long_video import (
        _audio_sample_count,_binary,_normalize_segment,_probe,_rank_zero_value,_run,
        _strict_decode,_validate_segment,_validated_segment_streams,_video_codec_native_frames,
    )
    run_dir = Path(run_dir)
    total_seconds = clip_frames/24
    published_samples = round(total_seconds*32000)
    published_frames = clip_frames
    audio_path = run_dir/'audio.float.wav'
    raw_video_path = run_dir / "video.one_shot.raw.mp4"
    normalized_video_path = run_dir / "video.one_shot.mp4"
    final_path = run_dir / "video.mp4"
    runtime.decode_video_latents(full_video, output_path=raw_video_path)
    expected_native_frames = _video_codec_native_frames(int(full_video.shape[2]))

    def publish_video() -> dict[str, Any]:
        raw_probe = _probe(raw_video_path)
        _validated_segment_streams(
            raw_probe,
            label="lean canonical raw video",
            frames=expected_native_frames,
            expected_width=expected_output_width,
            expected_height=expected_output_height,
            require_audio=False,
        )
        _strict_decode(raw_video_path, require_audio=False)
        _normalize_segment(
            raw_video_path,
            normalized_video_path,
            seconds=total_seconds,
            source_probe=raw_probe,
            include_audio=False,
            native_frames=expected_native_frames,
            published_frames=published_frames,
        )
        _validate_segment(
            normalized_video_path,
            seconds=total_seconds,
            expected_width=expected_output_width,
            expected_height=expected_output_height,
            require_audio=False,
        )
        mux_args = native_mux_output_args(final_path, published_frames)
        _run([
            _binary("ffmpeg"), "-nostdin", "-v", "error", "-xerror", "-copyts",
            "-i", str(normalized_video_path), "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            *mux_args,
        ])
        probe = _probe(final_path)
        _validated_segment_streams(
            probe, label="long-tail published video", frames=clip_frames,
            expected_width=expected_output_width,
            expected_height=expected_output_height,
            require_audio=True,
        )
        audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
        if _audio_sample_count(audio_stream, label='long-tail AAC') != published_samples:
            raise RuntimeError('published AAC sample count differs from native frame duration')
        _strict_decode(final_path)
        summary = {
            "video": str(final_path),
            "audio": str(audio_path),
            "probe": probe,
            "segments": requests,
            "run_metadata": dict(run_metadata),
            "model_resolution": runtime.model_resolution,
            "audio_publication": audio_receipt,
            "video_publication": {
                "video_vae_decode_count": 1,
                "diagnostic_per_request_video_vae_decode_count": 0,
                "native_frames": expected_native_frames,
                "published_frames": published_frames,
                "latent_shape": list(full_video.shape),
                "decode_receipt": runtime.last_video_decode_receipt,
            },
            "streaming_state_released_before_decode": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "result.json", summary)
        print(f"[lean] final video complete: {final_path}", flush=True)
        return summary

    return _rank_zero_value(runtime, "canonical video publication", publish_video)


__all__ = ["generate_long_video_lean", "publish_complete_video"]
