"""Resumable eight-GPU independent I2VA benchmark with resident model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/taomate-fl2av/src"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def media_passed(folder, frames):
    report, video = folder/'validation.json', folder/'video.mp4'
    if not report.is_file() or not video.is_file():
        return False
    data = json.loads(report.read_text())
    return (data.get('status') == 'passed' and data.get('full_decode') is True
            and data.get('expected_delivery_frames', data.get('expected_native_frames')) == frames
            and data.get('video_sha256') == digest(video))


def load_cases(manifest_path, limit):
    from fl2va.plan import prepare_case
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["cases"]
    if len(entries) != 108 or len({e['case_id'] for e in entries}) != 108:
        raise ValueError("expected exactly 108 uniquely identified input cases")
    counts = {i: sum(e['source_index'] == i for e in entries) for i in range(1, 10)}
    if set(counts.values()) != {12}:
        raise ValueError("every source must have exactly twelve cases")
    prepared = []
    for entry in entries[:limit]:
        expected_id = f"case_{entry['source_index']:02d}_segment_{entry['segment_index']+1:02d}"
        if entry['case_id'] != expected_id or not 0 <= entry['segment_index'] < 12:
            raise ValueError('case identity or segment index differs')
        path = Path(entry["case_file"])
        if digest(path) != entry["case_sha256"]:
            raise ValueError(f"case changed after preparation: {entry['case_id']}")
        case, schedule, active, transport = prepare_case(path, 5)
        if case['case_id'] != entry['case_id']:
            raise ValueError('case file identity differs from the manifest')
        if case['task'] != 'i2va' or len(schedule.requests) != 1:
            raise ValueError("benchmark admits only independent first-image-only 5s cases")
        if digest(case['initial_conditions'][0]['uri']) != entry['anchor_sha256']:
            raise ValueError("edited anchor changed after preparation")
        prepared.append((entry, case, schedule, active, transport))
    return prepared


def delivery(native, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'video.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-y', '-i', str(native),
                    '-t', '5', '-frames:v', '120', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-ar', '32000',
                    '-ac', '2', '-movflags', '+faststart', str(target)], check=True)
    subprocess.run([sys.executable, str(REPO/'deploy/h20/validate_output.py'),
                    str(target), '--time', '5', '--delivery-seconds', '5'], check=True)


def write_status(args, prepared, stage, current):
    rows = []
    for entry, *_ in prepared:
        root = args.output / entry['case_id']
        passed = media_passed(root/'delivery', 120)
        rows.append({'case_id': entry['case_id'], 'source_index': entry['source_index'],
                     'segment_index': entry['segment_index'],
                     'teacher_complete': (root/'teacher/complete.json').exists(),
                     'passed': passed, 'video': str(root/'delivery/video.mp4')})
    data = {'stage': stage, 'current': current, 'total': len(rows),
            'passed': sum(r['passed'] for r in rows), 'cases': rows}
    temporary = args.output / 'status.tmp.json'
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(args.output / 'status.json')


def worker(args, prepared):
    import torch
    from digitalhuman_h3.runtime import H3Runtime
    from digitalhuman_h3.inference.acceleration import InferenceAccelerationConfig
    from fl2va.teacher import run_teacher
    from fl2va.student import run_student

    prompts = tuple(schedule.requests[0].prompt for _, _, schedule, _, _ in prepared)
    text_cache = args.output / f'text_features_{len(prompts)}.pt'
    if args.worker_stage == 'student':
        if not text_cache.is_file():
            raise FileNotFoundError('teacher stage must prepare all text features first')
        os.environ['H3_PRECOMPUTED_TEXT_PATH'] = str(text_cache)
    else:
        os.environ.pop('H3_PRECOMPUTED_TEXT_PATH', None)
    runtime_args = dict(model_root=args.model_root, variant='fl2va', tp_size=2, ulysses_degree=4)
    if args.worker_stage == 'student':
        runtime_args.update(adapter_checkpoint=args.checkpoint, allow_intermediate_checkpoint=True,
                            denoise_schedule='distilled4', inference_acceleration=
                            InferenceAccelerationConfig(streaming_attention_backend='sdpa'))
    runtime = H3Runtime(**runtime_args)
    features = runtime.precompute_text_prompts(prompts)
    if args.worker_stage == 'teacher' and runtime.is_main_process:
        torch.save({'prompts': prompts, 'encoded': features,
                    'producer': {'source': 'native_bf16_text_encoder',
                                 'model_root': str(args.model_root)}}, text_cache)
    runtime.barrier()
    for index, (entry, case, schedule, active, transport) in enumerate(prepared):
        root = args.output / entry['case_id']
        if runtime.is_main_process:
            root.mkdir(parents=True, exist_ok=True)
            source_receipt = root/'input.json'
            if source_receipt.exists() and json.loads(source_receipt.read_text()) != entry:
                raise ValueError('input identity differs from the existing output')
            source_receipt.write_text(json.dumps(entry, indent=2))
        runtime.barrier()
        item_args = SimpleNamespace(model_root=args.model_root, checkpoint=args.checkpoint,
                                    seconds=5, teacher_root=root/'teacher')
        runtime._last_clean_video_latents = None
        runtime._last_clean_audio_latents = None
        if args.worker_stage == 'teacher':
            if not (root/'teacher/complete.json').is_file():
                if runtime.is_main_process and (root/'teacher').exists():
                    (root/'teacher').rename(root/f'teacher.incomplete.{time.time_ns()}')
                runtime.barrier()
                item_args.output_dir = root/'teacher'
                run_teacher(item_args, case, schedule, transport, runtime=runtime,
                            features=(features[index],), close_runtime=False)
        else:
            if not (root/'native/probe.complete.json').is_file():
                if runtime.is_main_process and (root/'native').exists():
                    (root/'native').rename(root/f'native.incomplete.{time.time_ns()}')
                runtime.barrier()
                item_args.output_dir = root/'native'
                run_student(item_args, case, schedule, active, transport, runtime=runtime,
                            features=(features[index],), close_runtime=False, keep_transformer=True)
            if runtime.is_main_process:
                if not media_passed(root/'native', 124):
                    subprocess.run([sys.executable, str(REPO/'deploy/h20/validate_output.py'),
                                    str(root/'native/video.mp4'), '--time', '5'], check=True)
                if not media_passed(root/'delivery', 120):
                    delivery(root/'native/video.mp4', root/'delivery')
        runtime.barrier()
        if runtime.is_main_process:
            write_status(args, prepared, args.worker_stage, entry['case_id'])
            print(f"BATCH_PROGRESS stage={args.worker_stage} case={index+1}/{len(prepared)} "
                  f"id={entry['case_id']}", flush=True)
    runtime.barrier()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=108)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker-stage', choices=('teacher', 'student'))
    args = parser.parse_args()
    if not 1 <= args.limit <= 108:
        parser.error('--limit must be between 1 and 108')
    args.output = args.output.resolve()
    prepared = load_cases(args.manifest, args.limit)
    if args.dry_run:
        print(json.dumps({'count': len(prepared), 'task': 'i2va', 'tail_constraint': False,
                          'native_frames': 124, 'delivered_frames': 120,
                          'delivered_seconds': 5, 'kv_reset_between_cases': True}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(H3_MODEL_ROOT=str(args.model_root), H3_OUTPUT_ROOT=str(args.output),
                      H3_RELEASE_PRECOMPUTED_TEXT_ENCODER='1')
    if args.worker_stage:
        worker(args, prepared)
        return
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', NCCL_SOCKET_IFNAME='lo', GLOO_SOCKET_IFNAME='lo',
               NCCL_DEBUG='WARN', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nproc-per-node=8', str(Path(__file__).resolve()),
               '--manifest', str(args.manifest.resolve()), '--output', str(args.output),
               '--model-root', str(args.model_root), '--checkpoint', str(args.checkpoint),
               '--limit', str(args.limit)]
    for stage in ('teacher', 'student'):
        marker = 'teacher/complete.json' if stage == 'teacher' else 'delivery/validation.json'
        complete = (all(media_passed(args.output/e['case_id']/'delivery', 120) for e, *_ in prepared)
                    if stage == 'student' else
                    all((args.output/e['case_id']/marker).exists() for e, *_ in prepared))
        if complete:
            if stage == 'student' or (args.output/f'text_features_{len(prepared)}.pt').exists():
                print(f'STAGE_ALREADY_COMPLETE {stage}', flush=True)
                continue
        subprocess.run(command + ['--worker-stage', stage], env=env, check=True)
    write_status(args, prepared, 'complete', None)
    if not all(media_passed(args.output/e['case_id']/'delivery', 120) for e, *_ in prepared):
        raise RuntimeError('not every requested delivery passed validation')
    print(f'BATCH_COMPLETE count={len(prepared)} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
