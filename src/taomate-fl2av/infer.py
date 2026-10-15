#!/usr/bin/env python3
"""First/last-frame audio-video inference with an 85-frame final block."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--time', type=int, choices=(5, 10, 15, 30), default=30)
    parser.add_argument('--case', type=Path, default=ROOT / 'examples/seated_001/case.json')
    parser.add_argument('--model-root', type=Path, help='MiniMax-H3 model directory containing FL2VA/')
    parser.add_argument('--checkpoint', type=Path, help='online generator LoRA directory (safetensors release or completed training checkpoint)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/seated_001')
    parser.add_argument('--teacher-root', type=Path, help='reuse a compatible completed teacher cache')
    parser.add_argument('--dry-run', action='store_true', help='print exact plan without loading weights')
    parser.add_argument('--stage', choices=('all', 'teacher', 'student'), default='all', help=argparse.SUPPRESS)
    args = parser.parse_args()
    from fl2va.plan import prepare_case
    case, schedule, active, transport = prepare_case(args.case.resolve(), args.time)
    if args.dry_run:
        print(json.dumps(dict(case=case, whole=schedule.plan.to_dict(),
            requests=[asdict(r) for r in schedule.requests],
            active=[p.to_dict() for p in active], transport=[p.to_dict() for p in transport]),
            ensure_ascii=False, indent=2))
        return
    if args.model_root is None or args.checkpoint is None:
        parser.error('--model-root and --checkpoint are required for inference')
    args.model_root = args.model_root.expanduser().resolve(strict=True)
    args.checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    args.teacher_root = (args.teacher_root or args.output / 'teacher').expanduser().resolve()
    args.seconds = args.time
    os.environ['H3_MODEL_ROOT'] = str(args.model_root)
    os.environ['H3_OUTPUT_ROOT'] = str(args.output)
    if args.stage != 'all':
        if int(os.environ.get('WORLD_SIZE', '1')) != 8:
            parser.error('this recipe requires eight ranks (TP2/SP4)')
        if args.stage == 'teacher':
            from fl2va.teacher import run_teacher
            args.output_dir = args.teacher_root
            run_teacher(args, case, schedule, transport)
        elif args.stage == 'student':
            from fl2va.student import run_student
            args.output_dir = args.output / 'result'
            run_student(args, case, schedule, active, transport)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR',
        'MASTER_PORT', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE', 'TORCHELASTIC_RUN_ID',
        'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',
        'TORCHELASTIC_USE_AGENT_STORE', 'TORCHELASTIC_ERROR_FILE'):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='1', H3_RELEASE_PRECOMPUTED_TEXT_ENCODER='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NCCL_SOCKET_IFNAME='lo',
        GLOO_SOCKET_IFNAME='lo', NCCL_DEBUG='WARN')
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes=1', '--nproc-per-node=8', str(ROOT / 'infer.py'), '--time', str(args.time),
        '--case', str(args.case.resolve()), '--model-root', str(args.model_root),
        '--checkpoint', str(args.checkpoint), '--output', str(args.output),
        '--teacher-root', str(args.teacher_root)]
    if not (args.teacher_root / 'complete.json').is_file():
        subprocess.run(command + ['--stage', 'teacher'], env=env, check=True)
    subprocess.run(command + ['--stage', 'student'], env=env, check=True)
    print(f"Video: {args.output / 'result/video.mp4'}")


if __name__ == '__main__':
    main()
