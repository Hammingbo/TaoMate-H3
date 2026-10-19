"""Combine prepared five-second prompts into nine continuous AR inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    entries = []
    for index in range(1, 10):
        folder = args.root/'inputs'/f'case_{index:02d}'
        paths = [folder/f'segment_{i:02d}.json' for i in range(1, 13)]
        if not all(p.is_file() for p in paths):
            if args.allow_partial:
                continue
            raise FileNotFoundError(f'source {index} does not yet have all twelve prompts')
        sources = [json.loads(p.read_text()) for p in paths]
        anchor = Path(sources[0]['initial_conditions'][0]['uri'])
        if any(s['initial_conditions'] != sources[0]['initial_conditions'] for s in sources):
            raise ValueError('a continuous clip must have one fixed initial image')
        case = dict(sources[0], case_id=f'case_{index:02d}_continuous',
                    description='Continuous 60s AR clip with twelve current-only prompts',
                    segments=[dict(s['segments'][0], segment_id=i) for i,s in enumerate(sources)])
        target = args.root/'continuous_inputs'/f'case_{index:02d}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(case, ensure_ascii=False, indent=2))
        entries.append(dict(case_id=case['case_id'], source_index=index, segment_index=0,
            case_file=str(target.absolute()), case_sha256=digest(target), anchor_sha256=digest(anchor),
            prompt_sha256=hashlib.sha256(json.dumps([s['prompt'] for s in case['segments']],
                                                  ensure_ascii=False).encode()).hexdigest(),
            original_inputs=[dict(path=str(p.absolute()), sha256=digest(p)) for p in paths]))
    target = args.root/'continuous_manifest.json'
    target.write_text(json.dumps(dict(continuous=True, requested_seconds=60,
        prompts_per_video=12, partial=len(entries)!=9, cases=entries), indent=2))
    print(f'CONTINUOUS_INPUTS count={len(entries)} manifest={target}')


if __name__ == '__main__':
    main()
