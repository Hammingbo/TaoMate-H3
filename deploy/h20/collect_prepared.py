"""Collect only fully prepared twelve-clip sources for an early GPU pilot."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    entries = []
    for folder in sorted((args.root/'inputs').glob('case_*')):
        paths = sorted(folder.glob('segment_*.json'))
        if len(paths) != 12:
            continue
        source_index = int(folder.name.split('_')[1])
        for index, path in enumerate(paths):
            case = json.loads(path.read_text())
            entries.append({'case_id': case['case_id'], 'source_index': source_index,
                'segment_index': index, 'reference_start_seconds': index*5,
                'case_file': str(path.absolute()), 'case_sha256': digest(path),
                'anchor_sha256': digest(case['initial_conditions'][0]['uri']),
                'prompt_sha256': hashlib.sha256(case['segments'][0]['prompt'].encode()).hexdigest()})
    target = args.root/'pilot_manifest.json'
    target.write_text(json.dumps({'partial': True, 'expected_total': 108, 'cases': entries}, indent=2))
    print(f'PILOT_INPUTS count={len(entries)} manifest={target}')


if __name__ == '__main__':
    main()
