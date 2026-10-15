"""The supported native schedules and original benchmark prompt selection."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from digitalhuman_h3.streaming.geometry import StreamPlan, plan_native_stream
from digitalhuman_h3.inference.base10_hiar import Base10HiARRequest, Base10HiARSchedule
from digitalhuman_h3.inference.prompts import official_three_field_prompt


def _merge_tail(plan: StreamPlan):
    first, last = plan.phases[-3], plan.phases[-1]
    return StreamPlan(plan.native_frame_count, (*plan.phases[:-3], replace(first,
        group_count=5, frame_stop=last.frame_stop,
        video_latent_stop=last.video_latent_stop, audio_latent_stop=last.audio_latent_stop,
        audio_sample_stop=last.audio_sample_stop)))


def geometry(time: int):
    if time not in (5, 10, 15, 30):
        raise ValueError('--time must be 5, 10, 15 or 30')
    count = {5: 2, 10: 6, 15: 10, 30: 21}[time]
    frames = 39 + 34 * (count - 2) + 85
    whole = _merge_tail(plan_native_stream(frames))
    active, transport = [], []
    for begin in range(0, count, 7):
        phases = whole.phases[begin:begin+7]
        start = phases[0]
        local = [replace(p, index=j,
            frame_start=p.frame_start-start.frame_start, frame_stop=p.frame_stop-start.frame_start,
            video_latent_start=p.video_latent_start-start.video_latent_start,
            video_latent_stop=p.video_latent_stop-start.video_latent_start,
            audio_latent_start=p.audio_latent_start-start.audio_latent_start,
            audio_latent_stop=p.audio_latent_stop-start.audio_latent_start,
            audio_sample_start=p.audio_sample_start-start.audio_sample_start,
            audio_sample_stop=p.audio_sample_stop-start.audio_sample_start)
            for j, p in enumerate(phases)]
        active.append(StreamPlan(local[-1].frame_stop, tuple(local)))
        official = plan_native_stream(5 + sum(p.group_count for p in phases)*17)
        if phases[-1].group_count == 5:
            official = _merge_tail(official)
        transport.append(official)
    return whole, active, transport


def prepare_case(case_path: Path, time: int):
    source = json.loads(case_path.read_text(encoding='utf-8'))
    whole, active, transport = geometry(time)
    if len(source['segments']) != 6:
        raise ValueError('case.json must contain the six original ordered 5s segments')
    source_segments = source['segments'][:time//5]
    requests, segments = [], []
    for index, plan in enumerate(active):
        selected = source_segments[2*index:2*index+2]
        segment = selected[0]
        speeches, actions = [], []
        for item in selected:
            normalized = official_three_field_prompt(item['prompt'])
            if f"<d>[Chinese] {item['speech_text']}</d>" not in normalized:
                raise ValueError('speech metadata differs from source prompt')
            speeches.append(item['speech_text'])
            actions.append(re.search(r"Speaker_1's Actions: ([^\n]+)", item['prompt']).group(1))
        dialogue = ''.join(speeches)
        if len(selected) == 2:
            merged = re.sub(r'Speaker_1 says: ".*?"', lambda _: f'Speaker_1 says: "{dialogue}"', segment['prompt'])
            merged = re.sub(r"Speaker_1's Actions: [^\n]+", lambda _: "Speaker_1's Actions: First: " + actions[0] + ' Then: ' + actions[1], merged)
            prompt = official_three_field_prompt(merged)
        else:
            prompt = official_three_field_prompt(segment['prompt'])
        phase_start = index * 7
        first, last = whole.phases[phase_start], whole.phases[phase_start+len(plan.phases)-1]
        requests.append(Base10HiARRequest(index=index, prompt=prompt, seed=segment['seed'],
            phase_start=phase_start, phase_stop=phase_start+len(plan.phases),
            frame_start=first.frame_start, frame_stop=last.frame_stop,
            video_latent_start=first.video_latent_start, video_latent_stop=last.video_latent_stop,
            audio_latent_start=first.audio_latent_start, audio_latent_stop=last.audio_latent_stop))
        segments.append(dict(prompt=prompt, seed=segment['seed'], dialogue=dialogue,
            original_segment_ids=[s['segment_id'] for s in selected],
            active_frame_start=first.frame_start, active_frame_stop=last.frame_stop))
    conditions = []
    for condition in source['initial_conditions']:
        image = (case_path.parent / condition['uri']).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        conditions.append(dict(type='image', role='keyframe', uri=str(image), frame_index=condition['frame_index']))
    if [c['frame_index'] for c in conditions] != [0, -1]:
        raise ValueError('case.json must specify first/end image conditions [0,-1]')
    case = dict(case_id=source['case_id'], source_case_id=source['case_id'], task='ii2va',
        title=source['title'], category=source['category'], description=source['description'],
        source_manifest=str(case_path.resolve()), source_target_seconds=time,
        text_policy=('unaltered_first_5s_segment' if time == 5 else
            'unaltered_adjacent_pair_then_single_segment' if time == 15 else
            'unaltered_adjacent_pair_current_only'),
        segments=segments, initial_conditions=conditions)
    return case, Base10HiARSchedule(whole, tuple(requests)), active, transport
