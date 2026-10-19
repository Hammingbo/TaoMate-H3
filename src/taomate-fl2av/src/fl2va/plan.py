"""The supported native schedules and original benchmark prompt selection."""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

from digitalhuman_h3.streaming.geometry import StreamPlan, plan_native_stream, native_frames_for_duration
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
    if time == 60:
        return prepare_continuous_case(case_path, source)
    whole, active, transport = geometry(time)
    if len(source['segments']) < time // 5:
        raise ValueError('case.json has fewer ordered 5s segments than requested')
    source_segments = source['segments'][:time//5]
    requests, segments = [], []
    for index, plan in enumerate(active):
        selected = source_segments[2*index:2*index+2]
        segment = selected[0]
        speeches, actions = [], []
        for item in selected:
            raw = item['prompt']
            normalized = official_three_field_prompt(raw) if raw.startswith('Summary: ') else raw
            dialogues = re.findall(r'<d>\s*\[Chinese\]\s*(.*?)</d>', normalized, re.DOTALL)
            if [d.strip() for d in dialogues] != [item['speech_text'].strip()]:
                raise ValueError('speech metadata differs from source prompt')
            if any(normalized.count(field + ':') != 1 for field in
                   ('integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music')):
                raise ValueError('prompt must contain exactly the three H3 fields')
            speeches.append(item['speech_text'])
            action = re.search(r"Speaker_1's Actions: ([^\n]+)", raw)
            actions.append(action.group(1) if action else '')
        dialogue = ''.join(speeches)
        if len(selected) == 2:
            if not all(item['prompt'].startswith('Summary: ') for item in selected):
                raise ValueError('formatted custom prompts currently support independent --time 5 only')
            merged = re.sub(r'Speaker_1 says: ".*?"', lambda _: f'Speaker_1 says: "{dialogue}"', segment['prompt'])
            merged = re.sub(r"Speaker_1's Actions: [^\n]+", lambda _: "Speaker_1's Actions: First: " + actions[0] + ' Then: ' + actions[1], merged)
            prompt = official_three_field_prompt(merged)
        else:
            raw = segment['prompt']
            prompt = official_three_field_prompt(raw) if raw.startswith('Summary: ') else raw
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
    indices = [c['frame_index'] for c in conditions]
    if indices not in ([0], [0, -1]):
        raise ValueError('case.json must specify first image [0] or first/end images [0,-1]')
    if indices == [0] and time != 5:
        raise ValueError('first-image-only cases currently support independent --time 5 only')
    if indices == [0] and any(re.search(r'Picture\s+2\b', r.prompt) for r in requests):
        raise ValueError('first-image-only prompt must not reference Picture 2')
    case = dict(case_id=source['case_id'], source_case_id=source['case_id'],
        task='ii2va' if len(conditions) == 2 else 'i2va',
        title=source['title'], category=source['category'], description=source['description'],
        source_manifest=str(case_path.resolve()), source_target_seconds=time,
        text_policy=('unaltered_first_5s_segment' if time == 5 else
            'unaltered_adjacent_pair_then_single_segment' if time == 15 else
            'unaltered_adjacent_pair_current_only'),
        segments=segments, initial_conditions=conditions)
    return case, Base10HiARSchedule(whole, tuple(requests)), active, transport


def prepare_continuous_case(case_path: Path, source):
    """Twelve current-only prompts on one native timeline; no repeated codec prefix.

    Prompt boundaries follow the nearest 17-frame group. Only the final request
    has the native five-group tail. Audio uses global rounded 40 Hz boundaries.
    """
    if len(source['segments']) != 12:
        raise ValueError('continuous 60s cases require exactly twelve ordered prompts')
    native_frames = native_frames_for_duration(60)
    units = plan_native_stream(native_frames, groups_per_phase=1).phases
    cuts = [0] + [round((120*i-5)/17) for i in range(1,12)] + [len(units)]
    phases, active, transport, requests, segments = [], [], [], [], []
    for index, segment in enumerate(source['segments']):
        if segment['segment_id'] != index:
            raise ValueError('continuous prompt IDs must be ordered 0 through 11')
        prompt = segment['prompt']
        dialogues = re.findall(r'<d>\s*\[Chinese\]\s*(.*?)</d>', prompt, re.DOTALL)
        if [d.strip() for d in dialogues] != [segment['speech_text'].strip()]:
            raise ValueError('speech metadata differs from source prompt')
        if any(prompt.count(field + ':') != 1 for field in
               ('integrated_multimodal_description', 'overall_soundscape', 'non_diegetic_music')):
            raise ValueError('prompt must contain exactly the three H3 fields')
        if re.search(r'Picture\s+2\b', prompt):
            raise ValueError('first-image-only prompt must not reference Picture 2')
        chosen = units[cuts[index]:cuts[index+1]]
        groups = len(chosen)
        counts = []
        remaining = groups - (5 if index == 11 else 0)
        while remaining:
            count = min(2, remaining); counts.append(count); remaining -= count
        if index == 11:
            counts.append(5)
        begin = len(phases)
        cursor = 0
        for count in counts:
            first, last = chosen[cursor], chosen[cursor+count-1]
            phases.append(replace(first, index=len(phases), group_count=count,
                frame_stop=last.frame_stop, video_latent_stop=last.video_latent_stop,
                audio_latent_stop=last.audio_latent_stop, audio_sample_stop=last.audio_sample_stop))
            cursor += count
        global_phases = phases[begin:]
        start = global_phases[0]
        local = [replace(p, index=j,
            frame_start=p.frame_start-start.frame_start, frame_stop=p.frame_stop-start.frame_start,
            video_latent_start=p.video_latent_start-start.video_latent_start,
            video_latent_stop=p.video_latent_stop-start.video_latent_start,
            audio_latent_start=p.audio_latent_start-start.audio_latent_start,
            audio_latent_stop=p.audio_latent_stop-start.audio_latent_start,
            audio_sample_start=p.audio_sample_start-start.audio_sample_start,
            audio_sample_stop=p.audio_sample_stop-start.audio_sample_start)
            for j,p in enumerate(global_phases)]
        active.append(StreamPlan(local[-1].frame_stop, tuple(local)))
        base = plan_native_stream(5+groups*17, groups_per_phase=1)
        transport_phases, cursor = [], 0
        for count in counts:
            first, last = base.phases[cursor], base.phases[cursor+count-1]
            transport_phases.append(replace(first, index=len(transport_phases), group_count=count,
                frame_stop=last.frame_stop, video_latent_stop=last.video_latent_stop,
                audio_latent_stop=last.audio_latent_stop, audio_sample_stop=last.audio_sample_stop))
            cursor += count
        transport.append(StreamPlan(base.native_frame_count, tuple(transport_phases)))
        last = global_phases[-1]
        requests.append(Base10HiARRequest(index=index, prompt=prompt, seed=segment['seed'],
            phase_start=begin, phase_stop=len(phases), frame_start=start.frame_start,
            frame_stop=last.frame_stop, video_latent_start=start.video_latent_start,
            video_latent_stop=last.video_latent_stop, audio_latent_start=start.audio_latent_start,
            audio_latent_stop=last.audio_latent_stop))
        segments.append(dict(prompt=prompt, seed=segment['seed'], dialogue=segment['speech_text'],
            original_segment_ids=[index], active_frame_start=start.frame_start,
            active_frame_stop=last.frame_stop, reference_start_seconds=index*5))
    if len(source['initial_conditions']) != 1 or source['initial_conditions'][0]['frame_index'] != 0:
        raise ValueError('continuous benchmark requires only the initial frame')
    condition = source['initial_conditions'][0]
    image = (case_path.parent/condition['uri']).resolve(strict=True)
    case = dict(case_id=source['case_id'], source_case_id=source['case_id'], task='i2va',
        title=source['title'], category=source['category'], description=source['description'],
        source_manifest=str(case_path.resolve()), source_target_seconds=60,
        text_policy='twelve_unaltered_5s_prompts_continuous_clean_kv', segments=segments,
        initial_conditions=[dict(type='image', role='keyframe', uri=str(image), frame_index=0)])
    return case, Base10HiARSchedule(StreamPlan(native_frames, tuple(phases)), tuple(requests)), active, transport
