"""Prepare independent 5-second I2VA cases using an existing live-generation checkout.

Run on the CPU development host, where the reference videos and gateway client
are available. No gateway credentials are copied into this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


RULES = """Analyze the attached reference livestream and return JSON only.
Apply MiniMax-H3 base-mode writing rules to EVERY clip independently.
There is one presenter and one voice. Every clip is a continuous static shot,
exactly 5.00 seconds, conditioned ONLY on the actual edited start image Picture 1.
Never refer to Picture 2, an end image, looping or a prescribed final pose.
The edited Picture 1 is authoritative for identity, clothing, product, scenery,
lighting and composition. It may differ from the attached reference video.
Use neutral terms 'the presenter' and 'they'; do not copy the old person's
appearance, gender, old brand, package color or backdrop as visual requirements.
Refer to the existing product visible in Picture 1; never invent an absent object.
Use the video only for sales progression, observable simple actions and speech.
Keep one intact closed product, with small natural movements. No unpacking,
extracting contents, application to the body, cuts, zooms or additional people.
Keep printed details already in the image; add no captions, logos or overlays.

Each shot contains shot_id, duration=5, anchor_action, speech_script and h3_prompt.
h3_prompt has EXACTLY three nonempty English fields:
integrated_multimodal_description, overall_soundscape, non_diegetic_music.
The first field begins [Shot 1], mentions <Picture 1>, states 5.00 seconds,
describes a clear natural voice outside dialogue tags, identifies the visible
presenter as (S1), and contains ONE <d>[Chinese]EXACT speech_script</d> block.
Target 14-20 Chinese characters of natural speech per five-second clip. Keep the
spoken line short; never copy an entire long sentence from the reference video.
No stage directions inside dialogue. Never omit (S1) in later clips. Describe
one simple motion developing naturally over five seconds without an end-image
constraint. All descriptive prose is English; Chinese is allowed only inside
dialogue and verbatim actually visible text in double quotes.
overall_soundscape describes quiet room tone, breathing and gentle fabric or
object sounds, without repeating dialogue. non_diegetic_music is N/A.
Do not put field labels, a full formatted prompt, Markdown or timestamps after
[Shot 1] inside any field. Every shot is independently I2VA, never FL2VA.
"""


def resolve_shared(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    # The two containers expose the same user volume under different spellings.
    for old, new in (("/data/haomingqin/", "/data/haomngqin/"),
                     ("/data/haomngqin/", "/data/haomingqin/")):
        if value.startswith(old):
            candidate = Path(new + value[len(old):])
            if candidate.exists():
                return candidate
    raise FileNotFoundError(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_script(script, rules, count=12):
    shots = script.get("shots")
    if not isinstance(shots, list) or len(shots) != count:
        raise ValueError(f"expected {count} shots")
    for index, shot in enumerate(shots, 1):
        if shot.get("shot_id") != index or shot.get("duration") != 5:
            raise ValueError("shot order or five-second duration differs")
        count = len(re.findall(r"[\u3400-\u9fff]", shot.get("speech_script", "")))
        if count > 28:
            raise ValueError(f"shot {index} has {count} Chinese characters; shorten its speech to 14-20 characters, and update its dialogue tag identically")
        shot["h3_prompt"] = rules.validate_h3_fields(shot, use_end_frame=False, duration=5)
    return shots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-set", type=Path, required=True)
    parser.add_argument("--live-generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.live_generation.resolve()))
    rules = importlib.import_module("live_generation.pipelines.h3_prompt_rules")
    client = importlib.import_module("live_generation.pipelines.pipeline_video_chain_h3")
    source = json.loads(args.test_set.read_text())
    if len(source["cases"]) != 9:
        raise ValueError("this benchmark expects nine source tasks")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "taomate_h3.independent_i2va_benchmark.v1", "duration": 5,
                "segments_per_source": 12, "tail_constraint": False,
                "test_set_sha256": sha256(args.test_set),
                "preparer_sha256": sha256(Path(__file__)),
                "prompt_policy": rules.H3_PROMPT_POLICY_VERSION,
                "prompt_rules_sha256": sha256(Path(rules.__file__)),
                "live_generation_commit": subprocess.check_output(
                    ["git", "-C", str(args.live_generation), "rev-parse", "HEAD"], text=True).strip(),
                "cases": []}
    for source_case in source["cases"]:
        source_id = int(source_case["index"])
        folder = args.output / "inputs" / f"case_{source_id:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        original = resolve_shared(source_case["output_dir"])
        result = json.loads((original / "result.json").read_text())
        image = resolve_shared(result["initial_first_frame"])
        video = resolve_shared(source_case["source_video"])
        anchor = folder / ("anchor" + image.suffix.lower())
        if not anchor.exists():
            shutil.copyfile(image, anchor)
        if sha256(anchor) != sha256(image):
            raise ValueError("cached anchor differs from the edited first image")
        script_path = folder / "script_5s.json"
        if script_path.exists():
            script = json.loads(script_path.read_text())
            shots = validate_script(script, rules)
        else:
            error = ""
            rejected = ""
            for attempt in range(3):
                request = (
                    "Prepare exactly 12 sequential five-second plans covering the first "
                    "60 seconds of the attached video, in order. Return an object with shots. "
                    f"The edited image replaces these roles: {source_case.get('edit_mode', 'none')}. "
                    "Each clip starts independently from the SAME edited initial image. "
                    "Preserve the overall presentation progression, but make speech fit "
                    "each five-second clip. Do not reuse a fifteen-second speech paragraph. "
                    "No end-image constraint. "
                    + ("Correct this previous validation error in every clip: " + error if error else "")
                )
                if rejected:
                    request += ("\nRepair this rejected draft. Keep all valid shots and their "
                                "wording; fix the failed speech/format consistently in speech_script "
                                "and h3_prompt. Return all twelve corrected shots.\n" + rejected)
                text, elapsed = client.call_gemini_wanqing(
                    str(video), request, system_prompt=RULES)
                if not text:
                    error = "No complete JSON was returned"
                    continue
                try:
                    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                    script = json.loads(cleaned)
                    (folder/f'draft_{attempt+1}.json').write_text(json.dumps(script, ensure_ascii=False, indent=2))
                    rejected = json.dumps(script, ensure_ascii=False)
                    shots = validate_script(script, rules)
                except (ValueError, TypeError, KeyError) as exc:
                    error = str(exc)[:500]
                    print(f"SCRIPT_RETRY source={source_id} attempt={attempt+1} reason={error}", flush=True)
                    continue
                script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2))
                break
            else:
                raise RuntimeError(f"source {source_id}: script preparation failed: {error}")
        for segment_index, shot in enumerate(shots):
            case_id = f"case_{source_id:02d}_segment_{segment_index+1:02d}"
            case_path = folder / f"segment_{segment_index+1:02d}.json"
            prompt = rules.format_h3_prompt(shot, use_end_frame=False, duration=5)
            payload = {"case_id": case_id, "category": source_case.get("category", ""),
                       "title": source_case.get("run_name", case_id),
                       "description": "Independent five-second first-image-conditioned test",
                       "initial_conditions": [{"type": "image", "role": "keyframe",
                                               "uri": str(anchor.resolve()), "frame_index": 0}],
                       "segments": [{"segment_id": 0, "seed": 9301 + source_id*100 + segment_index,
                                     "speech_text": shot["speech_script"], "prompt": prompt}]}
            case_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            manifest["cases"].append({"case_id": case_id, "source_index": source_id,
                "segment_index": segment_index, "reference_start_seconds": segment_index*5,
                "case_file": str(case_path.resolve()), "case_sha256": sha256(case_path),
                "anchor_sha256": sha256(anchor), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        print(f"SOURCE_PREPARED {source_id}/9 clips=12", flush=True)
    assert len(manifest["cases"]) == 108
    target = args.output / "manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"ALL_INPUTS_READY {target} count=108", flush=True)


if __name__ == "__main__":
    main()
