from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Sequence


SPEECH_PREFIX = 'Speaker_1 says: "'
SPEECH_SUFFIX = '"\nSpeaker_1\'s Emotion:'

PROMPT_FORMAT_SOURCE = "source"
PROMPT_FORMAT_OFFICIAL_THREE_FIELD = "official_three_field"
PROMPT_FORMATS = (
    PROMPT_FORMAT_SOURCE,
    PROMPT_FORMAT_OFFICIAL_THREE_FIELD,
)

_SPEECH_BLOCK = re.compile(
    r'^Speaker_1 says: "(?P<dialogue>.+)"\n'
    r"Speaker_1's Emotion: (?P<emotion>.+)\n"
    r"Speaker_1's Voice Description: (?P<voice>.+)$"
)


def _flatten(value: str) -> str:
    return " ".join(value.split())


def official_three_field_prompt(prompt: str) -> str:
    """Render the live benchmark prompt in MiniMax H3's official T2VA form."""

    if not prompt.startswith("Summary: "):
        raise ValueError("prompt lacks Summary header")
    summary_and_narration, speech_marker, speech_and_sound = prompt.partition(
        "\nSpeech Attribution:\n"
    )
    if not speech_marker:
        raise ValueError("prompt lacks Speech Attribution section")
    summary, narration_marker, narration = summary_and_narration.partition(
        "\nNarration 1:\n"
    )
    if not narration_marker:
        raise ValueError("prompt lacks Narration 1 section")
    speech, sound_marker, soundscape = speech_and_sound.partition(
        "\nSound-Visual Alignment: "
    )
    if not sound_marker:
        raise ValueError("prompt lacks Sound-Visual Alignment section")
    match = _SPEECH_BLOCK.fullmatch(speech)
    if match is None:
        raise ValueError("speech block differs from the live benchmark contract")

    summary_text = summary.removeprefix("Summary: ")
    voice = _flatten(match.group("voice"))
    emotion = _flatten(match.group("emotion")).removesuffix(".")
    dialogue = match.group("dialogue")
    integrated = " ".join(
        (
            _flatten(summary_text),
            _flatten(narration),
            f"Speaker 1 with {voice} (S1) says with {emotion} delivery: "
            f"<d>[Chinese] {dialogue}</d>",
        )
    )
    return (
        f"integrated_multimodal_description: [Shot 1] {integrated}"
        f"\n\noverall_soundscape: {_flatten(soundscape)}"
        "\n\nnon_diegetic_music: N/A"
    )


def format_benchmark_prompts(
    cases: Sequence[dict[str, Any]],
    *,
    prompt_format: str,
) -> list[dict[str, Any]]:
    """Copy benchmark cases and apply the selected auditable prompt format."""

    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"unsupported prompt format: {prompt_format}")
    normalized = deepcopy(list(cases))
    if prompt_format == PROMPT_FORMAT_SOURCE:
        return normalized
    for case in normalized:
        segments = case.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"{case.get('case_id')} segments must be a list")
        for index, segment in enumerate(segments):
            prompt = segment.get("prompt") if isinstance(segment, dict) else None
            if not isinstance(prompt, str):
                raise ValueError(
                    f"{case.get('case_id')} segment {index} prompt is invalid"
                )
            segment["prompt"] = official_three_field_prompt(prompt)
    return normalized


def without_speech_overlap(
    cases: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return benchmark cases whose continuation prompts speak only new text."""

    normalized = deepcopy(list(cases))
    for case in normalized:
        previous_sentence: str | None = None
        segments = case.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"{case.get('case_id')} segments must be a list")
        for index, segment in enumerate(segments):
            prompt = segment.get("prompt") if isinstance(segment, dict) else None
            if not isinstance(prompt, str):
                raise ValueError(
                    f"{case.get('case_id')} segment {index} prompt is invalid"
                )
            speech_start = prompt.find(SPEECH_PREFIX)
            if speech_start < 0:
                raise ValueError(
                    f"{case.get('case_id')} segment {index} has no Speaker_1 speech"
                )
            speech_start += len(SPEECH_PREFIX)
            speech_stop = prompt.find(SPEECH_SUFFIX, speech_start)
            if speech_stop < 0:
                raise ValueError(
                    f"{case.get('case_id')} segment {index} speech is not terminated"
                )
            speech = prompt[speech_start:speech_stop]
            current_sentence = speech
            if previous_sentence and speech.startswith(previous_sentence):
                current_sentence = speech[len(previous_sentence) :].lstrip()
            if not current_sentence:
                raise ValueError(
                    f"{case.get('case_id')} segment {index} has no current speech"
                )
            prompt = prompt[:speech_start] + current_sentence + prompt[speech_stop:]
            segment["prompt"] = prompt
            previous_sentence = current_sentence
    return normalized


__all__ = [
    "PROMPT_FORMAT_OFFICIAL_THREE_FIELD",
    "PROMPT_FORMAT_SOURCE",
    "PROMPT_FORMATS",
    "format_benchmark_prompts",
    "official_three_field_prompt",
    "without_speech_overlap",
]
