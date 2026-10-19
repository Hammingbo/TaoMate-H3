"""Validate native FL2VA geometry, streams, and full audio-video decoding."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--time", type=int, choices=(5, 10, 15, 30), required=True)
    parser.add_argument("--delivery-seconds", type=int, choices=(5,),
                        help="validate an explicitly cropped 120-frame delivery, preserving the native file separately")
    args = parser.parse_args()
    expected_frames = {5: 124, 10: 260, 15: 396, 30: 770}[args.time]
    if args.delivery_seconds:
        expected_frames = 24 * args.delivery_seconds
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(args.video)
    ], text=True))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (int(video["width"]), int(video["height"])) == (768, 1376), video
    assert int(video["nb_frames"]) == expected_frames, video
    assert Fraction(video["r_frame_rate"]) == 24, video
    assert video["codec_name"] == "h264", video
    assert audio["codec_name"] == "aac", audio
    assert (int(audio["sample_rate"]), int(audio["channels"])) == (32000, 2), audio
    assert abs(float(probe["format"]["duration"]) - expected_frames / 24) < 0.1, probe["format"]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(args.video),
                    "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], check=True)
    receipt = {"status": "passed", "video": str(args.video.resolve()),
               "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
               "requested_time_mode": args.time, "expected_native_frames": expected_frames,
               "expected_native_seconds": expected_frames / 24, "probe": probe,
               "full_decode": True}
    if args.delivery_seconds:
        receipt["delivery_seconds"] = args.delivery_seconds
        receipt["expected_delivery_frames"] = receipt.pop("expected_native_frames")
        receipt["expected_delivery_seconds"] = receipt.pop("expected_native_seconds")
    args.video.with_name("validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "probe"}, indent=2))


if __name__ == "__main__":
    main()
