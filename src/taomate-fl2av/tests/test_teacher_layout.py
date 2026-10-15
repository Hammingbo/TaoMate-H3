"""Packed teacher layout tests, without loading model weights."""
from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from digitalhuman_h3.streaming.geometry import video_temporal_positions
from fl2va.plan import prepare_case


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/seated_001/case.json"


@unittest.skipIf(torch is None, "Packed-layout checks require the documented Torch dependency")
class TeacherLayoutTests(unittest.TestCase):
    def test_clip_global_endpoint_and_last_request_visibility(self):
        from fl2va.teacher import teacher_packed, visual_contract
        for seconds in (5, 10, 15, 30):
            case, schedule, _, _ = prepare_case(EXAMPLE, seconds)
            contract = visual_contract(case, schedule)
            self.assertEqual(contract["end_frame"], schedule.plan.native_frame_count)
            self.assertEqual(contract["end_video_latent"], schedule.video_latent_count)
            self.assertEqual(contract["tail_start_request_index"], len(schedule.requests) - 1)
            for request in schedule.requests:
                packed = teacher_packed(torch, text_len=64, latent_h=86, latent_w=48,
                                        request=request, schedule=schedule)
                positions = packed["img_position_ids"][
                    packed["img_pos"][~packed["update_mask"]], 0]
                final = request.index == len(schedule.requests) - 1
                self.assertEqual(positions.numel(), 1032 * (2 if final else 1))
                self.assertEqual(positions[0].item(), 64)
                if final:
                    endpoint = 64 + float(video_temporal_positions(
                        schedule.video_latent_count, 1)[0]) - 5 / 3
                    self.assertAlmostEqual(positions[1032].item(), endpoint, places=9)
                audio_rows = 2 * request.audio_latent_count + (80 if request.index else 0)
                self.assertEqual(packed["audio_pos"].numel(), audio_rows)
                if request.index:
                    self.assertFalse(packed["audio_update_mask"][:80].any().item())


if __name__ == "__main__":
    unittest.main()
