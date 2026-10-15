"""CPU schedule/input contracts; not numerical or perceptual quality claims."""
from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from fl2va.plan import geometry, prepare_case


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/seated_001/case.json"


class PlanTests(unittest.TestCase):
    def test_geometry_and_case_api_only_accept_time(self):
        self.assertEqual(tuple(inspect.signature(geometry).parameters), ("time",))
        self.assertEqual(tuple(inspect.signature(prepare_case).parameters), ("case_path", "time"))

    def test_5s_tail85_native_geometry(self):
        whole, active, transport = geometry(5)
        self.assertEqual([p.frame_count for p in whole.phases], [39, 85])
        self.assertEqual(whole.native_frame_count, 124)
        self.assertEqual(whole.phases[-1].group_count, 5)
        self.assertEqual(whole.phases[-1].video_latent_stop, 37)
        self.assertEqual(whole.phases[-1].audio_latent_stop, 207)
        self.assertEqual(whole.phases[-1].audio_sample_stop, 165333)
        self.assertEqual([p.to_dict() for p in active], [whole.to_dict()])
        self.assertEqual([p.to_dict() for p in transport], [whole.to_dict()])
        self.assertAlmostEqual(whole.native_frame_count / 24, 5.166666666666667)

    def test_10s_tail85_native_geometry(self):
        whole, active, transport = geometry(10)
        self.assertEqual([p.frame_count for p in whole.phases], [39] + [34] * 4 + [85])
        self.assertEqual(whole.native_frame_count, 260)
        self.assertEqual(whole.phases[-1].group_count, 5)
        self.assertEqual(whole.phases[-1].video_latent_stop, 77)
        self.assertEqual(whole.phases[-1].audio_latent_stop, 433)
        self.assertEqual(whole.phases[-1].audio_sample_stop, 346667)
        self.assertEqual([p.to_dict() for p in active], [whole.to_dict()])
        self.assertEqual([p.to_dict() for p in transport], [whole.to_dict()])
        self.assertAlmostEqual(whole.native_frame_count / 24, 10.833333333333334)

    def test_15s_tail85_canonical_whole_and_transport(self):
        whole, active, transport = geometry(15)
        self.assertEqual([p.frame_count for p in whole.phases], [39] + [34] * 8 + [85])
        self.assertEqual(whole.native_frame_count, 396)
        self.assertEqual(whole.native_frame_count / 24, 16.5)
        self.assertEqual(whole.phases[-1].group_count, 5)
        self.assertEqual(whole.phases[-1].video_latent_stop, 117)
        self.assertEqual(whole.phases[-1].audio_latent_stop, 660)
        self.assertEqual(whole.phases[-1].audio_sample_stop, 528000)
        self.assertEqual([len(p.phases) for p in active], [7, 3])
        self.assertEqual([p.native_frame_count for p in active], [243, 153])
        self.assertEqual([p.phases[-1].video_latent_stop for p in active], [72, 45])
        self.assertEqual([p.phases[-1].audio_latent_stop for p in active], [405, 255])
        self.assertEqual([p.native_frame_count for p in transport], [243, 158])
        self.assertEqual([p.phases[-1].video_latent_stop for p in transport], [72, 47])
        self.assertEqual([p.phases[-1].audio_latent_stop for p in transport], [405, 263])
        self.assertEqual([p.frame_count for p in active[1].phases], [34, 34, 85])
        self.assertEqual([p.frame_count for p in transport[1].phases], [39, 34, 85])

    def test_30s_tail85_canonical_whole_and_transport(self):
        whole, active, transport = geometry(30)
        self.assertEqual([p.frame_count for p in whole.phases], [39] + [34] * 19 + [85])
        self.assertEqual(whole.native_frame_count, 770)
        self.assertAlmostEqual(whole.native_frame_count / 24, 32.083333333333336)
        self.assertEqual(whole.phases[-1].group_count, 5)
        self.assertEqual(whole.phases[-1].video_latent_stop, 227)
        self.assertEqual(whole.phases[-1].audio_latent_stop, 1283)
        self.assertEqual(whole.phases[-1].audio_sample_stop, 1026667)
        self.assertEqual([len(p.phases) for p in active], [7, 7, 7])
        self.assertEqual([p.native_frame_count for p in active], [243, 238, 289])
        self.assertEqual([p.phases[-1].video_latent_stop for p in active], [72, 70, 85])
        self.assertEqual([p.phases[-1].audio_latent_stop for p in active], [405, 397, 481])
        self.assertEqual([p.native_frame_count for p in transport], [243, 243, 294])
        self.assertEqual([p.phases[-1].video_latent_stop for p in transport], [72, 72, 87])
        self.assertEqual([p.phases[-1].audio_latent_stop for p in transport], [405, 405, 490])

    def test_request_offsets_and_transport_prefixes_are_continuous(self):
        prefixes = {
            5: ([0], [0]),
            10: ([0], [0]),
            15: ([0, 2], [0, 8]),
            30: ([0, 2, 2], [0, 8, 9]),
        }
        for time in (5, 10, 15, 30):
            with self.subTest(time=time):
                _, schedule, active, transport = prepare_case(EXAMPLE, time)
                self.assertEqual([t.phases[-1].video_latent_stop - a.phases[-1].video_latent_stop
                                  for a, t in zip(active, transport, strict=True)], prefixes[time][0])
                self.assertEqual([t.phases[-1].audio_latent_stop - a.phases[-1].audio_latent_stop
                                  for a, t in zip(active, transport, strict=True)], prefixes[time][1])
                self.assertEqual(sum(p.native_frame_count for p in active), schedule.plan.native_frame_count)
                self.assertEqual(sum(p.phases[-1].video_latent_stop for p in active), schedule.video_latent_count)
                self.assertEqual(sum(p.phases[-1].audio_latent_stop for p in active), schedule.audio_latent_count)
                for left, right in zip(schedule.requests[:-1], schedule.requests[1:], strict=True):
                    self.assertEqual(left.frame_stop, right.frame_start)
                    self.assertEqual(left.video_latent_stop, right.video_latent_start)
                    self.assertEqual(left.audio_latent_stop, right.audio_latent_start)
                for plan in (schedule.plan, *active, *transport):
                    for left, right in zip(plan.phases[:-1], plan.phases[1:], strict=True):
                        self.assertEqual(left.frame_stop, right.frame_start)
                        self.assertEqual(left.video_latent_stop, right.video_latent_start)
                        self.assertEqual(left.audio_latent_stop, right.audio_latent_start)
                        self.assertEqual(left.audio_sample_stop, right.audio_sample_start)

    def test_only_supported_time_modes(self):
        for value in (0, 20, 30.5):
            with self.subTest(time=value), self.assertRaises(ValueError):
                geometry(value)

    def test_authored_source_speech_actions_seeds_and_conditions(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        for time in (5, 10, 15, 30):
            with self.subTest(time=time):
                case, schedule, active, transport = prepare_case(EXAMPLE, time)
                self.assertEqual(case["source_case_id"], "seated_001")
                self.assertEqual([r.seed for r in schedule.requests], [9301] * len(schedule.requests))
                self.assertEqual(len(schedule.requests), len(active))
                self.assertEqual(len(active), len(transport))
                self.assertEqual("".join(s["dialogue"] for s in case["segments"]),
                                 "".join(s["speech_text"] for s in source["segments"][:time // 5]))
                for request, segment in zip(schedule.requests, case["segments"], strict=True):
                    self.assertIn("<d>[Chinese] " + segment["dialogue"] + "</d>", request.prompt)
                    for source_id in segment["original_segment_ids"]:
                        action = source["segments"][source_id]["prompt"].split(
                            "Speaker_1's Actions: ", 1)[1].split("\n", 1)[0]
                        self.assertIn(action, request.prompt)
                self.assertEqual([c["frame_index"] for c in case["initial_conditions"]], [0, -1])
                for condition in case["initial_conditions"]:
                    self.assertTrue(Path(condition["uri"]).is_file())

    def test_5s_preserves_original_first_prompt(self):
        from digitalhuman_h3.inference.prompts import official_three_field_prompt
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        case, schedule, _, _ = prepare_case(EXAMPLE, 5)
        self.assertEqual(schedule.requests[0].prompt,
                         official_three_field_prompt(source["segments"][0]["prompt"]))
        self.assertEqual(case["segments"][0]["original_segment_ids"], [0])

    def test_10s_preserves_first_adjacent_pair_without_later_speech(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        case, schedule, _, _ = prepare_case(EXAMPLE, 10)
        _, long_schedule, _, _ = prepare_case(EXAMPLE, 30)
        request = schedule.requests[0]
        self.assertEqual(len(schedule.requests), 1)
        self.assertEqual(request.prompt, long_schedule.requests[0].prompt)
        self.assertEqual((request.phase_start, request.phase_stop), (0, 6))
        self.assertEqual((request.frame_start, request.frame_stop), (0, 260))
        self.assertEqual((request.video_latent_start, request.video_latent_stop), (0, 77))
        self.assertEqual((request.audio_latent_start, request.audio_latent_stop), (0, 433))
        self.assertEqual(case["segments"][0]["original_segment_ids"], [0, 1])
        for segment in source["segments"][2:]:
            self.assertNotIn(segment["speech_text"], request.prompt)

    def test_15s_preserves_only_first_three_source_segments(self):
        from digitalhuman_h3.inference.prompts import official_three_field_prompt
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        case, schedule, _, _ = prepare_case(EXAMPLE, 15)
        _, long_schedule, _, _ = prepare_case(EXAMPLE, 30)
        self.assertEqual([r.prompt for r in schedule.requests],
                         [long_schedule.requests[0].prompt,
                          official_three_field_prompt(source["segments"][2]["prompt"])])
        self.assertEqual([(r.phase_start, r.phase_stop) for r in schedule.requests], [(0, 7), (7, 10)])
        self.assertEqual([(r.frame_start, r.frame_stop) for r in schedule.requests], [(0, 243), (243, 396)])
        self.assertEqual([(r.video_latent_start, r.video_latent_stop) for r in schedule.requests],
                         [(0, 72), (72, 117)])
        self.assertEqual([(r.audio_latent_start, r.audio_latent_stop) for r in schedule.requests],
                         [(0, 405), (405, 660)])
        self.assertEqual([s["original_segment_ids"] for s in case["segments"]], [[0, 1], [2]])
        for segment in source["segments"][3:]:
            for request in schedule.requests:
                self.assertNotIn(segment["speech_text"], request.prompt)

    def test_missing_input_image_is_actionable(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        source["initial_conditions"][0]["uri"] = "missing-first.png"
        with tempfile.TemporaryDirectory(prefix="fl2va-test-input-") as temp:
            path = Path(temp) / "case.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                prepare_case(path, 30)

    def test_missing_source_segments_is_actionable(self):
        source = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        source["segments"] = source["segments"][:2]
        with tempfile.TemporaryDirectory(prefix="fl2va-test-input-") as temp:
            path = Path(temp) / "case.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_case(path, 30)


if __name__ == "__main__":
    unittest.main()
