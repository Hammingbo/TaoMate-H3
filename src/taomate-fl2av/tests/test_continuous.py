"""Global geometry and exact prompt contracts for the continuous benchmark."""
import json
import tempfile
import unittest
from pathlib import Path

from fl2va.plan import prepare_case
import test_i2va


class ContinuousTests(unittest.TestCase):
    def test_twelve_prompts_share_one_native_timeline(self):
        with tempfile.TemporaryDirectory() as root:
            path, source = test_i2va.FirstImageTests().make_case(root)
            template = source['segments'][0]
            source['segments'] = [dict(template, segment_id=i, seed=100+i) for i in range(12)]
            path.write_text(json.dumps(source))
            case, schedule, active, transport = prepare_case(path, 60)
            self.assertEqual(case['task'], 'i2va')
            self.assertEqual(schedule.plan.native_frame_count, 1450)
            self.assertEqual(schedule.video_latent_count, 427)
            self.assertEqual(len(schedule.requests), 12)
            self.assertEqual([r.prompt for r in schedule.requests], [s['prompt'] for s in source['segments']])
            self.assertEqual([p.group_count for p in schedule.plan.phases if p.group_count > 2], [5])
            self.assertEqual(schedule.plan.phases[-1].group_count, 5)
            previous = None
            for i, (request, target, base) in enumerate(zip(schedule.requests, active, transport)):
                self.assertEqual(request.frame_count, target.native_frame_count)
                self.assertEqual(request.video_latent_count, target.phases[-1].video_latent_stop)
                self.assertEqual(request.audio_latent_count, target.phases[-1].audio_latent_stop)
                if previous:
                    self.assertEqual(request.frame_start, previous.frame_stop)
                    self.assertEqual(request.video_latent_start, previous.video_latent_stop)
                    self.assertEqual(request.audio_latent_start, previous.audio_latent_stop)
                    self.assertEqual(base.phases[-1].video_latent_stop-request.video_latent_count, 2)
                    self.assertIn(base.phases[-1].audio_latent_stop-request.audio_latent_count, (8,9))
                self.assertLess(abs(request.frame_start/24-i*5), 0.3)
                self.assertEqual(target.phases[0].frame_start, 0)
                for left, right in zip(target.phases, target.phases[1:]):
                    self.assertEqual(left.frame_stop, right.frame_start)
                    self.assertEqual(left.audio_latent_stop, right.audio_latent_start)
                previous = request
            self.assertEqual(sum(p.native_frame_count for p in active), 1450)
            self.assertEqual(schedule.requests[-1].audio_latent_stop, round(1450*40/24))

    def test_incomplete_or_unordered_prompt_sequences_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, source = test_i2va.FirstImageTests().make_case(root)
            with self.assertRaisesRegex(ValueError, 'twelve'):
                prepare_case(path, 60)
            source['segments'] *= 12
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'ordered'):
                prepare_case(path, 60)


if __name__ == '__main__':
    unittest.main()
