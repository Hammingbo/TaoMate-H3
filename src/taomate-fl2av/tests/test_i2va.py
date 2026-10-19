"""Single-image contracts: no fabricated end reference, exact custom dialogue."""
import json
import tempfile
import unittest
from pathlib import Path

from fl2va.plan import prepare_case
from fl2va.teacher import visual_contract

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/seated_001/case.json'


class FirstImageTests(unittest.TestCase):
    def make_case(self, root):
        source = json.loads(EXAMPLE.read_text())
        image = EXAMPLE.parent / source['initial_conditions'][0]['uri']
        source['initial_conditions'] = [dict(source['initial_conditions'][0], uri=str(image))]
        source['segments'] = [dict(segment_id=0, seed=123, speech_text='请看画面中的商品。',
            prompt='For the target video, <Picture 1> is the start image.\n\n'
            'integrated_multimodal_description: [Shot 1] The presenter in <Picture 1> '
            'speaks clearly (S1): <d>[Chinese]请看画面中的商品。</d> '
            'They make a small gesture over 5.00 seconds.\n\n'
            'overall_soundscape: Quiet room tone.\n\nnon_diegetic_music: N/A')]
        path = Path(root)/'case.json'
        path.write_text(json.dumps(source))
        return path, source

    def test_one_custom_five_second_prompt_has_no_tail(self):
        with tempfile.TemporaryDirectory() as root:
            path, source = self.make_case(root)
            case, schedule, _, _ = prepare_case(path, 5)
            self.assertEqual(case['task'], 'i2va')
            self.assertEqual(schedule.requests[0].prompt, source['segments'][0]['prompt'])
            self.assertEqual([c['frame_index'] for c in case['initial_conditions']], [0])
            self.assertEqual(visual_contract(case, schedule)['tail_visibility'], 'none')
            self.assertIsNone(visual_contract(case, schedule)['tail_start_request_index'])

    def test_missing_end_image_cannot_be_referenced_in_text(self):
        with tempfile.TemporaryDirectory() as root:
            path, source = self.make_case(root)
            source['segments'][0]['prompt'] += ' Reach Picture 2.'
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'Picture 2'):
                prepare_case(path, 5)

    def test_dialogue_is_not_silently_rewritten(self):
        with tempfile.TemporaryDirectory() as root:
            path, source = self.make_case(root)
            source['segments'][0]['speech_text'] = '另一个句子。'
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'speech metadata'):
                prepare_case(path, 5)

    def test_teacher_layout_encodes_only_one_image(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch dependency is available on the GPU host')
        from fl2va.teacher import teacher_packed
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.make_case(root)
            _, schedule, _, _ = prepare_case(path, 5)
            packed = teacher_packed(torch, text_len=64, latent_h=86, latent_w=48,
                request=schedule.requests[0], schedule=schedule, has_end_frame=False)
            positions = packed['img_position_ids'][packed['img_pos'][~packed['update_mask']], 0]
            self.assertEqual(positions.numel(), 1032)
            self.assertTrue((positions == 64).all().item())


if __name__ == '__main__':
    unittest.main()
