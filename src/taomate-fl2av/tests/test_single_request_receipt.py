"""Request zero must record its already-bound first/end keyframe indices."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class ReceiptTest(unittest.TestCase):
    def test_zero_tail_request_keeps_visibility_record(self):
        root = Path(__file__).resolve().parents[1]
        source = ast.parse((root/'src/digitalhuman_h3/streaming/runtime.py').read_text())
        overlay = next(n for n in source.body if isinstance(n, ast.ClassDef)
            and n.name == 'H3StreamingRuntimeOverlay')
        run = next(n for n in overlay.body if isinstance(n, ast.FunctionDef) and n.name == '_run')
        value = next(n.value for n in ast.walk(run) if isinstance(n, ast.keyword)
            and n.arg == 'clip_keyframe_visible_frame_indices')
        expression = ast.Expression(body=value)
        context = dict(self=SimpleNamespace(_clip_tail_start_request=0),
            clip_keyframes=SimpleNamespace(keyframe_frame_indices=(0, -1)))
        self.assertEqual(eval(compile(expression, '<visibility-receipt>', 'eval'), context), (0, -1))


if __name__ == '__main__':
    unittest.main()
