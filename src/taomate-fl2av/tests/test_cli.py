"""The launcher and both worker stages derive fixed-tail85 plans from --time."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_cli():
    spec = importlib.util.spec_from_file_location("fl2va_cli", ROOT / "infer.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


class CliTests(unittest.TestCase):
    def test_dry_run_all_time_modes_use_tail85(self):
        expected = {5: (124, [2]), 10: (260, [6]), 15: (396, [7, 3]), 30: (770, [7, 7, 7])}
        cli = load_cli()
        for time, (frames, request_blocks) in expected.items():
            with self.subTest(time=time):
                output = io.StringIO()
                with patch.object(sys, "argv", ["infer.py", "--time", str(time), "--dry-run"]), \
                        contextlib.redirect_stdout(output):
                    cli.main()
                plan = json.loads(output.getvalue())
                phases = plan["whole"]["phases"]
                self.assertEqual(plan["whole"]["native_frame_count"], frames)
                self.assertEqual(phases[-1]["frame_stop"] - phases[-1]["frame_start"], 85)
                self.assertEqual(phases[-1]["group_count"], 5)
                self.assertEqual([len(p["phases"]) for p in plan["active"]], request_blocks)
                self.assertEqual(plan["transport"][-1]["phases"][-1]["group_count"], 5)
                self.assertEqual([r["seed"] for r in plan["requests"]], [9301] * len(request_blocks))

    def test_removed_tail_frames_option_is_rejected(self):
        cli = load_cli()
        for value in ("68", "85"):
            with self.subTest(value=value):
                error = io.StringIO()
                with patch.object(sys, "argv", ["infer.py", "--time", "15", "--tail-frames", value, "--dry-run"]), \
                        contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                    cli.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("unrecognized arguments: --tail-frames", error.getvalue())

    def test_torchrun_two_stages_forward_only_time_for_geometry(self):
        cli = load_cli()
        for time in (5, 10, 15, 30):
            with self.subTest(time=time), tempfile.TemporaryDirectory(prefix="fl2va-cli-test-") as temp:
                argv = ["infer.py", "--time", str(time), "--model-root", temp,
                        "--checkpoint", temp, "--output", str(Path(temp) / "output")]
                with patch.object(sys, "argv", argv), patch.dict(os.environ, {}), \
                        patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
                    cli.main()
                self.assertEqual(run.call_count, 2)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual([cmd[-1] for cmd in commands], ["teacher", "student"])
                for command in commands:
                    self.assertEqual(command[command.index("--time") + 1], str(time))
                    self.assertNotIn("--tail-frames", command)

    def test_worker_stages_receive_tail85_schedule_for_each_time(self):
        cli = load_cli()
        expected = {5: 124, 10: 260, 15: 396, 30: 770}
        for time, frames in expected.items():
            for stage in ("teacher", "student"):
                with self.subTest(time=time, stage=stage), \
                        tempfile.TemporaryDirectory(prefix="fl2va-cli-test-") as temp:
                    argv = ["infer.py", "--time", str(time), "--stage", stage,
                            "--model-root", temp, "--checkpoint", temp,
                            "--output", str(Path(temp) / "output")]
                    with patch.object(sys, "argv", argv), \
                            patch.dict(os.environ, {"WORLD_SIZE": "8", "RANK": "0"}), \
                            patch(f"fl2va.{stage}.run_{stage}") as worker:
                        cli.main()
                    worker.assert_called_once()
                    schedule = worker.call_args.args[2]
                    self.assertEqual(schedule.plan.native_frame_count, frames)
                    self.assertEqual(schedule.plan.phases[-1].frame_count, 85)
                    self.assertEqual(schedule.plan.phases[-1].group_count, 5)

    def test_inherited_single_rank_environment_does_not_select_worker(self):
        cli = load_cli()
        with tempfile.TemporaryDirectory(prefix="fl2va-cli-test-") as temp:
            path = Path(temp)
            teacher = path / "teacher"
            teacher.mkdir()
            (teacher / "complete.json").write_text("{}")
            argv = ["infer.py", "--time", "5", "--model-root", temp,
                    "--checkpoint", temp, "--output", str(path / "output"), "--teacher-root", str(teacher)]
            with patch.object(sys, "argv", argv), patch.dict(os.environ,
                    {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}), \
                    patch("subprocess.run") as run, contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(run.call_count, 1)
            env = run.call_args.kwargs["env"]
            self.assertNotIn("RANK", env)
            self.assertNotIn("LOCAL_RANK", env)
            self.assertNotIn("WORLD_SIZE", env)
            self.assertEqual(run.call_args.args[0][-2:], ["--stage", "student"])


if __name__ == "__main__":
    unittest.main()
