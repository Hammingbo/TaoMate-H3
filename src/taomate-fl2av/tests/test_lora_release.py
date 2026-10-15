"""Public adapters preserve the online bank and the server streaming contract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from digitalhuman_h3.inference.adapter_contract import MODEL_REVISION, SOURCE_REVISION
from digitalhuman_h3.inference.lora_checkpoint import (
    H3LoRACheckpointError, canonical_h3_lora_targets, load_h3_lora_checkpoint,
)
from digitalhuman_h3.runtime.direct import _stage3_extended_streaming_contract

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    torch = None


def release_config():
    return dict(
        rank=2, alpha=2.0, weight_source="generator", optimizer_step=300,
        stage="stage3", model_revision=MODEL_REVISION, source_revision=SOURCE_REVISION,
        runtime_contract=dict(
            full_prefix=dict(enabled=True, half_life_seconds=60, modality="video_only",
                schema="digitalhuman_h3.training.stage3_full_prefix.v1"),
            self_forcing=dict(keyframe_condition_policy="clip_absolute_persistent",
                phase_latent_handoff=dict(mode="off")),
        ),
    )


class StreamingContractTests(unittest.TestCase):
    def test_release_and_training_checkpoint_preserve_streaming_settings(self):
        with tempfile.TemporaryDirectory(prefix="fl2va-adapter-test-") as temp:
            root = Path(temp)
            config = release_config()
            (root / "complete.json").write_text(json.dumps(dict(contract=config)))
            native = _stage3_extended_streaming_contract(root)
            (root / "complete.json").unlink()
            (root / "adapter_model.safetensors").touch()
            (root / "config.json").write_text(json.dumps(config))
            self.assertEqual(_stage3_extended_streaming_contract(root), native)
            self.assertEqual(native, (60.0, "off", "video_only", "clip_absolute_persistent"))
            (root / "config.json").rename(root / "adapter_config.json")
            self.assertEqual(_stage3_extended_streaming_contract(root), native)


@unittest.skipIf(torch is None, "Adapter checks require Torch and safetensors")
class AdapterReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="fl2va-adapter-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = release_config()
        self.state = {}
        for index, target in enumerate(canonical_h3_lora_targets()):
            self.state[f"{target}.lora_a"] = torch.full((2, 3), float(index))
            self.state[f"{target}.lora_b"] = torch.full((4, 2), float(index) + 0.5)

    def save_public(self, state=None):
        (self.root / "config.json").write_text(json.dumps(self.config))
        save_file(self.state if state is None else state,
            str(self.root / "adapter_model.safetensors"))

    def test_training_to_public_round_trip_keeps_exact_online_tensors(self):
        native = self.root / "native"
        native.mkdir()
        payload = native / "adapter_optimizer.pt"
        model = {f"{bank}.{name}": tensor + offset
            for bank, offset in (("generator", 0), ("fake_score", 1000), ("anchor_score", 2000))
            for name, tensor in self.state.items()}
        torch.save(dict(model=model, optimizer={}), payload)
        contract = dict(self.config, trainable_mode="lora",
            trainable_config=dict(mode="lora", lora=dict(rank=2, alpha=2.0)),
            objective=dict(anchor_video_weight=1))
        (native / "complete.json").write_text(json.dumps(dict(
            status="complete", qualification_status="intermediate_restartable",
            optimizer_step=300, contract=contract,
            files={"adapter_optimizer.pt": dict(size_bytes=payload.stat().st_size)},
        )))
        original = load_h3_lora_checkpoint(native, allow_intermediate=True)
        self.save_public(dict(original.state))
        public = load_h3_lora_checkpoint(self.root)
        self.assertEqual((public.role, public.optimizer_step, public.rank, public.alpha),
            ("generator", 300, 2, 2.0))
        self.assertEqual((public.loaded_tensor_count, public.missing_tensor_count,
            public.redundant_tensor_count), (416, 0, 0))
        self.assertEqual(set(public.state), set(original.state))
        for name, tensor in public.state.items():
            self.assertTrue(torch.equal(tensor, original.state[name]), name)
            self.assertTrue(torch.equal(tensor, self.state[name]), name)
        self.assertEqual(_stage3_extended_streaming_contract(self.root),
            _stage3_extended_streaming_contract(native))

    def test_t2av_ema_adapter_is_rejected(self):
        self.config["weight_source"] = "generator_ema"
        self.save_public()
        with self.assertRaisesRegex(H3LoRACheckpointError, "online generator"):
            load_h3_lora_checkpoint(self.root)

    def test_incomplete_public_inventory_is_rejected(self):
        state = dict(self.state)
        state.pop(next(iter(state)))
        self.save_public(state)
        with self.assertRaisesRegex(H3LoRACheckpointError, "missing=1"):
            load_h3_lora_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
