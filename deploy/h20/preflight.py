"""Check the eight-GPU official FL2VA recipe before loading large weights."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError("Official environment requires Python 3.10 or 3.11")
    import torch

    if torch.__version__.split("+")[0] != "2.8.0":
        raise RuntimeError(f"Expected torch 2.8.0, got {torch.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("Expose exactly eight CUDA GPUs with CUDA_VISIBLE_DEVICES")
    devices = []
    for index in range(8):
        prop = torch.cuda.get_device_properties(index)
        if prop.major != 9 or prop.total_memory < 90 * 1024**3:
            raise RuntimeError(f"GPU {index} differs from the H20 96GB recipe: {prop}")
        with torch.cuda.device(index):
            value = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
            assert (value @ value).mean().item() == 32
        devices.append({"index": index, "name": prop.name, "memory_bytes": prop.total_memory})
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"Required executable unavailable: {binary}")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"],
                                       text=True, stderr=subprocess.STDOUT)
    if not all(name in encoders.split() for name in ("libx264", "aac")):
        raise RuntimeError("FFmpeg must support libx264 and AAC")
    partition = args.model_root / "FL2VA"
    for component in ("transformer", "text_encoder", "vae", "audio_vae"):
        if not (partition / component).is_dir():
            raise RuntimeError(f"Missing model component: {partition / component}")
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.checkpoint / name).is_file():
            raise RuntimeError(f"Missing adapter file: {args.checkpoint / name}")
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "diffusers", "transformers", "numpy")}
    print(json.dumps({"python": sys.version, "packages": packages,
                      "cuda_runtime": torch.version.cuda, "devices": devices}, indent=2))


if __name__ == "__main__":
    main()
