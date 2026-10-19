"""Download only the official H3 FL2VA partition and FL2VA adapter; resumable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    args.models_dir.mkdir(parents=True, exist_ok=True)
    receipts = []
    for repo, folder, patterns in (
        ("MiniMaxAI/MiniMax-H3", "MiniMax-H3", ["model_index.json", "FL2VA/*"]),
        ("TaoLiveAIGC/TaoMate-H3-FL2VA", "TaoMate-H3-FL2VA",
         ["config.json", "adapter_config.json", "adapter_model.safetensors", "LICENSE", "NOTICE"]),
    ):
        info = HfApi().model_info(repo)
        destination = args.models_dir / folder
        print(f"Downloading {repo}@{info.sha} -> {destination}", flush=True)
        snapshot_download(repo_id=repo, revision=info.sha, local_dir=destination,
                          allow_patterns=patterns, max_workers=args.workers)
        receipts.append({"repo_id": repo, "revision": info.sha, "directory": str(destination)})
        (args.models_dir / "download_receipt.json").write_text(
            json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print("MODELS_READY", flush=True)


if __name__ == "__main__":
    main()
