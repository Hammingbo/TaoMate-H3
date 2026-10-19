# Official FL2VA case on 8 × H20 96GB

This recipe runs the unmodified `seated_001` first/last-frame case at 768×1376,
24 fps, TP2/SP4, BF16 and the official SDPA backend. FlashAttention-3 and vLLM
are required by the separate accelerated T2AV entry point, not this FL2VA recipe.

Use Linux, Python 3.10/3.11, and the official PyTorch 2.8.0 CUDA 12.8 wheel.
Install FFmpeg and FFprobe with libx264 and AAC through your distribution's
package manager. Keep the existing system Python and packages separate.

```bash
export VENV_PATH=/absolute/path/to/venvs/taomate-h3
export BASE_PYTHON=python3.10
bash deploy/h20/setup.sh

# This step can run on a CPU development host with Internet access.
# A shared filesystem makes the downloaded models available to the GPU host.
"$VENV_PATH/bin/python" deploy/h20/download_models.py --models-dir /absolute/path/to/models

# GPU host. Use the same absolute environment path on both hosts.
export MODEL_ROOT=/absolute/path/to/models/MiniMax-H3
export CHECKPOINT=/absolute/path/to/models/TaoMate-H3-FL2VA
bash deploy/h20/run_official.sh 5
bash deploy/h20/run_official.sh 30
```

For a corporate package mirror set `PIP_INDEX_URL` and `TORCH_INDEX_URL` to the
approved mirror before setup. `HF_ENDPOINT` may select an available model mirror
before starting the download process. Credentials should stay in the standard
credential store or environment, never in scripts, logs or Git.

The downloader pins the resolved Hugging Face commit for each snapshot and
records both revisions in `download_receipt.json`. Interrupted downloads can
resume by rerunning the same command.

The runner checks the environment and all eight GPUs, verifies NCCL all-reduce,
records the source commit and exact native plan, runs teacher then student, and
validates H.264/AAC streams and full decoding. Each invocation creates a fresh
output directory and writes `run.log` and `exit_code`. The native `--time 5`
output is 124 frames / 5.1667 seconds; `--time 30` is 770 frames / 32.0833 seconds.
Outputs are not cropped to the nominal mode duration.

Run detached with your usual job manager or `nohup` and retain the printed PID.
Run the short case first, then run 30 seconds after it passes. On hosts with
restricted egress, fetch code and models through a development host rather than
changing network security settings on the GPU host.

If the development host has an older glibc, prepare wheels for the GPU host
explicitly. Installing PyTorch on the CPU host can otherwise hide recent wheels
as incompatible even though they work on the GPU host.

```bash
# CPU host, Python 3.10; use a shared destination.
"$VENV_PATH/bin/python" deploy/h20/prepare_wheelhouse.py --destination /shared/wheelhouse
# GPU host, using the shared Python 3.10 environment:
WHEELHOUSE_PATH=/shared/wheelhouse bash deploy/h20/setup.sh
```

If distribution packages cannot be fetched, an existing Linux FFmpeg on the CPU
host can also be copied to shared storage. Keep the source package's licenses
with your internal tool distribution. The target Linux host must have a glibc
at least as new as the source host.

```bash
python deploy/h20/copy_ffmpeg.py --source-prefix /path/to/existing/conda/env \
  --destination /shared/taomate-media-tools
export PATH=/shared/taomate-media-tools/bin:$PATH
```

Alternatively download a complete FFmpeg/FFprobe distribution on the CPU host:

```bash
bash deploy/h20/fetch_ffmpeg.sh /shared/taomate-tools
# GPU host:
export PATH=/shared/taomate-tools/media-dist/bin:$PATH
```

This uses the BtbN Linux build linked by the FFmpeg download page, verifies the
published SHA256 checksum and retains its license files. Execute these newer
binaries on the GPU container, not the older CPU host.

To keep the final Conda environment on a shared mount while installing from an
older CPU host, first create that environment with Conda and then run:

```bash
/shared/1_code_space/envs/taomate-h3/bin/python deploy/h20/install_shared.py \
  --wheelhouse /shared/wheelhouse
# GPU host:
/shared/1_code_space/envs/taomate-h3/bin/python -m pip check
```

This explicitly installs wheels for the newer GPU platform without importing
CUDA code on the CPU host. All GPU hosts must mount the environment at the same
absolute path and provide a compatible Linux x86-64 runtime and NVIDIA driver.
