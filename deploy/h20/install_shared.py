"""Install GPU wheels into a shared Python 3.10 environment from an older CPU host.

Run with the destination Conda environment's Python. No CUDA library is imported
on the CPU host. The resulting binary packages are for the newer GPU container.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from prepare_wheelhouse import project_requirements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Run with the destination Conda Python 3.10")
    site = Path(sysconfig.get_path("purelib"))
    command = [sys.executable, "-m", "pip", "install", "--target", str(site),
               "--upgrade", "--only-binary=:all:", "--no-index", "--find-links",
               str(args.wheelhouse), "--python-version", "310", "--implementation", "cp",
               "--abi", "cp310"]
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
                     "manylinux2014_x86_64", "manylinux_2_17_x86_64", "linux_x86_64"):
        command += ["--platform", platform]
    subprocess.run(command + project_requirements() + ["setuptools>=69", "wheel"], check=True)
    # --target puts console scripts under site-packages/bin. Their shebangs
    # already name the destination interpreter, so expose them in env/bin.
    scripts = site / "bin"
    if scripts.is_dir():
        for script in scripts.iterdir():
            if script.is_file():
                shutil.copy2(script, Path(sys.prefix) / "bin" / script.name)
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                    "--no-build-isolation", "-e", str(root)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"SHARED_ENVIRONMENT_READY: {sys.prefix}", flush=True)


if __name__ == "__main__":
    main()
