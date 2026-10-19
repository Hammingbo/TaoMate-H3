"""Download GPU-host wheels on an older Linux CPU host without installing them."""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path


def project_requirements() -> list[str]:
    source = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    match = re.search(r"(?ms)^dependencies\s*=\s*(\[.*?^\])", source)
    if match is None:
        raise RuntimeError("Cannot locate project dependency list")
    return ast.literal_eval(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--index-url", default="https://pypi.org/simple")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Run with Python 3.10 for this cp310 wheelhouse")
    requirements = project_requirements()
    args.destination.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
               "--python-version", "310", "--implementation", "cp", "--abi", "cp310",
               "--dest", str(args.destination), "--index-url", args.index_url]
    # The target H20 container uses a newer glibc than the CPU development host.
    for platform in ("manylinux_2_28_x86_64", "manylinux_2_27_x86_64",
                     "manylinux2014_x86_64", "manylinux_2_17_x86_64", "linux_x86_64"):
        command += ["--platform", platform]
    subprocess.run(command + requirements + ["setuptools>=69", "wheel"], check=True)
    print("WHEELHOUSE_READY", flush=True)


if __name__ == "__main__":
    main()
