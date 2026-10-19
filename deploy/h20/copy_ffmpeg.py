"""Copy an installed Linux FFmpeg and its libraries to a shared tool directory."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

# Use the target host's glibc and loader; they are backward compatible with the
# old development host. Mixing two glibc installations is not supported.
GLIBC = re.compile(r"^(ld-linux|lib(c|m|pthread|dl|rt|resolv|util|anl|nss_[^.]+)\.so)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    for folder in ("bin", "lib", "libexec"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    libraries = {}
    for binary in ("ffmpeg", "ffprobe"):
        source = args.source_prefix / "bin" / binary
        dependencies = subprocess.check_output(["ldd", str(source)], text=True)
        if "not found" in dependencies:
            raise RuntimeError(f"Source {binary} has missing libraries:\n{dependencies}")
        for line in dependencies.splitlines():
            match = re.search(r"=>\s+(/\S+)", line)
            if match:
                library = Path(match.group(1))
                if not GLIBC.match(library.name):
                    libraries[library.name] = library
        shutil.copy2(source, destination / "libexec" / binary)
        wrapper = destination / "bin" / binary
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
                           'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
                           'export LD_LIBRARY_PATH="$HERE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
                           f'exec "$HERE/libexec/{binary}" "$@"\n')
        wrapper.chmod(0o755)
    for name, source in libraries.items():
        shutil.copy2(source, destination / "lib" / name)
    (destination / "source_manifest.json").write_text(json.dumps(
        {"source_prefix": str(args.source_prefix),
         "libraries": {name: str(path) for name, path in libraries.items()}}, indent=2) + "\n")
    for binary in ("ffmpeg", "ffprobe"):
        subprocess.run([str(destination / "bin" / binary), "-version"], check=True)


if __name__ == "__main__":
    main()
