# Modified for the public FL2VA inference-only release: unused server/training routes removed.
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_ROOT = Path.cwd() / "models" / "MiniMax-H3"
DEFAULT_OUTPUT_ROOT = Path.cwd() / "outputs"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _absolute_env_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name, str(default))).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return path


@dataclass(frozen=True)
class H3Paths:
    model_root: Path
    output_root: Path

    @classmethod
    def from_env(cls) -> "H3Paths":
        return cls(
            model_root=_absolute_env_path("H3_MODEL_ROOT", DEFAULT_MODEL_ROOT),
            output_root=_absolute_env_path("H3_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT),
        )

    def new_output_dir(self, category: str, label: str) -> Path:
        category = _SAFE_NAME.sub("-", category).strip("-.")
        label = _SAFE_NAME.sub("-", label).strip("-.")
        if not category or not label:
            raise ValueError("output category and label must contain a safe character")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.output_root / category / f"{stamp}-{label}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(mode=0o755, parents=True, exist_ok=False)
        return run_dir

    def output_dir(self, category: str, run_id: str) -> Path:
        for value, label in ((category, "category"), (run_id, "run_id")):
            if not value or _SAFE_NAME.sub("-", value).strip("-.") != value:
                raise ValueError(f"{label} contains unsafe characters")
        return self.output_root / category / run_id
