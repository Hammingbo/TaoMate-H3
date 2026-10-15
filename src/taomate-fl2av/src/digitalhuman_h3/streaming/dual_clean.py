"""Native LOW clean prediction and resolution-local clean-KV routing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class NativeLowCleanCapture:
    """Capture the last LOW x0 estimate before the Euler state update."""

    def __init__(self, *, target_slice: slice, sigmas: list[float]) -> None:
        self.target_slice = target_slice
        self.sigmas = sigmas
        self.video_rows: Any = None
        self.clean_video_rows: Any = None

    def on_initial_state(self, video_rows: Any, audio_rows: Any) -> None:
        self.video_rows = video_rows

    def on_velocity(self, step: int, video_velocity: Any, audio_velocity: Any) -> None:
        if step == len(self.sigmas) - 2:
            # H3 velocity is clean - noise: x0 = x_sigma + sigma * velocity.
            self.clean_video_rows = self.video_rows[self.target_slice].float().add(
                video_velocity.float(), alpha=float(self.sigmas[step])
            ).detach()


@contextmanager
def resolution_local_session(session: Any, *, restore: Any) -> Iterator[Any]:
    """Suspend HIGH's hook registration without changing its session or cache."""

    from .capability import set_official_attention_hook

    set_official_attention_hook(None)
    try:
        with session:
            yield session
    finally:
        set_official_attention_hook(restore.hook)
