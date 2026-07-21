"""One place to run external commands, so execution is uniform and mockable.

CLI wrappers (like Docker) subclass Command; tests pass their own runner instead
of really shelling out.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class Result:
    argv: list
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def text(self) -> str:
        """stdout if we have it, else stderr, stripped."""
        return (self.stdout or self.stderr or "").strip()


def run_subprocess(argv: Sequence[str]) -> Result:
    p = subprocess.run(list(argv), capture_output=True, text=True)
    return Result(list(argv), p.returncode, p.stdout or "", p.stderr or "")


class Command:
    """Base for CLI wrappers. `runner` turns an argv into a Result and is
    swappable in tests."""

    def __init__(self, runner: Callable[[Sequence[str]], object] = run_subprocess):
        self.runner = runner

    def run(self, argv: Sequence[str]) -> Result:
        r = self.runner(list(argv))
        if isinstance(r, Result):
            return r
        # accept a subprocess.CompletedProcess-like object from injected runners
        return Result(list(argv), r.returncode, r.stdout or "", r.stderr or "")
