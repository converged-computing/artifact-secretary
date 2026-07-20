"""A Target is *where* discovery happens: a filesystem root the inspection
library and the agent are confined to. Default is the host ("/"); it can also
be a build directory, an extracted container rootfs, or (composed) a child of
another target for nesting (container-in-VM, conda-in-container).

The Target enforces the containment that makes "let the agent look around" safe:
every path the tools touch is resolved and checked to stay under the root, and
only read operations are exposed. Even inside a sealed container this matters —
it keeps the agent from wandering out of the region we meant to give it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class OutsideTargetError(Exception):
    """Raised when a requested path escapes the target root."""


@dataclass
class Target:
    root: Path
    label: str = ""

    def __init__(self, root: str | os.PathLike, label: str = ""):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"target root is not a directory: {self.root}")
        self.label = label or str(self.root)

    def resolve(self, rel: str) -> Path:
        """Resolve a path *within* the target, following symlinks but refusing
        anything that lands outside the root. `rel` may be absolute-looking
        ('/usr/lib'); it is always interpreted relative to the target root, so
        the agent's mental model ('/usr/lib in the image') just works."""
        rel = rel.lstrip("/")
        candidate = (self.root / rel).resolve()
        # containment: the resolved real path must be the root or under it
        if candidate != self.root and self.root not in candidate.parents:
            raise OutsideTargetError(f"{rel!r} escapes target {self.root}")
        return candidate

    def display(self, p: Path) -> str:
        """Path as the agent should see it: rooted at the target, not the host."""
        try:
            return "/" + str(p.relative_to(self.root))
        except ValueError:
            return str(p)
