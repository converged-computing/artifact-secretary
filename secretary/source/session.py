"""Resolve a repository to an inspectable Target: either `git clone` it into a
workspace, or point at a path that already exists, then hand back an
Inspector-surface backend the shape tools drive.

This is the source-level sibling of ContainerSession. It keeps the same posture:
the exploration the agent does is read-only and contained to the resolved tree.
The one thing that ACTS -- the clone (network + write) -- is a session step, not
an agent tool, exactly as the image pull is in ContainerSession. Whether that
step runs at all is the caller's decision (ShapeTask gates it behind confirm),
so the locate-existing path stays entirely read-only.

Two backends, same surface:
  * host      -> clone/locate on the host, yield a local Inspector.
  * container -> clone/locate INSIDE a running base image (which supplies git),
                 copy the probe in, yield a RemoteInspector. Reuses
                 ContainerSession for the copy-probe/ensure-python machinery.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..command import Command, Result, run_subprocess
from ..inspection.inspector import Inspector
from ..inspection.target import Target

Runner = Callable[[Sequence[str]], object]


class SkipRepo(Exception):
    """We can't resolve or inspect this repository (clone failed, no such path)."""


def looks_like_url(repo: str) -> bool:
    return bool(re.match(r"^(https?://|git@|ssh://|git://)", repo)) or repo.endswith(
        ".git"
    )


class Git(Command):
    def clone(self, url: str, dest: str, ref: Optional[str] = None) -> Result:
        argv = ["git", "clone", "--depth", "1"]
        if ref:
            argv += ["--branch", ref]
        return self.run(argv + [url, dest])

    def head_commit(self, path: str) -> str:
        r = self.run(["git", "-C", path, "rev-parse", "HEAD"])
        return r.text if r.ok else ""


@dataclass
class SourceSession:
    """Host-side clone/locate. `repo` is a URL (cloned) or an existing path
    (located). `ref` selects a branch/tag when cloning."""

    repo: str
    ref: Optional[str] = None
    workdir: Optional[str] = None  # where clones land; a temp dir by default
    runner: Runner = run_subprocess
    keep_clone: bool = False
    on_progress: Optional[Callable[[str], None]] = None

    commit: str = ""
    path: str = ""  # resolved tree root
    cloned_by_us: bool = False
    _tmp: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    def __post_init__(self):
        self.git = Git(self.runner)

    @property
    def will_clone(self) -> bool:
        return looks_like_url(self.repo)

    def report(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def __enter__(self) -> Inspector:
        if self.will_clone:
            if self.workdir:
                dest = os.path.join(self.workdir, "repo")
            else:
                self._tmp = tempfile.TemporaryDirectory(prefix="artifact-secretary-")
                dest = os.path.join(self._tmp.name, "repo")
            self.report(f"cloning {self.repo}" + (f" @ {self.ref}" if self.ref else ""))
            r = self.git.clone(self.repo, dest, self.ref)
            if not r.ok:
                raise SkipRepo(f"clone failed: {r.text}")
            self.cloned_by_us = True
            self.path = dest
        else:
            if not os.path.isdir(self.repo):
                raise SkipRepo(f"no such directory: {self.repo}")
            self.path = os.path.abspath(self.repo)
            self.report(f"using existing tree {self.path}")
        self.commit = self.git.head_commit(self.path)
        self.report("inspecting")
        return Inspector(Target(self.path))

    def __exit__(self, *exc) -> None:
        if self._tmp and not self.keep_clone:
            self._tmp.cleanup()
            self._tmp = None


@dataclass
class ContainerSourceSession:
    """Clone/locate INSIDE a running base image. The base image supplies git and
    (for cloning) needs the network on, so this is the one place we run with
    --allow-network by default. Reuses ContainerSession for start/copy-probe/
    ensure-python; adds the clone step and roots the agent at the clone path.
    """

    repo: str
    base_image: str
    ref: Optional[str] = None
    clone_dir: str = "/tmp/artifact-secretary/repo"
    runner: Runner = run_subprocess
    keep_images: bool = False
    on_progress: Optional[Callable[[str], None]] = None

    commit: str = ""
    path: str = ""
    cloned_by_us: bool = False

    def __post_init__(self):
        # imported here so the SDK-free host path doesn't pull the container deps
        from ..container.session import ContainerSession

        self._session = ContainerSession(
            self.base_image,
            runner=self.runner,
            keep_images=self.keep_images,
            allow_network=True,  # cloning needs it; inspection still never does
        )
        self._session.on_progress = self.on_progress

    @property
    def will_clone(self) -> bool:
        return looks_like_url(self.repo)

    def report(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def __enter__(self):
        backend = self._session.__enter__()  # RemoteInspector over the base image
        docker, cid = self._session.docker, self._session.cid
        if self.will_clone:
            self.report(f"cloning {self.repo} into container")
            argv = ["git", "clone", "--depth", "1"]
            if self.ref:
                argv += ["--branch", self.ref]
            r = docker.execute(cid, argv + [self.repo, self.clone_dir])
            if not r.ok:
                self._session.__exit__(None, None, None)
                raise SkipRepo(f"clone failed in container: {r.text}")
            self.cloned_by_us = True
            self.path = self.clone_dir
        else:
            self.path = self.repo  # a path already present in the base image
        head = docker.execute(cid, ["git", "-C", self.path, "rev-parse", "HEAD"])
        self.commit = head.text if head.ok else ""
        return backend

    def __exit__(self, *exc) -> None:
        self._session.__exit__(*exc)
