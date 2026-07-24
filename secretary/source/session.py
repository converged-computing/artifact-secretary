"""Get a repository into a state we can inspect.

Clone it or point at a tree that exists, then hand back a backend the shape tools
drive. Exploration is read only and stays inside the tree. The clone is the one
thing that acts, and it is a session step rather than an agent tool so the caller
can gate it.

Host yields a local Inspector. Container does the same inside a base image that
supplies git and yields a RemoteInspector.
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

# default sandbox base. needs git already there since we never apt-get inside,
# and glibc because the bundled python would not run on musl
DEFAULT_BASE_IMAGE = "buildpack-deps:bookworm-scm"


SHA_RX = re.compile(r"^[0-9a-f]{7,40}$")

# rev-parse alone will not do. it echoes HEAD back on stdout when it cannot
# resolve, and git config global needs a HOME that docker exec may not set. so try
# a few strategies, pass safe.directory inline, and report what each one said
_COMMIT_STRATEGIES: tuple[tuple[str, list[str]], ...] = (
    ("rev-parse", ["rev-parse", "HEAD"]),
    ("log", ["log", "-1", "--format=%H"]),
    ("rev-parse-verify", ["rev-parse", "--verify", "HEAD"]),
    ("show-ref-head", ["show-ref", "--hash", "--head", "HEAD"]),
)


def resolve_commit(run, path: str) -> tuple[str, str]:
    """(commit, error). Empty error means the revision is pinned."""
    tried = []
    for name, argv in _COMMIT_STRATEGIES:
        r = run(["git", "-c", f"safe.directory={path}", "-C", path] + argv)
        out = (getattr(r, "stdout", "") or "").strip()
        err = (getattr(r, "stderr", "") or "").strip()
        first = out.split("\n")[0].strip() if out else ""
        if getattr(r, "ok", False) and SHA_RX.match(first):
            return first, ""
        rc = getattr(r, "returncode", "?")
        tried.append(
            f"{name}: rc={rc} out={first[:40]!r} err={err.splitlines()[0][:80]!r}"
            if err
            else f"{name}: rc={rc} out={first[:40]!r}"
        )
    return "", "; ".join(tried)


class SkipRepo(Exception):
    """We cannot resolve or inspect this repository."""


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

    def head_commit(self, path: str) -> tuple[str, str]:
        """(commit, error) via the shared multi-strategy resolver."""
        return resolve_commit(self.run, path)


@dataclass
class SourceSession:
    """Clone or locate on this machine. The repo is either a url we clone or a
    path that already exists, and ref picks a branch or tag when cloning."""

    repo: str
    ref: Optional[str] = None
    workdir: Optional[str] = None  # where clones land, a temp dir unless told otherwise
    runner: Runner = run_subprocess
    keep_clone: bool = False
    on_progress: Optional[Callable[[str], None]] = None

    commit: str = ""
    commit_error: str = ""  # why the revision would not resolve, when it did not
    path: str = ""  # resolved tree root
    cloned_by_us: bool = False
    _tmp: Optional[tempfile.TemporaryDirectory] = field(default=None, repr=False)

    def __post_init__(self):
        self.git = Git(self.runner)

    @property
    def will_clone(self) -> bool:
        return looks_like_url(self.repo)

    @property
    def root(self) -> str:
        # on the host the Target is the clone itself so backend paths start at slash
        return "/"

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
        self.commit, self.commit_error = self.git.head_commit(self.path)
        if self.commit_error:
            self.report(f"warning: could not resolve commit ({self.commit_error})")
        self.report("inspecting")
        return Inspector(Target(self.path))

    def __exit__(self, *exc) -> None:
        if self._tmp and not self.keep_clone:
            self._tmp.cleanup()
            self._tmp = None


@dataclass
class ContainerSourceSession:
    """Clone or locate inside a running base image. The image supplies git and
    cloning needs the network, so this is the one place we run with it on. Reuses
    ContainerSession for start and probe copy and adds the clone step.
    """

    repo: str
    base_image: str = DEFAULT_BASE_IMAGE
    ref: Optional[str] = None
    clone_dir: str = "/tmp/artifact-secretary/repo"
    runner: Runner = run_subprocess
    keep_images: bool = False
    on_progress: Optional[Callable[[str], None]] = None

    commit: str = ""
    commit_error: str = ""  # why the revision would not resolve, when it did not
    path: str = ""
    cloned_by_us: bool = False

    def __post_init__(self):
        # imported here so the host path does not drag in the container deps
        from ..container.session import ContainerSession

        # an unset base image used to reach docker as None and blow up deep in
        # subprocess, so catch it here where we can say something useful
        if not self.base_image:
            self.base_image = DEFAULT_BASE_IMAGE
        self._session = ContainerSession(
            self.base_image,
            runner=self.runner,
            keep_images=self.keep_images,
            allow_network=True,  # cloning needs the network, inspection still never does
        )
        self._session.on_progress = self.on_progress

    @property
    def will_clone(self) -> bool:
        return looks_like_url(self.repo)

    @property
    def root(self) -> str:
        # in the container the backend starts at slash so use the full clone path
        return self.path or self.clone_dir

    def report(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def __enter__(self):
        backend = self._session.__enter__()  # RemoteInspector over the base image
        docker, cid = self._session.docker, self._session.cid
        if self.will_clone and not docker.execute(cid, ["git", "--version"]).ok:
            # we never install into the sandbox, so a base without git is a dead end
            self._session.__exit__(None, None, None)
            raise SkipRepo(
                f"no git in base image {self.base_image!r}; use one that ships git "
                f"(default: {DEFAULT_BASE_IMAGE})"
            )
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
        self.commit, self.commit_error = resolve_commit(
            lambda argv: docker.execute(cid, argv), self.path
        )
        if self.commit_error:
            self.report(f"warning: could not resolve commit; {self.commit_error}")
        return backend

    def __exit__(self, *exc) -> None:
        self._session.__exit__(*exc)
