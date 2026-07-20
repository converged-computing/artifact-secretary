"""ContainerSession: run an image as a sealed, sleeping container, copy our
(pure-Python) inspection code plus pyelftools into it, and expose a
RemoteInspector that drives that code via `docker exec`.

The code is delivered with `docker cp`, not a bind mount. Under
docker-outside-of-docker (our devcontainer talks to the host daemon), a bind
mount's source path is resolved on the HOST filesystem, not inside the
devcontainer, so `-v` would mount nothing and the probe wouldn't be found.
`docker cp` streams the files through the daemon, so it works regardless of
whose filesystem the daemon sees. Files go under /tmp (writable by any image
user) and are read via PYTHONPATH=/tmp.

Assumes a recent python3 in the image (else skip). Images this run pulls are
reaped afterward; pre-existing images are left alone.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

import elftools

from .remote import RemoteInspector, PROBE_PYTHONPATH

Runner = Callable[[list], subprocess.CompletedProcess]


def real_runner(argv: list) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True)


class SkipContainer(Exception):
    """Raised when we can't run our inspector in the image (e.g. no python3)."""


def _run_flags() -> list:
    # sealed: no network, no privileges, image's own USER (no --user, matching
    # k8s), no docker socket. entrypoint replaced with sleep so image code never runs.
    return ["--rm", "-d", "--network=none", "--cap-drop=ALL",
            "--security-opt", "no-new-privileges", "--entrypoint", "sleep"]


def _package_dir() -> str:
    # .../secretary (this file is .../secretary/container/session.py). A local
    # read, so this is reliable even for editable installs.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _elftools_dir() -> str:
    return os.path.dirname(os.path.abspath(elftools.__file__))


@dataclass
class ContainerSession:
    image: str
    runner: Runner = real_runner
    keep_images: bool = False
    on_progress: Optional[Callable[[str], None]] = None

    cid: str = ""
    pulled_by_us: bool = False
    digest: str = ""

    def _sh(self, argv: list) -> subprocess.CompletedProcess:
        return self.runner(argv)

    def _say(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def _pull(self) -> int:
        self._say(f"pulling {self.image} ...")
        # With a progress callback and the real runner, let docker's own pull
        # output stream to the terminal (layer bars) instead of being captured.
        if self.on_progress and self.runner is real_runner:
            return subprocess.run(["docker", "pull", self.image]).returncode
        return self._sh(["docker", "pull", self.image]).returncode

    def _image_present(self) -> bool:
        return self._sh(["docker", "image", "inspect", self.image]).returncode == 0

    def _resolve_digest(self) -> str:
        p = self._sh(["docker", "inspect", "--format", "{{index .RepoDigests 0}}", self.image])
        return p.stdout.strip() if p.returncode == 0 else ""

    def _copy_probe(self) -> None:
        # deliver the package + pyelftools under PROBE_PYTHONPATH (/tmp) so
        # `python3 -m secretary.container.probe` resolves inside the container.
        for src in (_package_dir(), _elftools_dir()):
            dest = f"{self.cid}:{PROBE_PYTHONPATH}/{os.path.basename(src)}"
            if self._sh(["docker", "cp", src, dest]).returncode != 0:
                raise SkipContainer(f"could not copy inspector into container: {src}")

    def __enter__(self) -> RemoteInspector:
        self.pulled_by_us = not self._image_present()
        if self.pulled_by_us and self._pull() != 0:
            raise SkipContainer(f"pull failed: {self.image}")
        self.digest = self._resolve_digest()

        self._say("starting sealed container")
        run = self._sh(["docker", "run"] + _run_flags() + [self.image, "infinity"])
        if run.returncode != 0:
            raise SkipContainer(f"run failed: {run.stderr.strip()}")
        self.cid = run.stdout.strip()

        if self._sh(["docker", "exec", self.cid, "python3", "--version"]).returncode != 0:
            self.__exit__(None, None, None)
            raise SkipContainer("no usable python3 in image")

        self._say("copying inspector into container")
        self._copy_probe()
        self._say("inspecting")
        return RemoteInspector(self.cid, self.runner)

    def __exit__(self, *exc) -> None:
        if self.cid:
            self._sh(["docker", "rm", "-f", self.cid])
            self.cid = ""
        if self.pulled_by_us and not self.keep_images:
            self._say(f"removing pulled image {self.image}")
            self._sh(["docker", "rmi", self.image])
