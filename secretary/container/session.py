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

If the image lacks python3, we try to install it via its package manager
(temporary network, opt out with install_python=False); otherwise the image is
skipped. Images this run pulls are
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
    install_python: bool = True   # if the image lacks python3, try to install it
    install_network: str = "bridge"  # docker network to attach for the install
    on_progress: Optional[Callable[[str], None]] = None
    _install_detail: str = ""

    cid: str = ""
    pulled_by_us: bool = False
    digest: str = ""

    def _sh(self, argv: list) -> subprocess.CompletedProcess:
        return self.runner(argv)

    def _say(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def _has_python(self) -> bool:
        return self._sh(["docker", "exec", self.cid, "python3", "--version"]).returncode == 0

    _PM_INSTALL = {
        "apt-get": "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3",
        "dnf": "dnf install -y python3",
        "yum": "yum install -y python3",
        "apk": "apk add --no-cache python3",
        "zypper": "zypper --non-interactive install python3",
    }

    def _detect_pm(self) -> str:
        probe = "||".join(f"command -v {pm}" for pm in self._PM_INSTALL) + "||true"
        out = self._sh(["docker", "exec", "-u", "0", self.cid, "sh", "-c", probe]).stdout.strip()
        found = out.splitlines()[0] if out else ""
        return os.path.basename(found) if found else ""

    def _try_install_python(self) -> bool:
        """Install python3 with the image's package manager. Needs a package
        manager and, briefly, network — so we attach a network only for the
        install and detach afterwards, keeping the probe run sealed. Runs as root
        (-u 0); the probe still runs as the image user."""
        pm = self._detect_pm()
        if not pm:
            self._install_detail = "no supported package manager (apt-get/dnf/yum/apk/zypper) in image"
            return False
        self._say(f"python3 missing — installing via {pm} (temporary network on '{self.install_network}')")

        conn = self._sh(["docker", "network", "connect", self.install_network, self.cid])
        if conn.returncode != 0:
            self._install_detail = (f"could not attach network '{self.install_network}': "
                                    + (conn.stderr or conn.stdout or "").strip())
            self._say("  " + self._install_detail)
            return False
        try:
            res = self._sh(["docker", "exec", "-u", "0", self.cid, "sh", "-c", self._PM_INSTALL[pm]])
        finally:
            self._sh(["docker", "network", "disconnect", self.install_network, self.cid])

        if res.returncode != 0:
            tail = "\n".join((res.stderr or res.stdout or "").strip().splitlines()[-8:])
            self._install_detail = f"{pm} exited {res.returncode}:\n{tail}"
            self._say(f"  install failed (exit {res.returncode}); last output:\n{tail}")
            return False
        if not self._has_python():
            self._install_detail = f"{pm} reported success but python3 is still not on PATH"
            return False
        return True

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

        if not self._has_python():
            if not (self.install_python and self._try_install_python()):
                detail = self._install_detail
                self.__exit__(None, None, None)
                msg = "no usable python3 in image"
                if self.install_python:
                    msg += " — " + (detail or "could not install one")
                raise SkipContainer(msg)

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
