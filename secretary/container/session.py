"""Run an image as a sealed, sleeping container, copy our inspection code (and,
when needed, a python to run it) into it, and hand back a RemoteInspector that
drives that code via `docker exec`.

Everything is delivered with `docker cp`, not a bind mount: under
docker-outside-of-docker the daemon resolves a bind-mount source on the host, not
in our devcontainer, so `-v` would mount nothing. Files land in /tmp.

We don't rely on the image's python. If it's missing or older than we need, we
bring our own — a relocatable standalone CPython copied in (see runtime.py). That
needs no network in the container, no root, no package manager, so it works in
the default sealed posture. --allow-network only affects the container's network,
which the inspection itself never uses.

Pulled images are reaped on exit; images already present are left alone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import elftools

from ..command import run_subprocess
from . import runtime
from .docker import Docker
from .remote import PROBE_PYTHONPATH, RemoteInspector

Runner = Callable[[Sequence[str]], object]


class SkipContainer(Exception):
    """We can't run our inspector in this image (e.g. no usable python)."""


@dataclass
class ContainerSession:
    image: str
    runner: Runner = run_subprocess
    keep_images: bool = False
    install_python: bool = True  # bring our own python when the image's is unusable
    allow_network: bool = False  # run the container on a network (posture only)
    network: str = "bridge"
    on_progress: Optional[Callable[[str], None]] = None

    cid: str = ""
    pulled_by_us: bool = False
    digest: str = ""
    install_detail: str = ""  # why we couldn't provide a python, for the skip reason
    probe_python: str = (
        "python3"  # interpreter the probe runs under (image's, or bundled)
    )

    def __post_init__(self):
        self.docker = Docker(self.runner)

    @property
    def package_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def elftools_dir(self) -> str:
        return os.path.dirname(os.path.abspath(elftools.__file__))

    @property
    def run_flags(self) -> list:
        net = self.network if self.allow_network else "none"
        flags = ["--rm", "-d", f"--network={net}"]
        if not self.allow_network:
            flags += ["--cap-drop=ALL", "--security-opt", "no-new-privileges"]
        return flags + ["--entrypoint", "sleep"]

    def report(self, msg: str) -> None:
        if self.on_progress:
            self.on_progress(msg)

    def probe_runs_under(self, python: str) -> bool:
        """Can this interpreter actually import and run our probe? Catches a
        missing python, one too old, or one whose stdlib can't parse our copied
        deps (pyelftools uses `match`, so it needs 3.10+)."""
        r = self.docker.execute(
            self.cid,
            [python, "-c", "import secretary.container.probe"],
            env={"PYTHONPATH": PROBE_PYTHONPATH},
        )
        return r.ok

    def container_arch(self) -> str:
        r = self.docker.execute(self.cid, ["uname", "-m"])
        return runtime.normalize_arch(r.text if r.ok else "")

    def provide_python(self) -> bool:
        """docker cp a bundled standalone python in and point the probe at it."""
        arch = self.container_arch()
        try:
            local = runtime.fetch_python(arch)
        except Exception as e:
            self.install_detail = f"could not fetch a standalone python ({e})"
            return False
        src = os.path.join(local, "python")
        if not self.docker.copy(src, f"{self.cid}:{runtime.BUNDLED_ROOT}").ok:
            self.install_detail = "could not copy the bundled python into the container"
            return False
        self.probe_python = f"{runtime.BUNDLED_ROOT}/bin/python3"
        self.report(f"using a bundled python3 ({runtime.PBS_VERSION})")
        return True

    def copy_probe(self) -> None:
        for src in (self.package_dir, self.elftools_dir):
            dest = f"{self.cid}:{PROBE_PYTHONPATH}/{os.path.basename(src)}"
            if not self.docker.copy(src, dest).ok:
                raise SkipContainer(f"could not copy inspector into container: {src}")

    def start(self) -> None:
        net = self.network if self.allow_network else "none"
        self.report(
            f"starting container ({'network ' + net if self.allow_network else 'sealed, no network'})"
        )
        run = self.docker.run_container(self.image, self.run_flags, ["infinity"])
        if not run.ok:
            raise SkipContainer(f"run failed: {run.text}")
        self.cid = run.text

    def ensure_python(self) -> None:
        # requires the probe code to already be copied in (see __enter__)
        if self.probe_runs_under("python3"):
            return  # the image's own python can run the probe
        if (
            self.install_python
            and self.provide_python()
            and self.probe_runs_under(self.probe_python)
        ):
            return  # brought our own that can
        why = self.install_detail or (
            "image python can't run the inspector (missing, too old, or incompatible)"
            if self.install_python
            else "image python unusable and bundling disabled"
        )
        self.__exit__(None, None, None)
        raise SkipContainer(f"no usable python3 in image — {why}")

    def __enter__(self) -> RemoteInspector:
        self.pulled_by_us = not self.docker.has_image(self.image)
        if self.pulled_by_us:
            self.report(f"pulling {self.image} ...")
            if not self.docker.pull(self.image).ok:
                raise SkipContainer(f"pull failed: {self.image}")
        self.digest = self.docker.resolve_digest(self.image)

        self.start()
        self.report("copying inspector into container")
        self.copy_probe()
        self.ensure_python()  # pick an interpreter that can actually run it
        self.report("inspecting")
        return RemoteInspector(self.cid, self.docker, self.probe_python)

    def __exit__(self, *exc) -> None:
        if self.cid:
            self.docker.remove_container(self.cid)
            self.cid = ""
        if self.pulled_by_us and not self.keep_images:
            self.report(f"removing pulled image {self.image}")
            self.docker.remove_image(self.image)
