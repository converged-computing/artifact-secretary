"""The Inspector method surface, but every call runs inside the container via
`docker exec <python> -m secretary.container.probe`. Feeds the agent's tools
unchanged."""

from __future__ import annotations

import json

from .docker import Docker

# where ContainerSession docker-cp'd the package + pyelftools
PROBE_PYTHONPATH = "/tmp"


class RemoteInspector:
    def __init__(self, cid: str, docker: Docker, python: str = "python3"):
        self.cid = cid
        self.docker = docker
        self.python = python

    def _call(self, cmd: str, **kwargs) -> object:
        # one worker behind the public per-op methods below
        res = self.docker.execute(
            self.cid,
            [self.python, "-m", "secretary.container.probe", cmd, json.dumps(kwargs)],
            env={"PYTHONPATH": PROBE_PYTHONPATH},
        )
        if not res.ok:
            return {"error": res.text}
        return json.loads(res.stdout)

    def list_dir(self, path="/"):
        return self._call("list_dir", path=path)

    def find(self, root="/", name_glob=None, kind=None, limit=200):
        return self._call(
            "find", root=root, name_glob=name_glob, kind=kind, limit=limit
        )

    def inspect_elf(self, path):
        return self._call("inspect_elf", path=path)

    def scan_strings(self, path, patterns, max_hits=40):
        return self._call(
            "scan_strings", path=path, patterns=patterns, max_hits=max_hits
        )

    def read_text(self, path, max_bytes=256 * 1024):
        return self._call("read_text", path=path, max_bytes=max_bytes)

    def detect_provenance(self, dir_path):
        return self._call("detect_provenance", dir_path=dir_path)
