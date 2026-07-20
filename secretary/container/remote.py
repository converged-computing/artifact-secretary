"""RemoteInspector: the Inspector method surface, but each call runs inside the
container via `docker exec`. Feeds the agent's tools unchanged."""
from __future__ import annotations

import json
from typing import Callable

# where ContainerSession docker-cp'd the package + pyelftools
PROBE_PYTHONPATH = "/tmp"

Runner = Callable[[list], "object"]


class RemoteInspector:
    def __init__(self, cid: str, runner: Runner):
        self.cid = cid
        self.runner = runner

    def _call(self, cmd: str, **kwargs) -> object:
        argv = ["docker", "exec", "-e", f"PYTHONPATH={PROBE_PYTHONPATH}", self.cid,
                "python3", "-m", "secretary.container.probe", cmd, json.dumps(kwargs)]
        p = self.runner(argv)
        if p.returncode != 0:
            return {"error": (p.stderr or p.stdout or "").strip()}
        return json.loads(p.stdout)

    def list_dir(self, path="/"): return self._call("list_dir", path=path)
    def find(self, root="/", name_glob=None, kind=None, limit=200):
        return self._call("find", root=root, name_glob=name_glob, kind=kind, limit=limit)
    def inspect_elf(self, path): return self._call("inspect_elf", path=path)
    def scan_strings(self, path, patterns, max_hits=40):
        return self._call("scan_strings", path=path, patterns=patterns, max_hits=max_hits)
    def read_text(self, path, max_bytes=256 * 1024):
        return self._call("read_text", path=path, max_bytes=max_bytes)
    def detect_provenance(self, dir_path): return self._call("detect_provenance", dir_path=dir_path)
