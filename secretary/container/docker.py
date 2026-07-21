"""The docker CLI as one class: a method per operation we use, so nothing else
builds raw ["docker", ...] argv by hand."""

from __future__ import annotations

from typing import Mapping, Sequence

from ..command import Command, Result


class Docker(Command):
    def has_image(self, image: str) -> bool:
        return self.run(["docker", "image", "inspect", image]).ok

    def pull(self, image: str) -> Result:
        return self.run(["docker", "pull", image])

    def resolve_digest(self, image: str) -> str:
        r = self.run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", image]
        )
        return r.text if r.ok else ""

    def run_container(
        self, image: str, flags: Sequence[str], command: Sequence[str]
    ) -> Result:
        return self.run(["docker", "run", *flags, image, *command])

    def execute(
        self,
        cid: str,
        argv: Sequence[str],
        user: str | int | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Result:
        pre: list[str] = []
        if user is not None:
            pre += ["-u", str(user)]
        for k, v in (env or {}).items():
            pre += ["-e", f"{k}={v}"]
        return self.run(["docker", "exec", *pre, cid, *argv])

    def copy(self, src: str, dest: str) -> Result:
        return self.run(["docker", "cp", src, dest])

    def remove_container(self, cid: str) -> Result:
        return self.run(["docker", "rm", "-f", cid])

    def remove_image(self, image: str) -> Result:
        return self.run(["docker", "rmi", image])

    def disconnect_network(self, network: str, cid: str) -> Result:
        return self.run(["docker", "network", "disconnect", network, cid])
