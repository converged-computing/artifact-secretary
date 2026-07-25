"""ProfileTask: the first Task. Elicits a catalog + goal, then characterizes
each container into the digest-keyed manifest lookup. Read-only; no setup
approval needed (that gate is for tasks that act, like a later submit task).
Iterates the catalog, so it overrides execute() and builds tools per image over
that container's RemoteInspector."""

from __future__ import annotations

from typing import Any, Callable, Optional

from ..container.session import ContainerSession, SkipContainer
from ..catalog import host_arch, host_supports
from behalf import AgentRunner, ConfirmFn, Task
from ..framework.tools import inspection_tools
from ..model.manifest import LookupEntry, ManifestLookup, Reproduce

CHARACTERIZE = """You characterize a compiled HPC application inside a container
so a scheduler can decide where it can run. Use the read-only tools to LOCATE
the application binary (find/list_dir under /opt, /usr/local, /usr), determine
what it is COMPILED AGAINST with inspect_elf (arch + linked libraries reveal GPU
via libcudart/libamdhip64, interconnect via libfabric/libefa, and MPI flavor),
and determine PROVENANCE with detect_provenance/read_text/scan_strings. If the
image carries MORE THAN ONE build (e.g. CPU and CUDA LAMMPS), record EACH as a
separate variant with record_artifact. Never guess linkage; read it."""

SETUP = """You are setting up a container-profiling run. You already know the
output is a per-image-digest lookup of hardware facts (arch, linkage, capability,
provenance) — do NOT ask the user about its structure. Ask only what you cannot
infer: which container images to profile (references, ideally by digest), the
goal/what they care about characterizing, and whether to keep pulled images.
When you have those, call finalize_setup with a manifest:
{"catalog": [<image refs>], "goal": <string>, "keep_images": <bool>}."""


def _default_session(ref: str, keep_images: bool):
    return ContainerSession(ref, keep_images=keep_images)


class ProfileTask(Task):
    name = "profile"
    requires_setup_approval = False

    def __init__(self, session_factory: Optional[Callable[[str, bool], Any]] = None):
        # session_factory(ref, keep_images) -> context manager yielding a backend
        # (Inspector-like), and exposing .digest / .pulled_by_us. Injectable for tests.
        self.session_factory = session_factory or _default_session

    def setup_system_prompt(self) -> str:
        return SETUP

    def manifest_schema(self) -> dict:
        return {"catalog": [str], "goal": str, "keep_images": bool}

    def execute_system_prompt(self, manifest: dict) -> str:
        return CHARACTERIZE

    async def execute(
        self, runner: AgentRunner, manifest: dict, confirm_fn: ConfirmFn
    ) -> ManifestLookup:
        from behalf import console

        lookup = ManifestLookup()
        keep = bool(manifest.get("keep_images", False))
        any_arch = bool(manifest.get("any_arch", False))
        catalog = manifest.get("catalog", [])
        for i, ref in enumerate(catalog, 1):
            console.header(ref, i, len(catalog))
            entry = LookupEntry(reproduce=Reproduce(reference=ref))
            try:
                if not any_arch:
                    ok, arches = host_supports(ref)
                    if not ok:
                        raise SkipContainer(
                            f"image provides {', '.join(arches)} but host is "
                            f"linux/{host_arch()} — profile it on a matching host "
                            f"(or pass --any-arch to override)"
                        )
                sess = self.session_factory(ref, keep)
                if hasattr(sess, "on_progress"):
                    sess.on_progress = console.phase
                if hasattr(sess, "install_python"):
                    sess.install_python = bool(manifest.get("install_python", True))
                if hasattr(sess, "allow_network"):
                    sess.allow_network = bool(manifest.get("allow_network", False))
                    sess.network = manifest.get("network", "bridge")
                with sess as backend:
                    entry.reproduce.digest = getattr(sess, "digest", "") or ""
                    entry.reproduce.pulled_by_us = getattr(sess, "pulled_by_us", False)
                    sink = []
                    tools = inspection_tools(backend, sink)
                    await runner.run_agent(
                        self.execute_system_prompt(manifest),
                        f"Characterize the HPC application(s) in image {ref}. "
                        f"Record each distinct build variant.",
                        tools,
                        confirm_fn,
                    )
                    entry.artifacts = sink
                console.ok(f"recorded {len(entry.artifacts)} artifact(s)")
            except SkipContainer as e:
                entry.skipped = str(e)
                console.warn(f"skipped: {e}")
            lookup.add(entry)
        return lookup

    def validate_result(self, result: Any) -> None:
        assert isinstance(
            result, ManifestLookup
        ), "profile must produce a ManifestLookup"
