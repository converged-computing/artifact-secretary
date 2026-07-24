"""ShapeTask: characterize how the application in a REPOSITORY wants to run, so
a scheduler can derive its resource shape. The source-level sibling of
ProfileTask.

Unlike ProfileTask, this task can ACT: when asked to clone, it performs a
network + write. So requires_setup_approval is True, and the clone (a session
step, never an agent tool) is gated behind confirm_fn. When pointed at a tree
that already exists, nothing acts and the run is read-only like profile.
"""

from __future__ import annotations

from typing import Any, Optional

from behalf import AgentRunner, ConfirmFn, Task

from ..framework.source_tools import source_tools
from ..model.shape import ShapeLookup, ShapeReport
from ..source.session import ContainerSourceSession, SkipRepo, SourceSession

CHARACTERIZE = """You characterize how the application in a source repository
WANTS TO RUN so a scheduler can place it. Work read-only.

1. resolve_entrypoint first: find what the container/app runs on start
   (Dockerfile ENTRYPOINT/CMD, entrypoint.sh, run scripts) and the launcher it
   uses. That command is the spine of the run.
2. scan_shape_markers over the tree: this returns categorized, evidenced facts
   (parallelism, communication calls, shared memory, pinning, accelerator, I/O).
   These are ground truth -- do not second-guess them, but DO decide which
   matter for the real entrypoint.
3. record_shape once per distinct entrypoint. The hard categories are DERIVED
   from the markers you pass; you add only the soft judgments the markers can't
   settle -- communication INTENSITY (how much of the runtime is comm), and a
   refined PATTERN when the calls make it obvious (halo exchange -> neighbor) --
   each with a confidence. Never invent rank/thread counts: pass only counts you
   actually read from the launcher/env.

If the repo builds MORE THAN ONE runnable thing (a CPU and a GPU entrypoint),
record each as a separate shape."""

SETUP = """You are setting up a repository shape-profiling run. The output is a
schedule-shape grammar (parallelism, communication, memory, accelerator, launch,
topology) per repository -- do NOT ask about its structure. Ask only what you
cannot infer: which repositories (URLs to clone, or paths already present),
whether to run on the host or inside a base container (and which base image if
so), and an optional branch/tag. When you have those, call finalize_setup with:
{"repos": [<url-or-path>], "mode": "host"|"container", "base_image": <str|null>,
 "ref": <str|null>, "keep": <bool>}."""


class ShapeTask(Task):
    name = "shape"
    requires_setup_approval = True  # may clone (network + write)

    def setup_system_prompt(self) -> str:
        return SETUP

    def manifest_schema(self) -> dict:
        return {
            "repos": [str],
            "mode": str,
            "base_image": str,
            "ref": str,
            "keep": bool,
        }

    def execute_system_prompt(self, manifest: dict) -> str:
        return CHARACTERIZE

    def _make_session(self, repo: str, manifest: dict):
        ref = manifest.get("ref") or None
        if manifest.get("mode") == "container":
            return ContainerSourceSession(
                repo,
                base_image=manifest["base_image"],
                ref=ref,
                keep_images=bool(manifest.get("keep", False)),
            )
        return SourceSession(
            repo, ref=ref, keep_clone=bool(manifest.get("keep", False))
        )

    async def execute(
        self, runner: AgentRunner, manifest: dict, confirm_fn: ConfirmFn
    ) -> ShapeLookup:
        from behalf import console

        lookup = ShapeLookup()
        repos = manifest.get("repos", [])
        for i, repo in enumerate(repos, 1):
            console.header(repo, i, len(repos))
            report = ShapeReport(repo=repo)
            try:
                sess = self._make_session(repo, manifest)
                if hasattr(sess, "on_progress"):
                    sess.on_progress = console.phase
                # the one act: cloning. gate it, and only it.
                if getattr(sess, "will_clone", False):
                    ok = await confirm_fn(f"clone {repo} (network + write)?")
                    if not ok:
                        report.skipped = "clone declined"
                        console.warn("clone declined")
                        lookup.add(report)
                        continue
                with sess as backend:
                    report.commit = getattr(sess, "commit", "") or ""
                    sink: list[ShapeReport] = [report]
                    tools = source_tools(backend, sink, repo=repo, commit=report.commit)
                    await runner.run_agent(
                        self.execute_system_prompt(manifest),
                        f"Characterize how the application in {repo} wants to run. "
                        f"Start at the entrypoint. Record each distinct shape.",
                        tools,
                        confirm_fn,
                    )
                console.ok(f"recorded {len(report.shapes)} shape(s)")
            except SkipRepo as e:
                report.skipped = str(e)
                console.warn(f"skipped: {e}")
            lookup.add(report)
        return lookup

    def validate_result(self, result: Any) -> None:
        assert isinstance(result, ShapeLookup), "shape must produce a ShapeLookup"
