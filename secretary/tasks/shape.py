"""Work out how the application in a repository wants to run, so a scheduler can
place it. The source side sibling of ProfileTask.

The task hands the agent a cached and budgeted view of the clone plus the grammar
as context. The agent works out what the repo builds, interrogates the compute
source, and records one trace per subject and variant. Those stay separate,
because lammps with reaxff is not the same application as lammps with another
pair style.

Nothing about the classification is hard coded. There is no entrypoint seed and no
marker table. The agent reasons and record_shape only holds it to the schema
terminals, keeping anything else as a candidate to grow the schema.

The repository can arrive as a url, a local path, or a description the setup agent
resolves from what it knows. Cloning is the one thing that acts and the confirm
gate covers it. A tree already present is read only.
"""

from __future__ import annotations

import datetime
import time
from typing import Any

from behalf import AgentRunner, ConfirmFn, Task

from ..framework.source_tools import source_tools
from ..model.shape import RunInfo, ShapeLookup, ShapeReport, ShapeSchema
from ..source.reader import SourceReader, TokenBudget
from ..source.session import (
    DEFAULT_BASE_IMAGE,
    ContainerSourceSession,
    SkipRepo,
    SourceSession,
)

CHARACTERIZE = """You characterize how the application in a source repository
WANTS TO RUN, so a scheduler can decide how to place it. This is SOURCE analysis:
you read code, build files and docs. There is no binary and no container config to
inspect -- the sandbox you are running in is just a clean room with git in it.

First, identify the SUBJECT: what does this repo actually build and run? Read the
build system (CMakeLists.txt, Makefile, setup.py) and the docs to find the
executable(s) and how they are invoked -- for LAMMPS that is `lmp`, for many codes
it is a single target under src/ or build/. Do not assume a run script exists.

Then interrogate the compute source for scheduling behavior. Ask questions and
`search` for the evidence, reading around each hit:
  * parallelism -- MPI? threads/OpenMP? GPU kernels? task frameworks?
  * communication -- which collectives or point-to-point calls, and in what
    pattern (halo/neighbor exchange, all-to-all, reductions)?
  * memory -- shared-memory windows or /dev/shm? affinity pinning? huge pages?
  * accelerator -- CUDA/HIP/SYCL, and any direct GPU-to-GPU path (NCCL/RCCL)?
  * I/O -- parallel I/O libraries, checkpoint/restart?
  * launch geometry -- rank/thread/GPU counts, but ONLY where a real invocation
    states them. Never invent a count.
Follow the code that matters; do not try to read the whole tree. Check `seen` to
avoid re-reading and to watch your token budget.

Record with record_shape, and mind the SEPARATION: different subjects, and
different build/config variants of one subject, are different applications from a
scheduler's view -- LAMMPS with reaxff does not have the shape of LAMMPS with some
other pair style. Give each its own (subject, variant) trace; never pool them into
one list. If the user named a specific variant, trace that one. If not, either
enumerate the variants you can separate with real evidence, or -- when you cannot
separate them -- record ONE deliberately general trace (variant "general") with a
low confidence and a note saying it describes how the code generally works.

EVIDENCE. Cite what you actually read, using repo-relative paths (the tree you
are reading is the repository root -- never write an absolute container path like
/tmp/... into evidence, it means nothing once the sandbox is gone).
  * For a claim that something IS there: "src/comm.c:212" or "src/comm.c:212-240",
    optionally with a short parenthetical reason.
  * For a claim that something is ABSENT (value none/False/unknown): a line
    reference proves nothing. Record the search you ran instead, in the form
    "absent: <pattern> in <scope>" -- e.g. "absent: MPI_File|hdf5|adios in src/".
    A negative claim evidenced only by a line number is refused.
  * Do not put reasoning in place of evidence. "accelerator.kind=none implies no
    multi_gpu" is an inference, not evidence: either search for it or omit the field.

VARIANTS. Name a variant with lowercase tokens joined by a dash, taken from the
token list below, plus a package name where it tells builds apart. Prefer the
specific backend over the generic one. No prose and no brackets.

{variant_tokens}

WHAT LIMITS THIS CODE. memory.bound_by is often the single most useful field for a
scheduler, and documentation frequently states it outright -- a README saying
"this code is memory-access bound, doing 1-2 computations per memory access" is
direct evidence for bound_by=memory_bandwidth AND arithmetic_intensity=very_low.
Look for such statements before inferring. Keep the distinction sharp:
communication.bandwidth_sensitive is about the NETWORK; a code with no MPI at all
is False there no matter how memory-bandwidth-hungry it is.

PATTERNS ARE A SET. Most solvers do a neighbour/halo exchange AND a small global
reduction per iteration, so record both. If ranks form
an ordered chain (a sweep: upwind -> downwind), that is "wavefront" and you should
also set communication.ordered_dependency=true, because the scheduling consequence
(stragglers stall the front) differs from symmetric exchange.

Keep each record_shape call SMALL: at most ~8 assertions, a few evidence entries
each, and no prose beyond a short note. Call it repeatedly for the same
(subject, variant) instead of assembling one huge call -- assertions accumulate in
the trace, and an over-long single response can exceed the model's output limit and
lose the whole run.

The grammar you must target (fields marked (judgment) are ones the code cannot
settle mechanically -- that is your call, with evidence):

{schema}
"""

SETUP = """You are setting up a repository shape-profiling run. The output is a
schedule-shape grammar per repository -- do NOT ask about its structure.

The user may give you a repository URL, a local path, or just a DESCRIPTION of an
application ("trace lammps", "the reaxff build of lammps"). For a description,
resolve it to the canonical repository URL yourself from what you know -- do not
ask the user to look it up. Put the resolved URL in `repos` and what they said in
`resolved_from`, so the clone step (which the user confirms) shows the URL you
picked. If a clone later fails, the user can supply the right URL.

Also capture any refinement they gave about WHICH build or configuration to trace
(e.g. "reaxff", "the CUDA build") in `focus` -- absent that, the agent will
enumerate variants or fall back to a general characterization.

Cloning and analysis ALWAYS happen inside a throwaway sandbox container -- do not
ask the user where to run, and do not offer to run on the host. Running untrusted
repository contents on the host is unsafe, so it is available only as an explicit
command-line opt-out, never something you elicit or choose.

Ask only what you cannot infer: the repository (or description), an optional
branch/tag, and any focus. When you have those, call finalize_setup with:
{"repos": [<url-or-path>], "resolved_from": <str|null>, "focus": <str|null>,
"base_image": <str|null>, "ref": <str|null>, "keep": <bool>}."""


class ShapeTask(Task):
    name = "shape"
    requires_setup_approval = True  # may clone (network + write)

    def __init__(
        self,
        max_tokens: int = 60_000,
        model: str = "",
        backend: str = "",
        model_max_tokens: int = 0,
    ):
        self.max_tokens = max_tokens
        self.schema = ShapeSchema.load()
        # recorded into every report so a result stays attributable afterwards
        self.model = model
        self.backend = backend
        self.model_max_tokens = model_max_tokens

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
        return CHARACTERIZE.replace("{schema}", self.schema.describe()).replace(
            "{variant_tokens}", "  " + ", ".join(self.schema.variant_tokens())
        )

    def _make_session(self, repo: str, manifest: dict):
        ref = manifest.get("ref") or None
        # sandbox unless someone typed host exactly. a typo gets a container
        if manifest.get("mode", "container") != "host":
            return ContainerSourceSession(
                repo,
                base_image=manifest.get("base_image") or DEFAULT_BASE_IMAGE,
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
            report = ShapeReport(
                repo=repo,
                schema_version=self.schema.version,
                resolved_from=manifest.get("resolved_from") or "",
                run=RunInfo(
                    model=self.model,
                    backend=self.backend,
                    mode=manifest.get("mode", "container"),
                    base_image=manifest.get("base_image") or "",
                    focus=manifest.get("focus") or "",
                    max_source_tokens=self.max_tokens,
                    model_max_tokens=self.model_max_tokens,
                    started=datetime.datetime.now(datetime.timezone.utc)
                    .replace(microsecond=0)
                    .isoformat(),
                    tool_version=_tool_version(),
                ),
            )
            t0 = time.monotonic()
            try:
                sess = self._make_session(repo, manifest)
                if hasattr(sess, "on_progress"):
                    sess.on_progress = console.phase
                if getattr(sess, "will_clone", False):
                    # the args dict is what the user sees, so put the resolved url
                    # in it. that is how a bad guess gets caught
                    args = {"repo": repo, "effect": "network + write"}
                    if manifest.get("resolved_from"):
                        args["resolved_from"] = manifest["resolved_from"]
                    if manifest.get("ref"):
                        args["ref"] = manifest["ref"]
                    if not confirm_fn("clone", args):
                        report.skipped = "clone declined"
                        console.warn("clone declined")
                        lookup.add(report)
                        continue
                with sess as backend:
                    report.commit = getattr(sess, "commit", "") or ""
                    # an unresolved revision means the report is unpinned, so say
                    # so rather than leaving an empty commit field
                    commit_error = getattr(sess, "commit_error", "") or ""
                    if commit_error:
                        report.notes = (
                            report.notes + " " if report.notes else ""
                        ) + f"unpinned: could not resolve commit ({commit_error})"
                        console.warn(f"unpinned: {commit_error}")
                    budget = TokenBudget(limit=self.max_tokens)
                    reader = SourceReader(
                        backend, budget, root=getattr(sess, "root", "/")
                    )
                    sink: list[ShapeReport] = [report]
                    tools = source_tools(
                        reader, self.schema, sink, repo=repo, commit=report.commit
                    )
                    focus = manifest.get("focus")
                    task = (
                        f"Characterize how the application in {repo} wants to run. "
                        f"Identify what it builds, then interrogate the source. "
                        f"Record a separate trace per subject/variant."
                    )
                    if focus:
                        task += f" The user asked specifically about: {focus}."
                    await runner.run_agent(
                        self.execute_system_prompt(manifest),
                        task,
                        tools,
                        confirm_fn,
                    )
                    report.reads = reader.seen()  # coverage + budget = provenance
                n = sum(len(sh.assertions) for t in report.traces for sh in t.shapes)
                console.ok(
                    f"recorded {n} assertion(s) across {len(report.traces)} trace(s)"
                )
            except SkipRepo as e:
                report.skipped = str(e)
                console.warn(f"skipped: {e}")
            report.run.duration_s = time.monotonic() - t0
            lookup.add(report)
        return lookup

    def validate_result(self, result: Any) -> None:
        assert isinstance(result, ShapeLookup), "shape must produce a ShapeLookup"


def _tool_version() -> str:
    try:
        from importlib.metadata import version

        return version("artifact-secretary")
    except Exception:
        return ""
