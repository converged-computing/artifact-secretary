"""Tests for the source shape path.

These exercise the deterministic surfaces, the reader and the schema validation,
with no agent and no network so they run in CI. The classification itself is agent
judgment and is not unit tested. What is tested is that its output gets held to the
schema and that the traversal it drives stays bounded and cached.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretary.framework.source_tools import source_tools
from secretary.inspection.inspector import Inspector
from secretary.inspection.target import Target
from secretary.model.shape import Assertion, ScheduleShape, ShapeReport, ShapeSchema
from secretary.source.reader import SourceReader, TokenBudget, approx_tokens


def _tree(tmp_path):
    files = {
        "src/solver.c": "\n".join(
            [
                "#include <mpi.h>",
                "void exchange(void){",
                "  MPI_Sendrecv(halo);",
                "}",
                "int main(){",
                "  MPI_Init(0,0);",
                "  exchange();",
                "  MPI_Alltoall(buf);",
                "}",
            ]
        ),
        "run.sh": "srun --ntasks 128 --cpus-per-task 4 ./solver\n",
    }
    for rel, text in files.items():
        p = os.path.join(str(tmp_path), rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write(text)
    return Inspector(Target(str(tmp_path)))


def _reader(tmp_path, limit=60_000):
    return SourceReader(_tree(tmp_path), TokenBudget(limit=limit))


def test_read_window_and_cursor(tmp_path):
    r = _reader(tmp_path)
    out = r.read("/src/solver.c", start=1, count=2)
    assert "MPI_Sendrecv" not in out["content"]  # only the first two lines
    assert out["next"] == 3  # cursor to continue
    more = r.read("/src/solver.c", start=out["next"], count=3)
    assert "MPI_Sendrecv" in more["content"]


def test_reread_is_free(tmp_path):
    r = _reader(tmp_path)
    r.read("/src/solver.c", 1, 9)
    spent = r.budget.spent
    r.read("/src/solver.c", 1, 9)  # same lines again
    assert r.budget.spent == spent  # charged once, re-read free


def test_budget_truncates_new_reading(tmp_path):
    # a budget that fits only the first line or two of new reading
    r = _reader(tmp_path, limit=approx_tokens("#include <mpi.h>") + 1)
    out = r.read("/src/solver.c", 1, 9)
    assert out["truncated"] is True
    assert out["end"] < 9  # the window was shortened to fit the budget
    assert r.budget.remaining < approx_tokens("MPI_Alltoall(buf);")  # nearly spent


def test_search_returns_context_and_pattern_is_the_agents(tmp_path):
    r = _reader(tmp_path)
    out = r.search(r"MPI_Sendrecv", context=1)  # a signature the agent saw called
    assert out["hits"], "should find the call site"
    hit = out["hits"][0]
    assert "exchange" in hit["context"] or "MPI_Sendrecv" in hit["context"]
    assert hit["read_more"] is not None  # can expand around it


def test_seen_reports_coverage_and_budget(tmp_path):
    r = _reader(tmp_path)
    r.read("/src/solver.c", 1, 4)
    seen = r.seen()
    # paths are repo-relative so they stay meaningful outside the sandbox
    assert "src/solver.c" in seen["files"], seen["files"]
    assert seen["files"]["src/solver.c"]["lines_read"] == 4
    assert seen["budget"]["spent"] > 0


def test_schema_loads_and_describes():
    s = ShapeSchema.load()
    assert s.version.startswith("schedule-shape/")
    text = s.describe()
    assert "parallelism" in text and "topology.co_location" in text


def test_schema_validation():
    s = ShapeSchema.load()
    assert s.validate("parallelism", ["mpi", "gpu"])[0] is True
    assert s.validate("parallelism", ["telepathy"])[0] is False  # bad terminal
    # pattern is a set now. sparse solvers do a neighbor exchange and a reduction
    # and a single value quietly dropped the reduction, which is the costly part
    assert s.validate("communication.pattern", ["neighbor", "reduction"])[0] is True
    assert s.validate("communication.pattern", ["carrier_pigeon"])[0] is False
    assert s.validate("communication.pattern", "all_to_all")[0] is False  # not a list
    # the memory-boundedness family AMG needed and v1 could not express
    assert s.validate("memory.bound_by", "memory_bandwidth")[0] is True
    assert s.validate("memory.arithmetic_intensity", "very_low")[0] is True
    assert s.validate("memory.cache_reuse", "poor")[0] is True
    assert s.validate("launch.tasks", 128)[0] is True
    assert s.validate("launch.tasks", "128")[0] is False  # must be int
    assert s.validate("made.up.field", "x")[0] is False  # unknown field -> candidate


def test_shape_projects_to_nested_hint():
    shape = ScheduleShape(
        assertions=[
            Assertion("parallelism", ["mpi", "threaded"], ["src/solver.c:6"]),
            Assertion("communication.pattern", "all_to_all", ["src/solver.c:8"]),
            Assertion("topology.co_location", "preferred", ["src/solver.c:8"]),
        ]
    )
    hint = shape.to_hint()
    assert hint["parallelism"] == ["mpi", "threaded"]
    assert hint["communication"]["pattern"] == "all_to_all"
    assert hint["topology"]["co_location"] == "preferred"


# --- trace separation, different variants never share a shape list --------------


def test_traces_are_separate_per_subject_and_variant():
    """The whole point. A reaxff shape is not the shape of another pair style, so
    each subject and variant gets its own list."""
    report = ShapeReport(repo="https://github.com/lammps/lammps")
    reax = report.trace("lmp", "reaxff")
    kokkos = report.trace("lmp", "kokkos-cuda")
    other = report.trace("other-tool", "general")

    assert reax is not kokkos, "variants must not collapse into one trace"
    assert len({t.key() for t in report.traces}) == 3
    assert report.trace("lmp", "reaxff") is reax  # get-or-create, not duplicate
    assert len(report.traces) == 3

    reax.shapes.append(
        ScheduleShape(
            assertions=[Assertion("parallelism", ["mpi"], ["src/reaxc.cpp:10"])]
        )
    )
    assert len(kokkos.shapes) == 0, "assertions must not leak between traces"
    assert len(other.shapes) == 0


# --- the record_shape tool, routing and the two refusals ------------------------


def _record_tool(sink):
    """The record_shape handler over a stub reader (it never reads during record)."""

    class _StubReader:
        backend = object()
        root = "/"

        def extract(self, path, start, end=0, **kw):
            return []  # no source behind these fixtures

    return {t.name: t for t in source_tools(_StubReader(), ShapeSchema.load(), sink)}[
        "record_shape"
    ]


def test_record_shape_routes_into_named_trace():
    sink = [ShapeReport(repo="lammps")]
    record = _record_tool(sink)
    asyncio.run(
        record.handler(
            {
                "subject": "lmp",
                "variant": "reaxff",
                "assertions": [
                    {
                        "field": "parallelism",
                        "value": ["mpi"],
                        "evidence": ["src/reaxc.cpp:10"],
                    }
                ],
            }
        )
    )
    asyncio.run(
        record.handler(
            {
                "subject": "lmp",
                "variant": "kokkos-cuda",
                "assertions": [
                    {
                        "field": "accelerator.kind",
                        "value": "cuda",
                        "evidence": ["src/KOKKOS/pair.cpp:88"],
                    }
                ],
            }
        )
    )
    report = sink[0]
    # variant names are normalised to a canonical token slug so the same build
    # from two models/runs lands on one trace instead of two
    keys = sorted(t.key() for t in report.traces)
    assert keys == ["lmp:cuda-kokkos", "lmp:reaxff"], keys
    for t in report.traces:
        assert len(t.shapes) == 1, "each variant keeps only its own shape"


def test_record_shape_requires_subject_and_evidence():
    sink = [ShapeReport(repo="lammps")]
    record = _record_tool(sink)

    out = asyncio.run(record.handler({"assertions": []}))  # no subject
    assert "refused" in out["content"][0]["text"]
    assert sink[0].traces == []

    out = asyncio.run(
        record.handler(
            {
                "subject": "lmp",
                "assertions": [
                    {"field": "parallelism", "value": ["mpi"]}  # no evidence
                ],
            }
        )
    )
    assert "refused" in out["content"][0]["text"]
    assert sink[0].trace("lmp").shapes == [], "an unevidenced assertion is not recorded"


def test_record_shape_keeps_unknown_field_as_candidate():
    sink = [ShapeReport(repo="lammps")]
    record = _record_tool(sink)
    asyncio.run(
        record.handler(
            {
                "subject": "lmp",
                "assertions": [
                    {
                        "field": "communication.wavefront_depth",  # not in the schema
                        "value": 3,
                        "evidence": ["src/comm.cpp:42"],
                    }
                ],
            }
        )
    )
    trace = sink[0].trace("lmp")
    assert trace.shapes == [], "unknown fields do not become shapes"
    assert len(trace.unmatched) == 1
    assert "unknown field" in trace.unmatched[0].error


# --- the clone gate and the real confirm contract -------------------------------
#
# ConfirmFn = Callable[[str, dict], bool] -- two positional args, SYNCHRONOUS.
# Calling it with one arg (or awaiting it) blows up only at runtime on a clone,
# which no other test reaches, so pin the contract here.


class _FakeSession:
    """Stands in for SourceSession, says it would clone and yields a backend."""

    def __init__(self, will_clone=True):
        self.will_clone = will_clone
        self.commit = "deadbeef"
        self.root = "/"
        self.entered = False

    def __enter__(self):
        self.entered = True

        class _B:
            def read_text(self, path, **kw):
                return ""

            def scan_tree(self, root, patterns, **kw):
                return {}

            def find(self, *a, **kw):
                return []

        return _B()

    def __exit__(self, *exc):
        return None


def _run_shape_task(confirm_fn, session):
    """Drive ShapeTask.execute with a stub runner, returning (lookup, runner)."""
    from secretary.tasks.shape import ShapeTask

    class _Runner:
        def __init__(self):
            self.ran = False

        async def run_agent(self, system, task, tools, confirm):
            self.ran = True

    task = ShapeTask(max_tokens=1000)
    task._make_session = lambda repo, manifest: session
    runner = _Runner()
    lookup = asyncio.run(
        task.execute(
            runner,
            {"repos": ["https://github.com/lammps/lammps"], "resolved_from": "lammps"},
            confirm_fn,
        )
    )
    return lookup, runner


def test_clone_gate_uses_sync_two_arg_confirm():
    seen = {}

    def confirm(tool_name, args):  # the real ConfirmFn shape
        seen["tool_name"] = tool_name
        seen["args"] = args
        return True

    session = _FakeSession()
    lookup, runner = _run_shape_task(confirm, session)

    assert seen["tool_name"] == "clone", seen
    assert seen["args"]["repo"].endswith("/lammps")
    # the resolved-from is surfaced so a bad guess is catchable at the gate
    assert seen["args"]["resolved_from"] == "lammps"
    assert session.entered, "approval should proceed into the session"
    assert runner.ran, "the agent should have been run"
    assert not list(lookup.entries.values())[0].skipped


def test_clone_declined_skips_without_entering_session():
    session = _FakeSession()
    lookup, runner = _run_shape_task(lambda tool_name, args: False, session)

    assert not session.entered, "a declined clone must not open the session"
    assert not runner.ran, "a declined clone must not run the agent"
    assert list(lookup.entries.values())[0].skipped == "clone declined"


# --- error shows up only when there is one --------------------------------------


def test_clean_report_has_no_empty_error_keys():
    """Recorded assertions validated fine so nothing should carry an error key."""
    import json

    from secretary.model.shape import ShapeLookup

    lookup = ShapeLookup()
    report = ShapeReport(repo="lammps")
    trace = report.trace("lmp", "reaxff")
    trace.shapes.append(
        ScheduleShape(
            assertions=[
                Assertion("parallelism", ["mpi"], ["src/pair_reaxff.cpp:209"]),
                # a False value has to survive, it means something
                Assertion(
                    "communication.bandwidth_sensitive",
                    False,
                    ["src/fix_qeq_reaxff.cpp:1036"],
                ),
            ]
        )
    )
    lookup.add(report)
    text = lookup.to_json()

    assert '"error"' not in text, "a clean report must not carry empty error keys"
    parsed = json.loads(text)
    asserts = parsed["entries"]["lammps"]["traces"][0]["shapes"][0]["assertions"]
    assert len(asserts) == 2
    vals = {a["field"]: a["value"] for a in asserts}
    assert (
        vals["communication.bandwidth_sensitive"] is False
    ), "False must not be pruned"


def test_real_error_is_kept_in_output():
    """An unmatched assertion keeps its error since there is an actual one."""
    import json

    from secretary.model.shape import ShapeLookup

    sink = [ShapeReport(repo="lammps")]
    record = _record_tool(sink)
    asyncio.run(
        record.handler(
            {
                "subject": "lmp",
                "assertions": [
                    {
                        "field": "communication.pattern",
                        "value": ["carrier_pigeon"],  # not a schema terminal
                        "evidence": ["src/comm.cpp:42"],
                    }
                ],
            }
        )
    )
    lookup = ShapeLookup()
    lookup.add(sink[0])
    parsed = json.loads(lookup.to_json())
    unmatched = parsed["entries"]["lammps"]["traces"][0]["unmatched"]
    assert len(unmatched) == 1
    assert "error" in unmatched[0], unmatched[0]
    assert "carrier_pigeon" in unmatched[0]["error"]


# --- output layout, one shapes.json per app run ---------------------------------


def test_save_tree_writes_one_report_per_repo_run(tmp_path):
    """Each run lands in its own directory keyed by repo, revision and focus so
    runs never overwrite each other."""
    import json

    from secretary.model.shape import ShapeLookup

    lookup = ShapeLookup()
    for repo, commit in (
        ("https://github.com/lammps/lammps", "abc123def4567890"),
        ("https://github.com/LLNL/Kripke", ""),  # unpinned
    ):
        report = ShapeReport(repo=repo, commit=commit)
        trace = report.trace("lmp", "reaxff")
        trace.shapes.append(
            ScheduleShape(
                assertions=[Assertion("parallelism", ["mpi"], ["src/x.cpp:1"])]
            )
        )
        lookup.add(report)

    paths = lookup.save_tree(str(tmp_path), label="reaxff")
    rel = sorted(p[len(str(tmp_path)) + 1 :] for p in paths)
    assert rel == [
        "github.com/LLNL/Kripke/unpinned/reaxff/shapes.json",
        "github.com/lammps/lammps/abc123def456/reaxff/shapes.json",
    ], rel
    doc = json.load(open(paths[0]))
    assert doc["version"].startswith("schedule-shape/")
    assert "entry" in doc


def test_save_tree_skips_skipped_entries(tmp_path):
    from secretary.model.shape import ShapeLookup

    lookup = ShapeLookup()
    report = ShapeReport(repo="https://github.com/nope/nope")
    report.skipped = "clone failed"
    lookup.add(report)
    assert lookup.save_tree(str(tmp_path)) == []


def test_parse_repo_handles_url_forms_and_paths():
    from secretary.model.shape import parse_repo

    assert parse_repo("https://github.com/lammps/lammps") == [
        "github.com",
        "lammps",
        "lammps",
    ]
    assert parse_repo("git@github.com:LLNL/Kripke.git") == [
        "github.com",
        "LLNL",
        "Kripke",
    ]
    assert parse_repo("/opt/lammps")[0] == "local"


# --- run provenance gets recorded during extraction, not analysis ---------------


def test_run_info_is_recorded_in_the_report():
    """The artifact has to say which model and budget produced it, or analysis done
    later cannot attribute or compare anything."""
    import json

    from secretary.model.shape import RunInfo, ShapeLookup

    report = ShapeReport(
        repo="https://github.com/lammps/lammps",
        commit="abc123def4567",
        run=RunInfo(
            model="us.anthropic.claude-sonnet-5",
            backend="aws",
            mode="container",
            focus="reaxff",
            max_source_tokens=60000,
            model_max_tokens=16384,
            duration_s=12.34,
        ),
    )
    report.trace("lmp", "reaxff").shapes.append(
        ScheduleShape(assertions=[Assertion("parallelism", ["mpi"], ["src/x:1"])])
    )
    lookup = ShapeLookup()
    lookup.add(report)

    doc = json.loads(lookup.to_json())
    run = doc["entries"]["https://github.com/lammps/lammps@abc123def4567"]["run"]
    assert run["model"] == "us.anthropic.claude-sonnet-5"
    assert run["max_source_tokens"] == 60000
    assert run["model_max_tokens"] == 16384
    assert run["focus"] == "reaxff"
    assert run["mode"] == "container"


def test_run_info_survives_save_tree(tmp_path):
    import json

    from secretary.model.shape import RunInfo, ShapeLookup

    report = ShapeReport(repo="https://github.com/LLNL/Kripke", commit="fed987")
    report.run = RunInfo(model="opus", backend="aws")
    report.trace("kripke").shapes.append(
        ScheduleShape(assertions=[Assertion("parallelism", ["mpi"], ["src/x:1"])])
    )
    lookup = ShapeLookup()
    lookup.add(report)
    paths = lookup.save_tree(str(tmp_path))
    doc = json.load(open(paths[0]))
    assert doc["entry"]["run"]["model"] == "opus"


def test_cli_has_no_analysis_subcommands():
    """Summarising/comparing reports is analysis and lives in a separate script,
    so the tool must expose extraction only."""
    import contextlib
    import io
    import sys as _sys

    from secretary.cli.main import main as cli_main

    cli = _sys.modules["secretary.cli.main"]
    for name in ("_cmd_summarize", "_cmd_compare", "_print_rows"):
        assert not hasattr(cli, name), f"{name} should not exist in the CLI"

    err = io.StringIO()
    argv = _sys.argv
    try:
        with contextlib.suppress(SystemExit), contextlib.redirect_stderr(err):
            _sys.argv = ["artifact-secretary", "summarize", "somewhere"]
            cli_main()
    finally:
        _sys.argv = argv
    assert "invalid choice" in err.getvalue(), err.getvalue()


# --- fixes that came out of analysing the first sweep ---------------------------


def test_reader_paths_are_repo_relative(tmp_path):
    """The container clone sits at an absolute path, and if that leaks into
    evidence nobody can resolve the citation once the sandbox is gone. Everything
    the agent sees is relative to the repository root."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.cc").write_text("int main(){}\n" * 20)
    insp = Inspector(Target(str(tmp_path)))

    # act like the container case, backend at slash and the clone below it
    reader = SourceReader(insp, TokenBudget(limit=10_000), root="/src")
    out = reader.read("main.cc", 1, 3)
    assert out["path"] == "main.cc", out["path"]
    assert "/src" not in out["path"]
    hits = reader.search(r"int main")
    assert hits["hits"], hits
    assert not hits["hits"][0]["path"].startswith("/"), hits["hits"][0]["path"]
    assert "src/" not in hits["hits"][0]["path"]
    assert all(not k.startswith("/") for k in reader.seen()["files"])


def test_negative_claim_needs_absence_evidence():
    """A line cannot prove absence, which is what made checkpoint the least
    verifiable field in the first sweep."""
    sink = [ShapeReport(repo="x")]
    record = _record_tool(sink)

    out = asyncio.run(
        record.handler(
            {
                "subject": "app",
                "assertions": [
                    {
                        "field": "io.checkpoint",
                        "value": False,
                        "evidence": ["src/main.c:12"],
                    }
                ],
            }
        )
    )
    assert "refused" in out["content"][0]["text"]
    assert sink[0].trace("app").shapes == []

    out = asyncio.run(
        record.handler(
            {
                "subject": "app",
                "assertions": [
                    {
                        "field": "io.checkpoint",
                        "value": False,
                        "evidence": ["absent: checkpoint|restart|SCR_ in src/"],
                    }
                ],
            }
        )
    )
    assert "recorded 1" in out["content"][0]["text"]
    assert len(sink[0].trace("app").shapes) == 1


def test_variant_names_are_normalised():
    """Two models spelling the same build differently land on one trace. The point
    is that the two agree, not any particular spelling, since the vocabulary comes
    from the schema and can change there."""
    from secretary.framework.source_tools import _slug_variant

    sch = ShapeSchema.load()

    def slug(v):
        return _slug_variant(v, sch)

    assert slug("mpi-openmp-cpu") == slug("cpu-mpi-openmp")
    assert slug("mpi-openmp-cpu") == slug("cpu-mpi-threads")
    assert slug("reaxff-kokkos-cuda") == slug("cuda-kokkos-reaxff")
    assert slug("mpi + OpenMP threading (default build)") == slug("openmp-mpi")
    assert slug("c/mpi/pt2pt/standard (CPU host buffers)") == slug("cpu-mpi")
    assert "(" not in slug("mpi + OpenMP threading (default build)")
    assert slug("") == "general"


def test_variant_vocabulary_is_not_hard_coded():
    """No token list in code. Tokens come from the schema terminals, and a name the
    schema has never heard of, a pair style for instance, survives as a free token
    rather than being dropped by the tool."""
    import inspect

    from secretary.framework import source_tools

    sch = ShapeSchema.load()
    # look at the code only, prose is allowed to name examples
    import re as _re

    body = inspect.getsource(source_tools._slug_variant)
    body = body.replace(source_tools._slug_variant.__doc__ or "", "")
    body = _re.sub(r"#.*", "", body)
    for token in ("reaxff", "boomeramg", "kokkos", "cuda", "openmp", "sycl"):
        assert token not in body, f"{token} should come from the schema, not code"

    assert "cuda" in sch.variant_tokens()
    assert "reaxff" not in sch.variant_tokens()
    # and it still survives normalisation
    assert "reaxff" in source_tools._slug_variant("reaxff-cuda", sch)


def test_memory_boundedness_is_expressible():
    """There used to be nowhere to record that a code is memory access bound, so
    both models filed the amg docs under network bandwidth instead."""
    from secretary.model.shape import ShapeSchema

    s = ShapeSchema.load()
    assert s.validate("memory.bound_by", "memory_bandwidth")[0] is True
    assert s.validate("memory.arithmetic_intensity", "very_low")[0] is True
    assert s.validate("memory.cache_reuse", "poor")[0] is True
    # and the network field must say it is about the network
    assert "NETWORK" in s.fields["communication.bandwidth_sensitive"]["description"]


def test_snippets_are_captured_at_extraction_time(tmp_path):
    """The cited code gets stored while the clone is still there. Afterwards the
    sandbox is gone and re-fetching an unpinned repo gives whatever the default
    branch says today, which is not always what the model read."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "comm.c").write_text(
        "\n".join(f"line{i} MPI_Irecv" if i == 12 else f"line{i}" for i in range(1, 40))
        + "\n"
    )
    insp = Inspector(Target(str(tmp_path)))
    budget = TokenBudget(limit=50_000)
    reader = SourceReader(insp, budget, root="/")
    sink = [ShapeReport(repo="r")]
    tools = {t.name: t for t in source_tools(reader, ShapeSchema.load(), sink)}

    before = budget.spent
    asyncio.run(
        tools["record_shape"].handler(
            {
                "subject": "app",
                "assertions": [
                    {
                        "field": "communication.pattern",
                        "value": ["neighbor"],
                        "evidence": ["src/comm.c:12 (halo exchange)"],
                        "confidence": "high",
                    }
                ],
            }
        )
    )
    snips = sink[0].snippets
    assert "src/comm.c:12" in snips, snips
    text = "\n".join(t for _n, t in snips["src/comm.c:12"]["lines"])
    assert "MPI_Irecv" in text, text
    # context around the cited line, not just the line itself
    assert len(snips["src/comm.c:12"]["lines"]) >= 5
    # and it must not touch the source budget, this goes into the artifact and
    # never into the model context
    assert budget.spent == before, "snippet capture must not consume the budget"


def test_snippet_capture_is_capped_and_deduped(tmp_path):
    """A stray huge citation must not bloat the report, and about half of all real
    citations repeat a span, so each one gets stored once."""
    from secretary.framework.source_tools import MAX_SNIPPET_LINES_PER_REPORT

    (tmp_path / "big.c").write_text("\n".join(f"l{i}" for i in range(1, 2000)) + "\n")
    insp = Inspector(Target(str(tmp_path)))
    reader = SourceReader(insp, TokenBudget(limit=50_000), root="/")
    sink = [ShapeReport(repo="r")]
    tools = {t.name: t for t in source_tools(reader, ShapeSchema.load(), sink)}

    asyncio.run(
        tools["record_shape"].handler(
            {
                "subject": "app",
                "assertions": [
                    {
                        "field": "parallelism",
                        "value": ["mpi"],
                        "evidence": ["big.c:10-1500"],
                        "confidence": "high",
                    },
                    # same span twice -> stored once
                    {
                        "field": "communication.model",
                        "value": "collective",
                        "evidence": ["big.c:10-1500"],
                        "confidence": "high",
                    },
                ],
            }
        )
    )
    snips = sink[0].snippets
    assert len(snips) == 1, "the repeated span should be stored once"
    only = next(iter(snips.values()))
    assert len(only["lines"]) <= 24, len(only["lines"])
    total = sum(len(v["lines"]) for v in snips.values())
    assert total <= MAX_SNIPPET_LINES_PER_REPORT
