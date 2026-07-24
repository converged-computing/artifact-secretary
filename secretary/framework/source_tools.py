"""The source-exploration toolset as framework ToolSpecs, built over the same
Inspector method surface as inspection_tools (local Inspector or the container
RemoteInspector). All read-only. record_shape writes into a sink and derives the
schedule shape deterministically from the observed markers.

Where inspection_tools reads a compiled binary's linkage to characterize
hardware CAPABILITY, these tools read a repository's source and entrypoint to
characterize its runtime SHAPE. The split of labor is identical: the scan
primitives supply deterministic facts (which markers appear, and where), and the
agent supplies where to look and the soft judgments (communication intensity)
that no regex settles.
"""

from __future__ import annotations

import json
import re
import shlex
from typing import Any

from behalf import ToolSpec

from ..model.shape import (
    ALL_MARKER_PATTERNS,
    LAUNCH_FLAGS,
    ScheduleShape,
    ShapeReport,
    categorize_hits,
    derive_shape,
)


def _text(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}]}


# entrypoint files worth resolving, in rough priority order
_ENTRY_FILES = ("Dockerfile", "entrypoint.sh", "run.sh", "run", "launch.sh")
_ENV_KEYS = re.compile(
    r"\b(OMP_NUM_THREADS|OMP_\w+|FI_\w+|UCX_\w+|OMPI_\w+|SLURM_\w+|CUDA_VISIBLE_DEVICES|MV2_\w+)\b"
)


def _parse_launch_flags(command: str) -> dict:
    """Lift parsed-or-null counts out of a launcher invocation. Never guesses:
    a field is present only if the flag is literally there."""
    found: dict[str, int] = {}
    for field, rx in LAUNCH_FLAGS.items():
        m = re.search(rx, command)
        if m:
            found[field] = int(m.group(1))
    return found


def source_tools(
    backend, sink: list[ShapeReport], repo: str = "", commit: str = ""
) -> list[ToolSpec]:
    async def list_dir(a):
        return _text(backend.list_dir(a.get("path", "/")))

    async def find(a):
        return _text(
            backend.find(
                a.get("root", "/"),
                a.get("name_glob") or None,
                a.get("kind") or None,
                a.get("limit", 200),
            )
        )

    async def read_text(a):
        return {"content": [{"type": "text", "text": backend.read_text(a["path"])}]}

    async def scan_source(a):
        # generic regex sweep for chasing specifics the library doesn't cover
        return _text(backend.scan_tree(a.get("root", "/"), a["patterns"]))

    async def scan_shape_markers(a):
        # the deterministic core: run the whole marker library and hand back
        # categorized, evidenced hits ready for record_shape.
        raw = backend.scan_tree(a.get("root", "/"), ALL_MARKER_PATTERNS)
        return _text(categorize_hits(raw))

    async def resolve_entrypoint(a):
        """Locate and read the launch surface (Dockerfile ENTRYPOINT/CMD/ENV and
        any referenced run scripts), returning the raw text for the agent to
        interpret plus any launch flags/env we could parse deterministically."""
        root = a.get("root", "/")
        result: dict = {"files": {}, "launch": {}, "env": {}}
        for name in _ENTRY_FILES:
            for path in backend.find(root, name_glob=name, kind="file", limit=5):
                txt = backend.read_text(path)
                if isinstance(txt, str) and not txt.startswith("<error"):
                    result["files"][path] = txt[:8000]
                    result["launch"].update(_parse_launch_flags(txt))
                    for m in _ENV_KEYS.finditer(txt):
                        result["env"].setdefault(m.group(1), "")
        return _text(result)

    async def record_shape(a):
        markers = a.get("markers", {}) or {}
        launch = a.get("launch", {}) or {}
        env = a.get("env", {}) or {}
        shape = derive_shape(markers, launch, env)
        # agent-supplied soft judgments, carried with the derived facts
        comm = a.get("communication", {}) or {}
        if comm.get("intensity"):
            shape.communication.intensity = comm["intensity"]
        if comm.get("pattern"):
            shape.communication.pattern = comm["pattern"]
        shape.confidence = a.get("confidence", shape.confidence)

        report = _current_report(sink, repo, commit)
        if a.get("entrypoint_command"):
            report.entrypoint.command = a["entrypoint_command"]
        report.shapes.append(shape)
        return {
            "content": [
                {
                    "type": "text",
                    "text": "recorded shape: " + ",".join(shape.parallelism),
                }
            ]
        }

    return [
        ToolSpec(
            "list_dir", "List entries under a directory.", {"path": str}, list_dir
        ),
        ToolSpec(
            "find",
            "Find paths under root; optional name_glob and kind (file|dir|elf).",
            {"root": str, "name_glob": str, "kind": str},
            find,
        ),
        ToolSpec(
            "read_text",
            "Read a text file (entrypoints, run scripts, CMakeLists, configs, README).",
            {"path": str},
            read_text,
        ),
        ToolSpec(
            "resolve_entrypoint",
            "Locate and read the launch surface (Dockerfile ENTRYPOINT/CMD/ENV, "
            "run scripts) and parse any launcher flags/env found. Call this first.",
            {"root": str},
            resolve_entrypoint,
        ),
        ToolSpec(
            "scan_shape_markers",
            "Run the deterministic marker library over the tree; returns "
            "categorized, evidenced hits (parallelism, comm, memory, accel, io). "
            "Feed the result straight into record_shape.",
            {"root": str},
            scan_shape_markers,
        ),
        ToolSpec(
            "scan_source",
            "Generic regex sweep over the source tree (path+line hits) for "
            "chasing specifics beyond the marker library.",
            {"root": str, "patterns": list},
            scan_source,
        ),
        ToolSpec(
            "record_shape",
            "Record one schedule shape. The hard categories are DERIVED from the "
            "markers you pass; add only the soft judgments (communication "
            "intensity/pattern) and a confidence. Call once per distinct entrypoint.",
            {
                "markers": dict,
                "launch": dict,
                "env": dict,
                "communication": dict,
                "entrypoint_command": str,
                "confidence": str,
            },
            record_shape,
        ),
    ]


def _current_report(sink: list[ShapeReport], repo: str, commit: str) -> ShapeReport:
    if not sink:
        sink.append(ShapeReport(repo=repo, commit=commit))
    return sink[-1]
