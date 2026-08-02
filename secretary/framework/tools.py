"""The inspection toolset as framework ToolSpecs, built over any backend that
has the Inspector method surface (local Inspector, or RemoteInspector driving a
container via docker exec). All read-only. record_artifact writes into a sink
and derives capability from the observed linkage."""

from __future__ import annotations

import json
from typing import Any

from ..model.artifact import Artifact, Provenance, derive_capability
from behalf import ToolSpec


def _text(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}]}


def inspection_tools(backend, sink: list[Artifact]) -> list[ToolSpec]:
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

    async def inspect_elf(a):
        return _text(backend.inspect_elf(a["path"]))

    async def scan_strings(a):
        return _text(backend.scan_strings(a["path"], a.get("patterns", [])))

    async def read_text(a):
        return {"content": [{"type": "text", "text": backend.read_text(a["path"])}]}

    async def detect_provenance(a):
        return _text(backend.detect_provenance(a["dir_path"]))

    async def record_artifact(a):
        needed = a.get("needed", []) or []
        rpath = a.get("rpath", []) or []
        # inspect_elf reports device_code; the agent passes it back so a
        # statically linked CUDA or HIP build is still recognised as GPU capable.
        device_code = a.get("device_code", []) or []
        cap = derive_capability(needed, rpath, device_code)
        prov = Provenance(
            **{
                k: v
                for k, v in (a.get("provenance") or {}).items()
                if k in Provenance.__dataclass_fields__
            }
        )
        art = Artifact(
            application=a["application"],
            binary=a["binary"],
            arch=a.get("arch", "unknown"),
            needed=needed,
            rpath=rpath,
            runpath=a.get("runpath", []) or [],
            capability=cap,
            provenance=prov,
            confidence=a.get("confidence", "medium"),
            evidence=a.get("evidence", {}) or {},
        )
        sink.append(art)
        return {"content": [{"type": "text", "text": "recorded " + art.application}]}

    return [
        ToolSpec(
            "list_dir",
            "List entries under a directory. kind is dir|file|elf.",
            {"path": str},
            list_dir,
        ),
        ToolSpec(
            "find",
            "Find paths under root; optional name_glob and kind (elf|file|dir).",
            {"root": str, "name_glob": str, "kind": str},
            find,
        ),
        ToolSpec(
            "inspect_elf",
            "ELF arch, interpreter, and linkage (NEEDED, RPATH, RUNPATH, SONAME).",
            {"path": str},
            inspect_elf,
        ),
        ToolSpec(
            "scan_strings",
            "Grep printable strings for regex patterns (flags, compiler, cuda arch).",
            {"path": str, "patterns": list},
            scan_strings,
        ),
        ToolSpec(
            "read_text",
            "Read a text file (build logs, CMakeCache.txt, spack specs).",
            {"path": str},
            read_text,
        ),
        ToolSpec(
            "detect_provenance",
            "Inspect a build/install dir for how it was built (with evidence).",
            {"dir_path": str},
            detect_provenance,
        ),
        ToolSpec(
            "record_artifact",
            "Record one characterized build variant; capability is derived "
            "from the linkage and from any embedded device code. Pass "
            "device_code straight through from inspect_elf: a CUDA or HIP build "
            "linked against the static runtime has no gpu library to find, and "
            "the fatbin section is the only evidence it can use a GPU. "
            "Call once per distinct build found.",
            {
                "application": str,
                "binary": str,
                "arch": str,
                "needed": list,
                "rpath": list,
                "device_code": list,
                "provenance": dict,
                "confidence": str,
            },
            record_artifact,
        ),
    ]
