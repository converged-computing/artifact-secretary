"""The source exploration tools the agent gets.

Source analysis only. Every primitive reads source text and nothing here touches
a binary or a container config. The base image is a sandbox, never a subject.

No entrypoint seed and no marker table. The tools give repeatable structural
facts, the agent supplies judgment, and the schema constrains the output. Evidence
is required, and a negative claim has to cite a search rather than a line.
"""

from __future__ import annotations

import json
import re
from typing import Any

from behalf import ToolSpec

from ..model.shape import Assertion, ScheduleShape, ShapeReport, ShapeSchema

# two kinds of evidence. a location backs a positive claim, and cannot back a
# negative one since pointing at a line proves nothing is missing
LOCATION_RX = re.compile(r"^[^\s:]+:\d+")
# pull the path and line span back out of an evidence string
SPAN_RX = re.compile(r"^(?P<path>[^\s:()]+):(?P<lines>\d+(?:\s*[-,]\s*\d+)*)")

# cap so one stray huge citation cannot bloat a report
MAX_SNIPPET_LINES_PER_REPORT = 600


def parse_span(ev: str):
    m = SPAN_RX.match(ev.strip())
    if not m:
        return None
    nums = [int(n) for n in re.findall(r"\d+", m.group("lines"))]
    if not nums:
        return None
    return m.group("path"), min(nums), max(nums)


ABSENCE_RX = re.compile(r"^absent:", re.I)


def evidence_kind(ev: str) -> str:
    if ABSENCE_RX.match(ev.strip()):
        return "absence"
    if LOCATION_RX.match(ev.strip()):
        return "location"
    return "prose"


def _slug_variant(variant: str, schema: ShapeSchema) -> str:
    """Normalise a variant name so two spellings of one build match.

    All of the vocabulary comes from the schema. Anything not a terminal, like a
    package name, stays as a free token since the tool should not know app names.
    """
    rules = schema.variants or {}
    known = schema.variant_tokens()
    aliases = rules.get("aliases") or {}
    subsumes = rules.get("subsumes") or {}
    limit = int(rules.get("max_length") or 40)

    original = variant or ""
    raw = re.sub(r"[^a-z0-9]+", " ", original.lower()).split()
    if not raw:
        return "general"
    # whitespace means prose, and stray words cannot be canonicalised
    prose = bool(re.search(r"[\s(),/]", original.strip()))

    seen = []
    for tok in raw:
        t = tok
        for _ in range(4):  # aliases can chain
            nxt = aliases.get(t)
            if not nxt or nxt == t:
                break
            t = nxt
        if t not in seen:
            seen.append(t)

    for tok, covered in subsumes.items():
        if tok in seen:
            seen = [t for t in seen if t not in covered or t == tok]

    ordered = [t for t in known if t in seen]
    # only a few free tokens, or prose turns into a long slug
    keep = int(rules.get("max_free_tokens") or 2)
    if prose and not rules.get("free_tokens_from_prose", False):
        keep = 0
    free = sorted(t for t in seen if t not in known)[:keep]
    slug = "-".join(ordered + free)[:limit].strip("-")
    return slug or "general"


def _text(obj: Any) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(obj, indent=2)}]}


def source_tools(
    reader,
    schema: ShapeSchema,
    sink: list[ShapeReport],
    repo: str = "",
    commit: str = "",
) -> list[ToolSpec]:
    backend = reader.backend

    async def read(a):
        return _text(reader.read(a["path"], a.get("start", 1), a.get("count", 40)))

    async def search(a):
        return _text(reader.search(a["pattern"], a.get("context"), a.get("limit", 30)))

    async def find(a):
        # repo relative both ways, the clone path must not reach the agent
        hits = backend.find(
            reader.to_backend(a.get("root", "") or "."),
            a.get("name_glob") or None,
            a.get("kind") or None,
            a.get("limit", 200),
        )
        return _text([reader.to_repo(h) for h in hits])

    async def seen(a):
        return _text(reader.seen())

    async def record_shape(a):
        """Record assertions into the trace for ONE subject/variant. Separate
        subjects (or variants) get separate shape lists -- never pooled."""
        subject = (a.get("subject") or "").strip()
        if not subject:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "refused: `subject` is required so the shape lands in the right trace",
                    }
                ]
            }
        report = _current_report(sink, repo, commit)
        trace = report.trace(
            subject, _slug_variant(a.get("variant") or "general", schema)
        )
        if a.get("confidence"):
            trace.confidence = a["confidence"]
        if a.get("notes"):
            trace.notes = a["notes"]

        shape = ScheduleShape(
            label=a.get("label", ""), launch_command=a.get("launch_command", "")
        )
        recorded, rejected, unmatched = 0, [], 0
        for raw in a.get("assertions", []) or []:
            fld = raw.get("field", "")
            evidence = raw.get("evidence", []) or []
            if not evidence:
                rejected.append(
                    f"{fld}: refused, an assertion must cite evidence (path:line, or "
                    f"'absent:<pattern> in <scope>' for an absence)"
                )
                continue
            # absence needs a recorded search, a line proves nothing
            negative = schema.is_negative(raw.get("value"))
            if negative and {evidence_kind(e) for e in evidence} == {"location"}:
                rejected.append(
                    f"{fld}: refused, a negative claim needs 'absent:<pattern> in "
                    f"<scope>' evidence, not a line reference"
                )
                continue
            assertion = Assertion(
                field=fld,
                value=raw.get("value"),
                evidence=evidence,
                confidence=raw.get("confidence", "medium"),
            )
            # grab the cited code while the clone is still there
            for ev in evidence if hasattr(reader, "extract") else []:
                span = parse_span(ev)
                if not span:
                    continue
                path, lo, hi = span
                key = f"{path}:{lo}-{hi}" if hi != lo else f"{path}:{lo}"
                if key in report.snippets:
                    continue
                stored = sum(
                    len(v.get("lines") or []) for v in report.snippets.values()
                )
                if stored >= MAX_SNIPPET_LINES_PER_REPORT:
                    continue
                lines = reader.extract(path, lo, hi)
                if lines:
                    report.snippets[key] = {
                        "path": path,
                        "start": lo,
                        "end": hi,
                        "lines": [[n, t] for n, t in lines],
                    }
            ok, error = schema.validate(fld, assertion.value)
            if ok:
                shape.assertions.append(assertion)
                recorded += 1
            else:
                assertion.error = error
                trace.unmatched.append(assertion)  # candidate for schema growth
                unmatched += 1
        if shape.assertions:
            trace.shapes.append(shape)
        msg = f"recorded {recorded} assertion(s) under trace {trace.key()}"
        if unmatched:
            msg += f", {unmatched} unmatched (kept as schema candidates)"
        if rejected:
            msg += "; rejected: " + "; ".join(rejected)
        return {"content": [{"type": "text", "text": msg}]}

    return [
        ToolSpec(
            "find",
            "List paths under root; optional name_glob and kind (file|dir|elf). Use it "
            "to orient: build files (CMakeLists.txt, Makefile), src/ layout, docs.",
            {"root": str, "name_glob": str, "kind": str},
            find,
        ),
        ToolSpec(
            "search",
            "Find your own regex across the source tree, returning each hit with "
            "surrounding context and a `read_more` cursor. This is the main tool: ask a "
            "question (does it use MPI collectives? shared memory? OpenMP?) and search "
            "for the evidence.",
            {"pattern": str, "context": int, "limit": int},
            search,
        ),
        ToolSpec(
            "read",
            "Read a window of a file: `count` lines from `start` (1-based). Returns a "
            "`next` cursor; call again with it to keep reading. Reads are cached and "
            "charged once per line, so re-reads are free.",
            {"path": str, "start": int, "count": int},
            read,
        ),
        ToolSpec(
            "seen",
            "Coverage + remaining token budget for this run: which files/lines you have "
            "already read. Check it to avoid re-reading and to know when to stop.",
            {},
            seen,
        ),
        ToolSpec(
            "record_shape",
            "Record scheduling assertions for ONE subject/variant. `subject` is the "
            "executable/target (e.g. 'lmp'); `variant` is the build or config path "
            "(e.g. 'reaxff', 'kokkos-cuda', 'general'). Different subjects/variants are "
            "kept as SEPARATE traces with separate shapes -- never pool them. Each "
            "assertion is {field, value, evidence:[file:line], confidence}; `field` must "
            "be a grammar path from the schema you were given, unknown fields are kept "
            "as candidates, and an assertion with no evidence is refused.",
            {
                "subject": str,
                "variant": str,
                "label": str,
                "launch_command": str,
                "assertions": list,
                "confidence": str,
                "notes": str,
            },
            record_shape,
        ),
    ]


def _current_report(sink: list[ShapeReport], repo: str, commit: str) -> ShapeReport:
    if not sink:
        sink.append(ShapeReport(repo=repo, commit=commit))
    return sink[-1]
