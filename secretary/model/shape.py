"""How a source repository wants to run, so a scheduler can place it.

The source side sibling of the Artifact, but not a fixed table. Source is open
ended and the same intent gets spelled a hundred ways, so a regex table over it
misses real signal and invents false signal.

The traversal is cached and repeatable, the grammar comes from shape_schema.json,
and every assertion cites the lines it stands on. The classification is agent
judgment, held only to the schema terminals on the way out.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "shape_schema.json")


# --- grammar, loaded not hard coded -------------------------------------------


@dataclass
class ShapeSchema:
    """The scheduling grammar, loaded from shape_schema.json. Code only enforces
    conformance to whatever the schema says, never a fixed set of strings."""

    version: str
    fields: dict[
        str, dict
    ]  # dotted field path -> {type, terminals, judgment, description}
    variants: dict = field(default_factory=dict)  # how a variant name normalises
    negative_terminals: list = field(default_factory=list)  # values meaning not found

    @classmethod
    def load(cls, path: str = DEFAULT_SCHEMA_PATH) -> "ShapeSchema":
        with open(path) as fh:
            raw = json.load(fh)
        return cls(
            version=raw["version"],
            fields=raw["fields"],
            variants=raw.get("variants") or {},
            negative_terminals=raw.get("negative_terminals") or [],
        )

    def validate(self, field_path: str, value: Any) -> tuple[bool, str]:
        """Returns ok and a reason. A miss is not a failure, the caller keeps it
        in unmatched as a candidate the schema might grow to include."""
        spec = self.fields.get(field_path)
        if spec is None:
            return False, f"unknown field {field_path!r}"
        t = spec["type"]
        if t == "bool":
            return (isinstance(value, bool), "expected a boolean")
        if t == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
            return (ok, "expected an integer")
        if t == "enum":
            return (value in spec["terminals"], f"{value!r} not in {spec['terminals']}")
        if t == "set":
            if not isinstance(value, list):
                return False, "expected a list of terminals"
            bad = [v for v in value if v not in spec["terminals"]]
            return (False, f"{bad!r} not in {spec['terminals']}") if bad else (True, "")
        return False, f"unknown field type {t!r}"

    def is_negative(self, value) -> bool:
        """Does this value mean nothing was found."""
        return value is False or value is None or value in self.negative_terminals

    def variant_tokens(self) -> list[str]:
        """Tokens a variant name can use, taken from the schema terminals. No list
        in code, and no app names, since a package name is a free token."""
        out = []
        for path in self.variants.get("tokens_from") or []:
            spec = self.fields.get(path) or {}
            for t in spec.get("terminals") or []:
                if t not in self.negative_terminals and t not in out:
                    out.append(t)
        for t in self.variants.get("extra_tokens") or []:
            if t not in out:
                out.append(t)
        return out

    def describe(self) -> str:
        """Render the grammar for the agent so it targets the current terminals."""
        lines = [f"schedule-shape grammar {self.version}:"]
        for path, spec in self.fields.items():
            vocab = ""
            if spec["type"] in ("enum", "set"):
                vocab = "  {" + " | ".join(spec["terminals"]) + "}"
            elif spec["type"] in ("bool", "int"):
                vocab = f"  <{spec['type']}>"
            judged = "  (judgment)" if spec.get("judgment") else ""
            lines.append(
                f"  {path} [{spec['type']}]{vocab}{judged}  -- {spec['description']}"
            )
        return "\n".join(lines)


# --- what the agent records, each with its evidence ---------------------------


@dataclass
class Assertion:
    field: str  # a dotted grammar path like communication.pattern
    value: Any  # a schema-valid terminal / bool / int / list
    evidence: list[str] = field(
        default_factory=list
    )  # the path and line spans it stands on
    confidence: str = "medium"  # low|medium|high
    # only set when the schema rejected this, so it lives in unmatched
    # recorded assertions never carry one and it is dropped from the output
    error: str = ""


@dataclass
class ScheduleShape:
    """What a scheduler needs to know about one way of running a subject. A trace
    can hold several when the evidence separates distinct regimes."""

    label: str = ""  # optional short name for this regime, like multi-node
    launch_command: str = ""  # the invocation this shape was read from, if any
    assertions: list[Assertion] = field(default_factory=list)

    def to_hint(self) -> dict:
        """Project the assertions into the nested dict a scheduler consumes. Just
        a regroup by the dotted field paths the agent already committed to."""
        out: dict = {}
        for a in self.assertions:
            parts = a.field.split(".")
            d = out
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = a.value
        return out


# --- the result, a report per repo keyed by commit ----------------------------

SHAPE_SCHEMA_VERSION = "schedule-shape/report/v1"


def sanitize_segment(part: str) -> str:
    """Make one path segment safe, repo and focus strings carry odd characters."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", part).strip("_") or "_"


def parse_repo(repo: str) -> list[str]:
    """Split a repo url or path into segments for the output tree, so
    github.com/lammps/lammps becomes github.com, lammps, lammps. Mirrors
    manifest.parse_reference so source and image trees line up."""
    r = repo.strip().rstrip("/")
    if r.endswith(".git"):
        r = r[:-4]
    m = re.match(r"^(?:https?://|git://|ssh://)?(?:[^@/]+@)?([^/:]+)[/:](.+)$", r)
    if m and "." in m.group(1):  # looks like a host, so a remote url
        host, rest = m.group(1), m.group(2)
        return [sanitize_segment(host)] + [sanitize_segment(x) for x in rest.split("/")]
    return ["local", sanitize_segment(r.lstrip("/").replace("/", "_"))]


@dataclass
class RunInfo:
    """How this report was produced. Without it a report cannot be attributed to
    a model or a budget, so no later comparison is possible."""

    model: str = ""
    backend: str = ""
    mode: str = ""  # container|host
    base_image: str = ""
    focus: str = ""
    max_source_tokens: int = 0
    model_max_tokens: int = 0
    started: str = ""  # ISO8601 UTC
    duration_s: float = 0.0
    tool_version: str = ""


@dataclass
class ShapeTrace:
    """One subject and variant traced through the source, with its own shapes.

    Two builds of one code are different applications to a scheduler, so their
    shapes never get pooled.
    """

    subject: str  # the executable being traced, like lmp
    variant: str = "general"  # the build or config path, like kokkos-cuda or reaxff
    shapes: list[ScheduleShape] = field(default_factory=list)
    unmatched: list[Assertion] = field(default_factory=list)  # candidate schema growth
    confidence: str = "medium"  # low when this is only a rough how it generally works
    notes: str = ""

    def key(self) -> str:
        return f"{self.subject}:{self.variant}"


@dataclass
class ShapeReport:
    repo: str
    commit: str = ""
    schema_version: str = ""
    resolved_from: str = ""  # the description the repo URL was resolved from, if any
    run: RunInfo = field(default_factory=RunInfo)
    traces: list[ShapeTrace] = field(default_factory=list)
    # cited code, grabbed while the clone still exists so it stays auditable
    # keyed by span because about half of all citations repeat one
    snippets: dict = field(default_factory=dict)
    reads: dict = field(default_factory=dict)  # coverage and budget, the run provenance
    skipped: str = ""
    notes: str = ""

    def key(self) -> str:
        return f"{self.repo}@{self.commit}" if self.commit else self.repo

    def trace(self, subject: str, variant: str = "general") -> ShapeTrace:
        """Get or create the trace for one subject and variant, so assertions always
        land in the right list instead of a shared pool."""
        for t in self.traces:
            if t.subject == subject and t.variant == variant:
                return t
        t = ShapeTrace(subject=subject, variant=variant)
        self.traces.append(t)
        return t


def _drop_empty_errors(node):
    """Drop error keys that carry no error so a clean report has no empty fields.
    Only this one key, because pruning anything falsy would delete real values
    like bandwidth_sensitive being False."""
    if isinstance(node, dict):
        if node.get("error", None) == "":
            node.pop("error")
        for v in node.values():
            _drop_empty_errors(v)
    elif isinstance(node, list):
        for v in node:
            _drop_empty_errors(v)
    return node


@dataclass
class ShapeLookup:
    version: str = SHAPE_SCHEMA_VERSION
    entries: dict[str, ShapeReport] = field(default_factory=dict)

    def add(self, report: ShapeReport) -> None:
        self.entries[report.key()] = report

    def save_tree(self, root: str, label: str = "") -> list[str]:
        """One shapes.json per repo under root/host/org/repo/revision, so runs do
        not overwrite each other. Revision is unpinned when we could not get one,
        which makes a failed pin visible. Skipped entries write nothing.
        """
        from .. import utils

        written = []
        for entry in self.entries.values():
            if entry.skipped:
                continue
            segments = parse_repo(entry.repo)
            segments.append(
                sanitize_segment(entry.commit[:12]) if entry.commit else "unpinned"
            )
            if label:
                segments.append(sanitize_segment(label))
            path = "/".join([root.rstrip("/"), *segments, "shapes.json"])
            utils.write_json(
                path,
                _drop_empty_errors({"version": self.version, "entry": asdict(entry)}),
            )
            written.append(path)
        return written

    def to_json(self) -> str:
        return json.dumps(
            _drop_empty_errors(
                {
                    "version": self.version,
                    "entries": {k: asdict(v) for k, v in self.entries.items()},
                }
            ),
            indent=2,
            sort_keys=True,
        )
