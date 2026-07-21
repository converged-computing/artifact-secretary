"""The manifest lookup: the durable artifact this project produces and the next
agent (selection/submit, a separate task) consumes.

Keyed by image digest (immutable, reproducible). Each entry holds how to
reproduce the pull, the digest/tag seen, and the characterized artifacts — one
per build VARIANT (a LAMMPS image may carry a CPU and a CUDA build). Facts only:
no cluster, no jobspec. The projection to any output shape (e.g. fleetq
`requires`) belongs to the consuming task, not here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from .artifact import Artifact, Capability, Provenance, Variant

LOOKUP_SCHEMA_VERSION = "artifact-lookup/v1"


@dataclass
class Reproduce:
    reference: str            # what the user asked for (tag or digest ref)
    digest: str = ""          # resolved image digest (RepoDigest), the stable key
    registry: str = ""        # inferred registry/repo, for re-pull
    pulled_by_us: bool = False  # did this run pull it (=> safe to reap)


@dataclass
class LookupEntry:
    reproduce: Reproduce
    artifacts: list[Artifact] = field(default_factory=list)   # one per build variant
    skipped: str = ""         # reason if characterization was skipped
    notes: str = ""

    def key(self) -> str:
        return self.reproduce.digest or self.reproduce.reference


@dataclass
class ManifestLookup:
    version: str = LOOKUP_SCHEMA_VERSION
    entries: dict[str, LookupEntry] = field(default_factory=dict)

    def add(self, entry: LookupEntry) -> None:
        self.entries[entry.key()] = entry

    def to_json(self) -> str:
        return json.dumps(
            {"version": self.version,
             "entries": {k: asdict(v) for k, v in self.entries.items()}},
            indent=2, sort_keys=True,
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ManifestLookup":
        with open(path) as f:
            raw = json.load(f)
        lk = cls(version=raw.get("version", LOOKUP_SCHEMA_VERSION))
        # entries kept as plain dicts on load; consumers read fields by name.
        for k, v in raw.get("entries", {}).items():
            rep = Reproduce(**v["reproduce"])
            arts = [_artifact_from_dict(a) for a in v.get("artifacts", [])]
            lk.entries[k] = LookupEntry(reproduce=rep, artifacts=arts,
                                        skipped=v.get("skipped", ""), notes=v.get("notes", ""))
        return lk


def _artifact_from_dict(d: dict) -> Artifact:
    a = Artifact(application=d.get("application", ""), binary=d.get("binary", ""))
    a.arch = d.get("arch", "unknown")
    a.interpreter = d.get("interpreter", "")
    a.needed = d.get("needed", [])
    a.rpath = d.get("rpath", [])
    a.runpath = d.get("runpath", [])
    a.confidence = d.get("confidence", "medium")
    if d.get("capability"):
        a.capability = Capability(**d["capability"])
    if d.get("provenance"):
        a.provenance = Provenance(**{k: v for k, v in d["provenance"].items()
                                     if k in Provenance.__dataclass_fields__})
    a.variants = [Variant(**v) for v in d.get("variants", [])]
    a.evidence = d.get("evidence", {})
    return a


# --- writing a per-image manifest tree (registry/org/repo/tag/manifest.json) ---

import os as _os
import re as _re


def _sanitize(part: str) -> str:
    """Make one path segment safe (tags/repos can contain odd characters)."""
    return _re.sub(r"[^A-Za-z0-9._-]", "_", part) or "_"


def ref_to_parts(ref: str) -> list[str]:
    """Split an image reference into path segments + a final tag segment, e.g.
    'ghcr.io/converged-computing/lammps-reax:zen4-reax'
        -> ['ghcr.io','converged-computing','lammps-reax','zen4-reax'].
    Digest refs use 'sha256-<short>' as the tag segment."""
    tag = None
    if "@" in ref:
        name, digest = ref.split("@", 1)
        if ":" in name.rsplit("/", 1)[-1]:
            name, tag = name.rsplit(":", 1)
        else:
            tag = "sha256-" + digest.split(":")[-1][:12]
    else:
        last = ref.rsplit("/", 1)[-1]
        if ":" in last:
            name, tag = ref.rsplit(":", 1)
        else:
            name, tag = ref, "latest"
    return [_sanitize(x) for x in name.split("/")] + [_sanitize(tag)]


def entry_document(version: str, entry: "LookupEntry") -> dict:
    """A self-contained manifest for one image."""
    return {"version": version, "entry": asdict(entry)}


def save_tree(lookup: "ManifestLookup", root: str) -> list[str]:
    """Write one manifest.json per image under root, organized by the URI:
    root/<registry>/<org>/<repo>/<tag>/manifest.json. Returns the paths written."""
    written = []
    for entry in lookup.entries.values():
        parts = ref_to_parts(entry.reproduce.reference)
        d = _os.path.join(root, *parts)
        _os.makedirs(d, exist_ok=True)
        path = _os.path.join(d, "manifest.json")
        with open(path, "w") as f:
            f.write(json.dumps(entry_document(lookup.version, entry), indent=2, sort_keys=True))
        written.append(path)
    return written
