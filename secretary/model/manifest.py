"""The manifest lookup: what this project produces and the next task (selection/
submit) consumes.

Keyed by image digest (immutable, reproducible). Each entry holds how to
reproduce the pull, the digest/tag seen, and the characterized artifacts. Facts
only — no cluster, no jobspec; that projection belongs to the consuming task.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .. import utils
from .artifact import Artifact, Capability, Provenance, Variant

LOOKUP_SCHEMA_VERSION = "artifact-lookup/v1"


def sanitize_segment(part: str) -> str:
    """Make one path segment safe; tags/repos can carry odd characters."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", part) or "_"


def parse_reference(ref: str) -> list[str]:
    """Split an image reference into path segments + a tag segment, e.g.
    'ghcr.io/org/lammps-reax:zen4' -> ['ghcr.io','org','lammps-reax','zen4'].
    Digest refs use 'sha256-<short>' as the tag."""
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
    return [sanitize_segment(x) for x in name.split("/")] + [sanitize_segment(tag)]


@dataclass
class Reproduce:
    reference: str  # what the user asked for (tag or digest ref)
    digest: str = ""  # resolved image digest (RepoDigest), the stable key
    registry: str = ""  # inferred registry/repo, for re-pull
    pulled_by_us: bool = False


@dataclass
class Platform:
    """The container's libc and OS.

    Decides which flux view can be mounted into this image: a view links against
    the CONTAINER's libc, so one built on a newer glibc will not load. A property
    of the image, like arch, so it is recorded here rather than guessed later.
    """

    libc_flavor: str = ""
    libc_version: str = ""
    os_id: str = ""
    os_version_id: str = ""
    os_codename: str = ""


@dataclass
class LookupEntry:
    reproduce: Reproduce
    artifacts: list[Artifact] = field(default_factory=list)
    platform: Platform = field(default_factory=Platform)
    skipped: str = ""  # reason if characterization was skipped
    notes: str = ""

    def key(self) -> str:
        return self.reproduce.digest or self.reproduce.reference

    def as_document(self, version: str) -> dict:
        """A self-contained manifest for one image."""
        return {"version": version, "entry": asdict(self)}


@dataclass
class ManifestLookup:
    version: str = LOOKUP_SCHEMA_VERSION
    entries: dict[str, LookupEntry] = field(default_factory=dict)

    def add(self, entry: LookupEntry) -> None:
        self.entries[entry.key()] = entry

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "entries": {k: asdict(v) for k, v in self.entries.items()},
            },
            indent=2,
            sort_keys=True,
        )

    def write(self, path: str) -> None:
        """Write the whole lookup as one JSON file."""
        utils.write_text(path, self.to_json())

    def save_tree(self, root: str) -> list[str]:
        """Write one manifest.json per image under
        root/<registry>/<org>/<repo>/<tag>/. An entry whose characterization was
        skipped (e.g. an image whose arch we couldn't inspect) carries no facts,
        so no manifest is written for it. Returns the paths written."""
        written = []
        for entry in self.entries.values():
            if entry.skipped:
                continue
            path = "/".join(
                [root, *parse_reference(entry.reproduce.reference), "manifest.json"]
            )
            utils.write_json(path, entry.as_document(self.version))
            written.append(path)
        return written

    @classmethod
    def load(cls, path: str) -> "ManifestLookup":
        raw = utils.read_json(path)
        lk = cls(version=raw.get("version", LOOKUP_SCHEMA_VERSION))
        for k, v in raw.get("entries", {}).items():
            lk.entries[k] = LookupEntry(
                reproduce=Reproduce(**v["reproduce"]),
                artifacts=[artifact_from_dict(a) for a in v.get("artifacts", [])],
                skipped=v.get("skipped", ""),
                notes=v.get("notes", ""),
            )
        return lk


def artifact_from_dict(d: dict) -> Artifact:
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
        a.provenance = Provenance(
            **{
                k: v
                for k, v in d["provenance"].items()
                if k in Provenance.__dataclass_fields__
            }
        )
    a.variants = [Variant(**v) for v in d.get("variants", [])]
    a.evidence = d.get("evidence", {})
    return a
