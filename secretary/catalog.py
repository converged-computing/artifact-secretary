"""List container images (and tags) for a GitHub org/repo, token-free.

The GitHub Packages REST API needs a token with read:packages (and org SSO
authorization), and 403s otherwise — awkward for a quick catalog. For PUBLIC
packages we don't need it: the org packages page enumerates the package names,
and the GHCR registry (OCI Distribution API) lists tags and reports each tag's
real architecture anonymously. That lets us drop arm builds by what they ARE,
not by tag name (e.g. an arm64 image tagged 'hpc7g' is correctly excluded).

Pure stdlib. For private packages you'd need registry auth; that's out of scope.
"""

from __future__ import annotations

import json
import platform
import re
import urllib.parse
import urllib.request

_UA = "artifact-secretary"
_MANIFEST_ACCEPT = ",".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def _json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    with urllib.request.urlopen(req) as r:
        return json.load(r), dict(r.headers)


def _text(url: str) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"}
    )
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "ignore")


# --- package names from the public org packages page --------------------------


def list_packages(org: str, repo: str | None = None) -> list[str]:
    names: list[str] = []
    page = 1
    while True:
        q = {"per_page": "100", "page": str(page), "ecosystem": "container"}
        if repo:
            q["repo_name"] = repo
        html = _text(
            f"https://github.com/orgs/{urllib.parse.quote(org)}/packages?{urllib.parse.urlencode(q)}"
        )
        found = [
            urllib.parse.unquote(m)
            for m in re.findall(r"/packages/container/package/([^\"?]+)", html)
        ]
        new = [n for n in found if n not in names]
        if not new:
            break
        names += new
        page += 1
        if page > 50:  # safety
            break
    return sorted(set(names))


# --- tags and architecture from the anonymous registry ------------------------


def _registry_token(repo_path: str) -> str:
    tok, _ = _json(
        f"https://ghcr.io/token?scope=repository:{repo_path}:pull&service=ghcr.io"
    )
    return tok["token"]


def _tags(repo_path: str, token: str) -> list[str]:
    url = f"https://ghcr.io/v2/{repo_path}/tags/list?n=1000"
    tags: list[str] = []
    while url:
        data, headers = _json(url, {"Authorization": f"Bearer {token}"})
        tags += data.get("tags") or []
        nxt = ""
        for part in headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                nxt = "https://ghcr.io" + part[part.find("<") + 1 : part.find(">")]
        url = nxt
    return tags


def tag_arches(repo_path: str, tag: str, token: str) -> list[str]:
    """linux/<arch> platforms a tag provides. Handles multi-arch indexes and
    single-manifest images (arch read from the config blob)."""
    man, _ = _json(
        f"https://ghcr.io/v2/{repo_path}/manifests/{tag}",
        {"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT},
    )
    if "manifests" in man:  # index / manifest list
        out = []
        for m in man["manifests"]:
            p = m.get("platform") or {}
            if p.get("architecture") and p.get("architecture") != "unknown":
                out.append(f"{p.get('os','linux')}/{p['architecture']}")
        return sorted(set(out))
    cfg = (man.get("config") or {}).get("digest")
    if not cfg:
        return []
    conf, _ = _json(
        f"https://ghcr.io/v2/{repo_path}/blobs/{cfg}",
        {"Authorization": f"Bearer {token}"},
    )
    return [f"{conf.get('os', 'linux')}/{conf.get('architecture', 'unknown')}"]


def list_ghcr(
    org: str,
    repo: str | None = None,
    arch: str = "amd64",
    exclude: tuple[str, ...] = (),
    on_note=None,
) -> list[str]:
    """Image references ghcr.io/<org>/<pkg>:<tag> for a repo. Keeps only tags
    that provide linux/<arch> (set arch='' to keep all). `exclude` drops tags
    containing any of the given substrings."""

    def note(msg):
        if on_note:
            on_note(msg)

    refs: list[str] = []
    packages = list_packages(org, repo)
    note(f"{len(packages)} package(s)")
    for pkg in packages:
        repo_path = f"{org}/{pkg}"
        try:
            token = _registry_token(repo_path)
            tags = _tags(repo_path, token)
        except Exception as e:
            note(f"{pkg}: could not list tags ({e})")
            continue
        kept = 0
        for tag in tags:
            if any(x in tag.lower() for x in exclude):
                continue
            if arch:
                try:
                    if f"linux/{arch}" not in tag_arches(repo_path, tag, token):
                        continue
                except Exception:
                    continue  # unreadable manifest -> skip this tag
            refs.append(f"ghcr.io/{repo_path}:{tag}")
            kept += 1
        note(f"{pkg}: {kept} tag(s)")
    return sorted(set(refs))


# --- host architecture matching (derive arch from the image manifest, not tags) -

_HOST_ARCH = {"x86_64": "amd64", "amd64": "amd64",
              "aarch64": "arm64", "arm64": "arm64", "armv7l": "arm"}


def host_arch() -> str:
    """This machine's arch as an OCI arch string (amd64 / arm64)."""
    m = platform.machine().lower()
    return _HOST_ARCH.get(m, m)


def reference_arches(reference: str) -> list[str]:
    """linux/<arch> platforms a full image reference provides, read from the GHCR
    registry manifest (never the tag string). Returns [] when it can't be
    determined (non-GHCR reference, private, or offline) so callers can fail open.
    """
    if not reference.startswith("ghcr.io/"):
        return []  # only GHCR is readable anonymously here
    body = reference[len("ghcr.io/"):]
    tag = "latest"
    if ":" in body.rsplit("/", 1)[-1]:
        body, tag = body.rsplit(":", 1)
    try:
        token = _registry_token(body)
        return tag_arches(body, tag, token)
    except Exception:
        return []


def host_supports(reference: str) -> tuple[bool, list[str]]:
    """(can this host run the image, platforms it provides). Fail-open: if the
    platforms are unknown, returns True so we don't wrongly skip."""
    arches = reference_arches(reference)
    if not arches:
        return True, arches
    return f"linux/{host_arch()}" in arches, arches
