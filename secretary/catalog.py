"""List container images (and tags) published to GitHub Container Registry for an
org/repo, so a catalog can be derived programmatically instead of typed by hand.

Uses the GitHub Packages API, which needs a token with read:packages even for
public packages. Set GITHUB_TOKEN (or pass token=). Pure stdlib.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

API = "https://api.github.com"


def _get(url: str, token: str) -> tuple[list, dict]:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "artifact-secretary",
    })
    with urllib.request.urlopen(req) as r:
        return json.load(r), dict(r.headers)


def _paginate(url: str, token: str):
    while url:
        data, headers = _get(url, token)
        for item in data:
            yield item
        # follow RFC 5988 Link: <...>; rel="next"
        nxt = ""
        for part in headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                nxt = part[part.find("<") + 1:part.find(">")]
        url = nxt


def list_ghcr(org: str, repo: str | None = None,
              exclude: tuple[str, ...] = ("arm", "aarch64"),
              token: str | None = None) -> list[str]:
    """Return image references ghcr.io/<org>/<pkg>:<tag> for an org, optionally
    restricted to packages linked to `repo`. Tags containing any `exclude`
    substring are dropped (default: arm builds)."""
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("set GITHUB_TOKEN (needs read:packages) to list GHCR packages")

    refs: list[str] = []
    pkgs_url = f"{API}/orgs/{urllib.parse.quote(org)}/packages?package_type=container&per_page=100"
    for pkg in _paginate(pkgs_url, token):
        if repo and (pkg.get("repository") or {}).get("name") != repo:
            continue
        name = pkg["name"]
        enc = urllib.parse.quote(name, safe="")
        versions_url = f"{API}/orgs/{urllib.parse.quote(org)}/packages/container/{enc}/versions?per_page=100"
        for ver in _paginate(versions_url, token):
            for tag in ((ver.get("metadata") or {}).get("container") or {}).get("tags", []):
                low = tag.lower()
                if any(x in low for x in exclude):
                    continue
                refs.append(f"ghcr.io/{org}/{name}:{tag}")
    return sorted(set(refs))
