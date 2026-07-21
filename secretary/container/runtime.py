"""Provide a python interpreter we control.

Images ship wildly different pythons — none at all, or one too old to run our
code (we need 3.8+; the pytorch images carry 3.6). Rather than fight each case,
we bring our own: a relocatable standalone CPython that the session docker-cp's
into the container, same as it copies the probe. No image python, no network in
the container, no root, no package manager.

The build is fetched once and cached. It's glibc-only (covers ubuntu/debian/
rocky/conda); musl images (alpine) aren't supported here.
"""

from __future__ import annotations

import io
import os
import tarfile
import urllib.request

# pinned python-build-standalone release (override the whole URL via env if needed)
PBS_TAG = "20260718"
PBS_VERSION = "3.11.15"
BUNDLED_ROOT = "/tmp/pyroot"  # where it lands in the container

_ARCH = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def normalize_arch(machine: str) -> str:
    return _ARCH.get((machine or "").strip(), "x86_64")


def asset_url(arch: str) -> str:
    env = os.environ.get("ARTIFACT_SECRETARY_PYTHON_URL")
    if env:
        return env
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PBS_TAG}/cpython-{PBS_VERSION}%2B{PBS_TAG}-{arch}-unknown-linux-gnu-install_only.tar.gz"
    )


def cache_root() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "artifact-secretary", "python")


def fetch_python(arch: str, opener=urllib.request.urlopen) -> str:
    """Return a local dir holding python/bin/python3 for `arch`, downloading and
    extracting the standalone build the first time (cached thereafter)."""
    dest = os.path.join(cache_root(), arch)
    if os.path.exists(os.path.join(dest, "python", "bin", "python3")):
        return dest
    os.makedirs(dest, exist_ok=True)
    with opener(asset_url(arch)) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(dest)
    return dest
