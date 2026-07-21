"""Small terminal-output helper: a bit of color and structure for the live
trace. Colors switch off automatically when stdout isn't a TTY (piped, captured
in tests) or when NO_COLOR is set, so output stays clean everywhere."""

from __future__ import annotations

import os
import sys

_ON = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _wrap(code: str):
    if not _ON:
        return lambda s: str(s)
    return lambda s: f"\033[{code}m{s}\033[0m"


bold = _wrap("1")
dim = _wrap("2")
red = _wrap("31")
green = _wrap("32")
yellow = _wrap("33")
cyan = _wrap("36")


def header(text: str, index: int | None = None, total: int | None = None) -> None:
    tag = dim(f"  [{index}/{total}]") if index and total else ""
    print(f"\n{cyan('▸')} {bold(text)}{tag}", flush=True)


def phase(text: str) -> None:
    print(f"  {dim('·')} {dim(text)}", flush=True)


def op(name: str, detail: str = "") -> None:
    tail = f" {dim(detail)}" if detail else ""
    print(f"  {green('→')} {bold(name)}{tail}", flush=True)


def ok(text: str) -> None:
    print(f"  {green('✓')} {text}", flush=True)


def warn(text: str) -> None:
    print(f"  {yellow('⚠')} {text}", flush=True)


def think(chunk: str) -> None:
    """Streamed assistant text, dimmed so the operation lines stand out."""
    sys.stdout.write(dim(chunk))
    sys.stdout.flush()
