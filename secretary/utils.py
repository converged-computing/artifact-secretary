"""Small shared helpers, so file writing isn't hand-rolled in several places."""

from __future__ import annotations

import json
import os
from typing import Any


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def write_json(path: str, obj: Any, sort_keys: bool = True) -> None:
    write_text(path, json.dumps(obj, indent=2, sort_keys=sort_keys))


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
