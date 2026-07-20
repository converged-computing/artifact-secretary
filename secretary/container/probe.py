"""Runs INSIDE a container via `docker exec python3 -m secretary.container.probe
<cmd> <json-args>`. Runs the same Inspector over the container root and prints
JSON. Bind-mounted in with pyelftools; needs only a recent python3 in the image."""
import json
import sys

from ..inspection.target import Target
from ..inspection.inspector import Inspector


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(json.dumps({"error": "usage: <cmd> <json-args>"}))
        return 2
    cmd = argv[0]
    args = json.loads(argv[1]) if len(argv) > 1 and argv[1] else {}
    args = {k: v for k, v in args.items() if v is not None}
    fn = getattr(Inspector(Target("/")), cmd, None)
    if fn is None:
        print(json.dumps({"error": f"unknown command {cmd!r}"}))
        return 2
    print(json.dumps(fn(**args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
