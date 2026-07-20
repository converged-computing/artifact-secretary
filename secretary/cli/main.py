"""artifact-secretary command line. Client code: it selects a model backend at
runtime (--backend) and imports only that backend's runner, so you don't need
Google's ADK installed to run Claude, or vice versa. A missing backend package
fails immediately, with a clear message, when that backend is selected."""
from __future__ import annotations

import argparse
import json

import anyio

from ..framework.core import AgentRunner, run_task
from ..tasks.profile import ProfileTask


def make_runner(backend: str, model: str | None) -> AgentRunner:
    """Pick a runner by name. Each backend's SDK is imported only when chosen,
    and its absence surfaces here as a clear install hint."""
    if backend == "claude":
        try:
            from ..runner.sdk import SDKRunner
        except ImportError as e:
            raise SystemExit(f"--backend claude needs the Claude Agent SDK: pip install claude-agent-sdk ({e})")
        return SDKRunner(model) if model else SDKRunner()
    if backend == "gemini":
        try:
            from ..runner.adk import ADKRunner
        except ImportError as e:
            raise SystemExit(f"--backend gemini needs Google ADK: pip install google-adk ({e})")
        return ADKRunner(model=model) if model else ADKRunner()
    if backend == "aws":
        try:
            from ..runner.strands import StrandsRunner
        except ImportError as e:
            raise SystemExit(f"--backend aws needs Strands: pip install strands-agents ({e})")
        return StrandsRunner(model=model) if model else StrandsRunner()
    raise SystemExit(f"unknown backend {backend!r} (choose: claude, gemini, aws)")


def main():
    ap = argparse.ArgumentParser(prog="artifact-secretary")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="characterize a catalog of containers")
    p.add_argument("--backend", choices=["claude", "gemini", "aws"], default="claude",
                   help="model backend (default: claude)")
    p.add_argument("--model", default=None, help="model name for the chosen backend")
    p.add_argument("--manifest", help="saved run manifest JSON (skip conversation)")
    p.add_argument("--catalog", nargs="*", help="image refs (skip conversation)")
    p.add_argument("--goal", default="Characterize each image's build variants.")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--out", default="lookup.json")
    args = ap.parse_args()

    manifest = None
    if args.manifest:
        manifest = json.load(open(args.manifest))
    elif args.catalog:
        manifest = {"catalog": args.catalog, "goal": args.goal, "keep_images": args.keep_images}

    runner = make_runner(args.backend, args.model)
    outcome = anyio.run(run_task, ProfileTask(), runner, manifest)
    if outcome.result is not None:
        outcome.result.save(args.out)
        print(f"\nwrote {args.out} ({len(outcome.result.entries)} entries)")


if __name__ == "__main__":
    main()
