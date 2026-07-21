"""artifact-secretary command line. Client code: selects a model backend at
runtime (--backend) and imports only that backend's runner. Two subcommands:
`profile` (characterize a catalog into a manifest tree) and `list` (enumerate
GHCR images/tags for an org/repo to feed the catalog)."""
from __future__ import annotations

import argparse
import json

import anyio

from ..framework.core import AgentRunner, run_task
from ..tasks.profile import ProfileTask
from ..model.manifest import save_tree


def make_runner(backend: str, model: str | None) -> AgentRunner:
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


def _cmd_profile(args):
    manifest = None
    if args.manifest:
        manifest = json.load(open(args.manifest))
    elif args.catalog:
        manifest = {"catalog": args.catalog, "goal": args.goal, "keep_images": args.keep_images,
                    "install_python": args.install_python,
                    "install_network": args.install_network}

    runner = make_runner(args.backend, args.model)
    outcome = anyio.run(run_task, ProfileTask(), runner, manifest)
    if outcome.result is not None:
        paths = save_tree(outcome.result, args.out_dir)
        print(f"\nwrote {len(paths)} manifest(s) under {args.out_dir}/")
        for p in paths:
            print(f"  {p}")


def _cmd_list(args):
    import sys
    from ..catalog import list_ghcr
    arch = "" if args.all_arches else args.arch
    # progress to stderr so stdout stays clean image refs (pipeable into profile)
    refs = list_ghcr(args.org, args.repo, arch=arch, exclude=tuple(args.exclude),
                     on_note=lambda m: print(m, file=sys.stderr))
    for ref in refs:
        print(ref)


def main():
    ap = argparse.ArgumentParser(prog="artifact-secretary")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="characterize a catalog of containers into a manifest tree")
    p.add_argument("--backend", choices=["claude", "gemini", "aws"], default="claude")
    p.add_argument("--model", default=None, help="model name for the chosen backend")
    p.add_argument("--manifest", help="saved run manifest JSON (skip conversation)")
    p.add_argument("--catalog", nargs="*", help="image refs (skip conversation); pass one to run a single image")
    p.add_argument("--goal", default="Characterize each image's build variants.")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument("--no-install-python", dest="install_python", action="store_false",
                   help="do not install python3 into images that lack it (skip them instead)")
    p.add_argument("--install-network", default="bridge",
                   help="docker network to attach transiently when installing python3 (default: bridge)")
    p.set_defaults(install_python=True)
    p.add_argument("--out-dir", default="manifests",
                   help="root of the manifest tree (registry/org/repo/tag/manifest.json)")
    p.set_defaults(func=_cmd_profile)

    l = sub.add_parser("list", help="list GHCR images/tags for an org (optionally one repo)")
    l.add_argument("--org", required=True)
    l.add_argument("--repo", default=None, help="restrict to packages published to this repo")
    l.add_argument("--arch", default="amd64",
                   help="keep only tags that provide this linux arch (default: amd64)")
    l.add_argument("--all-arches", action="store_true", help="do not filter by architecture")
    l.add_argument("--exclude", nargs="*", default=[],
                   help="also drop tags containing any of these substrings")
    l.set_defaults(func=_cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
