"""artifact-secretary command line. Client code: selects a model backend at
runtime (--backend) and imports only that backend's runner. Two subcommands:
`profile` (characterize a catalog into a manifest tree) and `list` (enumerate
GHCR images/tags for an org/repo to feed the catalog)."""

from __future__ import annotations

import argparse
import json
import sys

import anyio

from behalf import AgentRunner, default_confirm, make_runner, run_task
from ..tasks.profile import ProfileTask
from ..runner import DEFAULT_MODEL_MAX_TOKENS, make_runner_with_output_cap
from ..source.session import DEFAULT_BASE_IMAGE
from ..tasks.shape import ShapeTask


def _cmd_profile(args):
    manifest = None
    if args.manifest:
        manifest = json.load(open(args.manifest))
    elif args.catalog:
        manifest = {
            "catalog": args.catalog,
            "goal": args.goal,
            "keep_images": args.keep_images,
            "install_python": args.install_python,
            "allow_network": args.allow_network,
            "network": args.network,
            "any_arch": args.any_arch,
        }

    runner = make_runner(args.backend, args.model)
    outcome = anyio.run(run_task, ProfileTask(), runner, manifest)
    if outcome.result is not None:
        paths = outcome.result.save_tree(args.out_dir)
        print(f"\nwrote {len(paths)} manifest(s) under {args.out_dir}/")
        for p in paths:
            print(f"  {p}")


def _auto_confirm(tool_name: str, args: dict) -> bool:
    """Approve without prompting, but print what was approved so an unattended
    sweep still leaves a record of the actions it took."""
    print(f"[auto-approved] {tool_name}: {json.dumps(args, sort_keys=True)}")
    return True


def _cmd_shape(args):
    manifest = None
    if args.manifest:
        manifest = json.load(open(args.manifest))
    elif args.repos:
        manifest = {
            "repos": args.repos,
            "mode": args.mode,
            "base_image": args.base_image,
            "ref": args.ref,
            "keep": args.keep,
            "focus": args.focus,
        }

    if (manifest or {}).get("mode") == "host":
        print(
            "WARNING: --mode host clones and analyses an untrusted repository "
            "directly on this machine, with no container isolation. The sandbox "
            "(default) is the supported path.",
            file=sys.stderr,
        )

    runner = make_runner_with_output_cap(
        args.backend, args.model, args.model_max_tokens
    )
    # a batch sweep cannot answer prompts, so --yes approves both gates in
    # process. piping yes in instead makes every successful run look failed under
    # pipefail, because yes takes a sigpipe and exits nonzero
    confirm_fn = _auto_confirm if args.yes else default_confirm
    approve_fn = (lambda m: _auto_confirm("finalize-setup", m)) if args.yes else None
    outcome = anyio.run(
        run_task,
        ShapeTask(
            max_tokens=args.max_source_tokens,
            model=args.model or "",
            backend=args.backend,
            model_max_tokens=args.model_max_tokens,
        ),
        runner,
        manifest,
        confirm_fn,
        approve_fn,
    )
    if outcome.result is None:
        return
    # tree layout by default, one shapes.json per repo run keyed by revision and
    # by focus so several focused runs of one repo sit side by side
    paths = outcome.result.save_tree(args.out_dir, label=args.focus or "")
    if paths:
        print(f"\nwrote {len(paths)} shape report(s) under {args.out_dir}/")
        for p in paths:
            print(f"  {p}")
    else:
        print("\nno shape reports written (all entries skipped)")
    if args.out:  # optional single-file copy of the whole lookup
        with open(args.out, "w") as fh:
            fh.write(outcome.result.to_json())
        print(f"  (also wrote the combined lookup to {args.out})")


def _cmd_list(args):
    import sys

    from ..catalog import list_ghcr

    arch = "" if args.all_arches else args.arch
    # progress to stderr so stdout stays clean image refs (pipeable into profile)
    refs = list_ghcr(
        args.org,
        args.repo,
        arch=arch,
        exclude=tuple(args.exclude),
        on_note=lambda m: print(m, file=sys.stderr),
    )
    for ref in refs:
        print(ref)


def main():
    ap = argparse.ArgumentParser(prog="artifact-secretary")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "profile", help="characterize a catalog of containers into a manifest tree"
    )
    p.add_argument("--backend", choices=["claude", "gemini", "aws"], default="claude")
    p.add_argument("--model", default=None, help="model name for the chosen backend")
    p.add_argument("--manifest", help="saved run manifest JSON (skip conversation)")
    p.add_argument(
        "--catalog",
        nargs="*",
        help="image refs (skip conversation); pass one to run a single image",
    )
    p.add_argument("--goal", default="Characterize each image's build variants.")
    p.add_argument("--keep-images", action="store_true")
    p.add_argument(
        "--any-arch",
        action="store_true",
        help="do not skip images whose manifest arch differs from the host",
    )
    p.add_argument(
        "--no-install-python",
        dest="install_python",
        action="store_false",
        help="do not bring a bundled python3 into images whose python is missing/too old (skip them)",
    )
    p.add_argument(
        "--allow-network",
        action="store_true",
        help="run the container on a network (posture only; inspection never needs it). "
        "default is sealed with no network",
    )
    p.add_argument(
        "--network",
        default="bridge",
        help="docker network to use with --allow-network (default: bridge)",
    )
    p.set_defaults(install_python=True)
    p.add_argument(
        "--out-dir",
        default="manifests",
        help="root of the manifest tree (registry/org/repo/tag/manifest.json)",
    )
    p.set_defaults(func=_cmd_profile)

    s = sub.add_parser(
        "shape", help="characterize how a repository wants to run (schedule shape)"
    )
    s.add_argument("--backend", choices=["claude", "gemini", "aws"], default="claude")
    s.add_argument("--model", default=None, help="model name for the chosen backend")
    s.add_argument("--manifest", help="saved run manifest JSON (skip conversation)")
    s.add_argument(
        "--repos",
        nargs="*",
        help="repo URLs to clone or paths already present (skip conversation)",
    )
    s.add_argument(
        "--mode",
        choices=["host", "container"],
        default="container",
        help="where to clone and analyse. the default sandbox container is the "
        "supported path; 'host' runs untrusted repository contents directly on "
        "your machine and is an unsafe escape hatch for debugging only",
    )
    s.add_argument(
        "--base-image",
        default=DEFAULT_BASE_IMAGE,
        help=f"sandbox base image to clone into; must ship git and be glibc-based "
        f"(default: {DEFAULT_BASE_IMAGE})",
    )
    s.add_argument("--ref", default=None, help="branch or tag to clone")
    s.add_argument(
        "--yes",
        action="store_true",
        help="approve the clone and setup gates without prompting (for batch runs)",
    )
    s.add_argument(
        "--focus",
        default=None,
        help="which build/config to trace (e.g. 'reaxff'); omit to enumerate variants",
    )
    s.add_argument(
        "--max-source-tokens",
        "--max-tokens",  # old name, kept so existing commands still work
        dest="max_source_tokens",
        type=int,
        default=60_000,
        help="ceiling on SOURCE served into the model's context while exploring "
        "(does not affect how much the model may generate)",
    )
    s.add_argument(
        "--model-max-tokens",
        type=int,
        default=DEFAULT_MODEL_MAX_TOKENS,
        help=f"cap on what the model may GENERATE per response. raise this if a "
        f"run dies with MaxTokensReachedException (default: "
        f"{DEFAULT_MODEL_MAX_TOKENS})",
    )
    s.add_argument(
        "--keep", action="store_true", help="keep the clone / pulled base image"
    )
    s.add_argument(
        "--out-dir",
        default="shapes",
        help="root of the shape tree: <host>/<org>/<repo>/<revision>[/<focus>]/"
        "shapes.json (default: shapes)",
    )
    s.add_argument(
        "--out",
        default=None,
        help="also write the whole lookup to this single JSON file",
    )
    s.set_defaults(func=_cmd_shape)

    l = sub.add_parser(
        "list", help="list GHCR images/tags for an org (optionally one repo)"
    )
    l.add_argument("--org", required=True)
    l.add_argument(
        "--repo", default=None, help="restrict to packages published to this repo"
    )
    l.add_argument(
        "--arch",
        default="amd64",
        help="keep only tags that provide this linux arch (default: amd64)",
    )
    l.add_argument(
        "--all-arches", action="store_true", help="do not filter by architecture"
    )
    l.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="also drop tags containing any of these substrings",
    )
    l.set_defaults(func=_cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
