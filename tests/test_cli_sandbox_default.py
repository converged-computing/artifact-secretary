"""The sandbox has to be the default everywhere.

Reading an untrusted repository straight on the host is unsafe, so host is an
explicit command line opt out that warns. It is never the default and setup never
asks for it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretary.cli.main import main as cli_main
from secretary.source.session import (
    DEFAULT_BASE_IMAGE,
    ContainerSourceSession,
    SourceSession,
)
from secretary.tasks.shape import ShapeTask


def _parser_defaults(argv):
    """Parse the shape args without running anything and hand back the namespace."""
    import argparse
    import contextlib
    import io

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real_parse(self, args, namespace)
        captured["ns"] = ns
        raise SystemExit(0)  # stop before doing any work

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.suppress(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            sys.argv = argv
            cli_main()
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return captured.get("ns")


def test_cli_mode_defaults_to_container():
    ns = _parser_defaults(["artifact-secretary", "shape", "--repos", "https://x/y"])
    assert ns.mode == "container", ns.mode
    assert ns.base_image == DEFAULT_BASE_IMAGE, ns.base_image


def test_task_sandboxes_unless_host_is_explicit():
    """Fail safe, only the exact word host escapes the sandbox."""
    task = ShapeTask()
    for manifest in (
        {},
        {"mode": ""},
        {"mode": "container"},
        {"mode": "HOST"},
        {"mode": "typo"},
    ):
        sess = task._make_session("https://github.com/x/y", manifest)
        assert isinstance(sess, ContainerSourceSession), manifest

    sess = task._make_session("https://github.com/x/y", {"mode": "host"})
    assert isinstance(sess, SourceSession), "explicit host should opt out"


def test_setup_prompt_does_not_offer_host():
    """The conversational path must not present running on the host as a choice."""
    prompt = ShapeTask().setup_system_prompt()
    assert "sandbox container" in prompt
    assert "do not offer to run on the host" in prompt
    # it must not ask the user to pick between the two
    assert '"host"|"container"' not in prompt


def test_host_mode_warns_on_stderr():
    """The opt-out must never be quiet. A missed string edit once dropped this
    warning silently, so assert it reaches stderr."""
    import contextlib
    import io

    from secretary.cli import main as _mainmod

    cli = sys.modules["secretary.cli.main"]

    class _Args:
        backend, model, manifest = "aws", None, None
        max_source_tokens, model_max_tokens = 100, 4096
        out, repos, ref, focus, keep = (
            "/tmp/out.json",
            ["https://x/y"],
            None,
            None,
            False,
        )
        base_image, mode = DEFAULT_BASE_IMAGE, "host"
        yes = False

    def _fake_run(*a, **kw):
        return type("O", (), {"result": None})()

    real_runner, real_anyio_run = cli.make_runner_with_output_cap, cli.anyio.run
    cli.make_runner_with_output_cap = lambda *a, **kw: None
    cli.anyio.run = _fake_run
    try:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cli._cmd_shape(_Args())
        assert "WARNING" in err.getvalue(), err.getvalue()
        assert "no container isolation" in err.getvalue()

        args = _Args()
        args.mode = "container"
        err2 = io.StringIO()
        with contextlib.redirect_stderr(err2):
            cli._cmd_shape(args)
        assert "WARNING" not in err2.getvalue(), "sandbox mode must not warn"
    finally:
        cli.make_runner_with_output_cap, cli.anyio.run = real_runner, real_anyio_run


def test_yes_flag_wires_both_gates():
    """--yes has to approve both the setup gate and the clone gate. Not having it
    is what made every batch rerun fail on an unrecognized argument."""
    import contextlib
    import io

    cli = sys.modules["secretary.cli.main"]
    seen = {}

    class _Result:
        entries = {}

        def save_tree(self, root, label=""):
            return []

        def to_json(self):
            return "{}"

    class _Args:
        backend, model, manifest, mode = "aws", None, None, "container"
        max_source_tokens, model_max_tokens = 100, 4096
        repos, ref, focus, keep = ["https://x/y"], None, None, False
        base_image, out_dir, out, yes = "b", "/tmp/tree", None, True

    def _fake(fn, task, runner, manifest, confirm_fn=None, approve_fn=None):
        seen["confirm"], seen["approve"] = confirm_fn, approve_fn
        return type("O", (), {"result": _Result()})()

    real_runner, real_run = cli.make_runner_with_output_cap, cli.anyio.run
    cli.make_runner_with_output_cap = lambda *a, **kw: None
    cli.anyio.run = _fake
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cli._cmd_shape(_Args())
        assert seen["confirm"] is cli._auto_confirm
        assert seen["approve"] is not None
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert seen["confirm"]("clone", {"repo": "x"}) is True
            assert seen["approve"]({"repos": ["x"]}) is True
        assert "auto-approved" in out.getvalue()
    finally:
        cli.make_runner_with_output_cap = real_runner
        cli.anyio.run = real_run
