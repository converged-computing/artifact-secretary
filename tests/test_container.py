import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary.container import runtime
from secretary.container.remote import RemoteInspector
from secretary.container.session import ContainerSession, SkipContainer


class FakeDocker:
    """A docker CLI stand-in. py_version=None means the image has no python3."""

    def __init__(self, present=False, image_python_ok=True, arch="x86_64"):
        self.present, self.image_python_ok, self.arch = present, image_python_ok, arch
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        cp = lambda rc=0, out="", err="": subprocess.CompletedProcess(
            argv, rc, out, err
        )
        a = argv
        if a[:3] == ["docker", "image", "inspect"]:
            return cp(0 if self.present else 1)
        if a[:2] == ["docker", "pull"]:
            return cp(0)
        if a[:3] == ["docker", "inspect", "--format"]:
            return cp(0, out=a[-1] + "@sha256:cafef00d\n")
        if a[:2] == ["docker", "run"]:
            return cp(0, out="container123\n")
        if a[:2] == ["docker", "cp"]:
            return cp(0)
        if "-c" in a and "import secretary.container.probe" in a:  # capability check
            interp = a[a.index("-c") - 1]
            if interp.startswith("/tmp/pyroot"):
                return cp(0)  # bundled python always works
            return cp(0 if self.image_python_ok else 1, err="SyntaxError: match")
        if a[-2:] == ["uname", "-m"]:
            return cp(0, out=self.arch + "\n")
        if "-m" in a and "secretary.container.probe" in a:  # actual probe op
            return cp(
                0,
                out=json.dumps(
                    {"is_elf": True, "arch": "arm64", "needed": ["libc.so.6"]}
                ),
            )
        if a[:3] == ["docker", "rm", "-f"]:
            return cp(0)
        if a[:2] == ["docker", "rmi"]:
            return cp(0)
        return cp(1, err="unexpected " + " ".join(a))

    def ran(self, *prefix):
        return any(c[: len(prefix)] == list(prefix) for c in self.calls)


def _no_download(monkey_dir="/fake/pyroot"):
    # avoid a real network fetch of the standalone python in tests
    runtime.fetch_python = lambda arch, **k: monkey_dir


def test_absent_image_pulled_probe_copied_then_reaped():
    fd = FakeDocker(present=False)
    with ContainerSession("reg/x:t", runner=fd) as insp:
        assert isinstance(insp, RemoteInspector)
    assert (
        fd.ran("docker", "pull") and fd.ran("docker", "cp") and fd.ran("docker", "rmi")
    )
    assert not any(
        "-v" in c for c in fd.calls
    ), "no bind mount (docker-outside-of-docker safe)"
    print("OK absent image: pulled + probe copied + reaped")


def test_present_image_not_reaped():
    fd = FakeDocker(present=True)
    with ContainerSession("reg/x:t", runner=fd):
        pass
    assert not fd.ran("docker", "pull") and not fd.ran("docker", "rmi")
    print("OK present image: left alone")


def test_uses_image_python_when_it_can_run_probe():
    fd = FakeDocker(present=True, image_python_ok=True)
    with ContainerSession("reg/x:t", runner=fd) as insp:
        assert insp.python == "python3", "should use the image's python"
    assert not any(
        c[-1].endswith(":/tmp/pyroot") for c in fd.calls if c[:2] == ["docker", "cp"]
    )
    print("OK image python used when it can run the probe")


def test_default_run_is_sealed():
    fd = FakeDocker(present=True)
    with ContainerSession("reg/x:t", runner=fd):
        pass
    runs = [c for c in fd.calls if c[:2] == ["docker", "run"]]
    assert all(
        "--network=none" in " ".join(c) and "--cap-drop=ALL" in " ".join(c)
        for c in runs
    )
    print("OK default run is sealed (no network, caps dropped)")


def test_bundles_python_when_image_python_cannot_run_probe():
    # covers missing, too-old, and incompatible (e.g. 3.9 vs pyelftools `match`)
    _no_download()
    fd = FakeDocker(present=True, image_python_ok=False)
    with ContainerSession("reg/pytorch:t", runner=fd) as insp:
        assert insp.python == "/tmp/pyroot/bin/python3", insp.python
        insp.inspect_elf("/opt/app/bin")
    assert any(
        c[:2] == ["docker", "cp"] and c[-1].endswith(":/tmp/pyroot") for c in fd.calls
    ), "should copy the bundled python in"
    probe = [c for c in fd.calls if "-m" in c and "secretary.container.probe" in c]
    assert (
        probe and "/tmp/pyroot/bin/python3" in probe[0]
    ), "probe must run under the bundled python"
    print("OK image python can't run probe -> bundled python used")


def test_skip_when_unusable_and_bundling_disabled():
    fd = FakeDocker(present=True, image_python_ok=False)
    try:
        with ContainerSession("reg/distroless:t", runner=fd, install_python=False):
            assert False, "should skip when bundling disabled"
    except SkipContainer as e:
        assert "no usable python3" in str(e)
        assert not any(
            c[-1].endswith(":/tmp/pyroot")
            for c in fd.calls
            if c[:2] == ["docker", "cp"]
        )
    print("OK unusable image python + bundling disabled -> skip")


def test_remote_inspector_parses_exec_json():
    fd = FakeDocker(present=True)
    with ContainerSession("reg/x:t", runner=fd) as insp:
        info = insp.inspect_elf("/opt/app/bin")
        assert info["arch"] == "arm64" and info["needed"] == ["libc.so.6"]
    probe = [c for c in fd.calls if "secretary.container.probe" in c]
    assert probe and "PYTHONPATH=/tmp" in probe[0]
    print("OK RemoteInspector parses exec JSON; probe run with PYTHONPATH=/tmp")


if __name__ == "__main__":
    for fn in [
        test_absent_image_pulled_probe_copied_then_reaped,
        test_present_image_not_reaped,
        test_uses_image_python_when_it_can_run_probe,
        test_default_run_is_sealed,
        test_bundles_python_when_image_python_cannot_run_probe,
        test_skip_when_unusable_and_bundling_disabled,
        test_remote_inspector_parses_exec_json,
    ]:
        fn()
    print("\nall container tests passed")
