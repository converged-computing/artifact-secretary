import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary.container.session import ContainerSession, SkipContainer
from secretary.container.remote import RemoteInspector


class FakeDocker:
    def __init__(self, present=False, has_python=True):
        self.present, self.has_python = present, has_python
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        cp = lambda rc=0, out="", err="": subprocess.CompletedProcess(argv, rc, out, err)
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
        if a[:2] == ["docker", "exec"] and a[-1] == "--version":
            return cp(0 if self.has_python else 127, out="Python 3.11.2")
        if a[:2] == ["docker", "exec"]:  # in-container probe call
            return cp(0, out=json.dumps({"is_elf": True, "arch": "arm64", "needed": ["libc.so.6"]}))
        if a[:3] == ["docker", "rm", "-f"]:
            return cp(0)
        if a[:2] == ["docker", "rmi"]:
            return cp(0)
        return cp(1, err="unexpected " + " ".join(a))

    def ran(self, *prefix):
        return any(c[:len(prefix)] == list(prefix) for c in self.calls)


def test_absent_image_pulled_probe_copied_then_reaped():
    fd = FakeDocker(present=False)
    with ContainerSession("reg/lammps:tag", runner=fd) as insp:
        assert isinstance(insp, RemoteInspector)
    assert fd.ran("docker", "pull"), "should pull an absent image"
    assert fd.ran("docker", "cp"), "should copy the probe into the container"
    assert not any("-v" in c for c in fd.calls), "should NOT bind-mount (docker-outside-of-docker safe)"
    assert fd.ran("docker", "rmi"), "should reap an image we pulled"
    assert fd.ran("docker", "rm", "-f"), "should remove the container"
    # both the package and pyelftools get copied
    cps = [c for c in fd.calls if c[:2] == ["docker", "cp"]]
    dests = [c[-1] for c in cps]
    assert any(d.endswith(":/tmp/secretary") for d in dests), dests
    assert any(d.endswith(":/tmp/elftools") for d in dests), dests
    print("OK absent image: pulled + probe copied (no bind mount) + reaped")


def test_present_image_not_reaped():
    fd = FakeDocker(present=True)
    with ContainerSession("reg/lammps:tag", runner=fd):
        pass
    assert not fd.ran("docker", "pull") and not fd.ran("docker", "rmi"), "user's cache must be left alone"
    print("OK present image: not pulled, not reaped")


def test_keep_images_opt_out():
    fd = FakeDocker(present=False)
    with ContainerSession("reg/x:t", runner=fd, keep_images=True):
        pass
    assert fd.ran("docker", "pull") and not fd.ran("docker", "rmi"), "keep_images should skip reaping"
    print("OK keep_images: pulled but kept")


def test_skip_when_no_python():
    fd = FakeDocker(present=True, has_python=False)
    try:
        with ContainerSession("reg/distroless:t", runner=fd):
            assert False, "should have skipped (no python3)"
    except SkipContainer as e:
        assert "python3" in str(e)
        assert fd.ran("docker", "rm", "-f"), "must clean up the container on skip"
        assert not fd.ran("docker", "cp"), "must not copy the probe if we're skipping"
    print("OK no python3 -> SkipContainer + cleanup, no copy")


def test_remote_inspector_parses_exec_json():
    fd = FakeDocker(present=True)
    with ContainerSession("reg/x:t", runner=fd) as insp:
        info = insp.inspect_elf("/opt/lammps/lmp")
        assert info["arch"] == "arm64" and info["needed"] == ["libc.so.6"]
    # the probe is invoked with PYTHONPATH=/tmp and the module entrypoint
    exec_calls = [c for c in fd.calls if c[:2] == ["docker", "exec"] and "secretary.container.probe" in c]
    assert exec_calls and "PYTHONPATH=/tmp" in exec_calls[0]
    print("OK RemoteInspector parses exec JSON; probe invoked with PYTHONPATH=/tmp")


if __name__ == "__main__":
    for fn in [test_absent_image_pulled_probe_copied_then_reaped, test_present_image_not_reaped,
               test_keep_images_opt_out, test_skip_when_no_python, test_remote_inspector_parses_exec_json]:
        fn()
    print("all container tests passed")
