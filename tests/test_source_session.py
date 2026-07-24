"""The container sandbox path.

It had no coverage and its first real run died deep in subprocess because an unset
base image reached docker as None. These pin the contract without needing docker.
A default base, a guard against an empty one, a real git check, and the clone
actually running inside the container.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretary.source.session import (
    DEFAULT_BASE_IMAGE,
    ContainerSourceSession,
    SkipRepo,
    SourceSession,
    looks_like_url,
)


class FakeDocker:
    """A docker CLI stand-in. has_git=False simulates a base without git."""

    def __init__(self, has_git=True, clone_ok=True, revparse_ok=True):
        self.has_git, self.clone_ok = has_git, clone_ok
        self.revparse_ok = revparse_ok
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        cp = lambda rc=0, out="", err="": subprocess.CompletedProcess(
            argv, rc, out, err
        )
        a = argv
        # every argv element has to be a real string. a None here is the bug that
        # blew up inside subprocess with an unreadable type error
        for part in a:
            assert isinstance(part, str), f"non-str in argv: {a!r}"
        if a[:3] == ["docker", "image", "inspect"]:
            return cp(0)  # image already present, no pull
        if a[:3] == ["docker", "inspect", "--format"]:
            return cp(0, out=a[-1] + "@sha256:cafef00d\n")
        if a[:2] == ["docker", "run"]:
            return cp(0, out="sandbox123\n")
        if a[:2] == ["docker", "cp"]:
            return cp(0)
        if a[:2] == ["docker", "exec"]:
            rest = a[2:]
            if "git" in rest and "--version" in rest:
                return cp(0, out="git version 2.39.5\n") if self.has_git else cp(1)
            if "clone" in rest:
                return cp(0) if self.clone_ok else cp(1, err="fatal: repo not found")
            if "rev-parse" in rest or "log" in rest or "show-ref" in rest:
                if not self.revparse_ok:
                    # the real failure, git echoes the word HEAD back on stdout
                    return cp(
                        128,
                        out="HEAD\n",
                        err="fatal: ambiguous argument 'HEAD': unknown revision",
                    )
                return cp(0, out="abc1234\n")
            if "-c" in rest and any("probe" in x for x in rest):
                return cp(0)  # image python can run the probe
            if "uname" in rest:
                return cp(0, out="x86_64\n")
            return cp(0)
        if a[:3] == ["docker", "rm", "-f"]:
            return cp(0)
        return cp(0)

    def execs(self):
        return [a[2:] for a in self.calls if a[:2] == ["docker", "exec"]]


def test_default_base_image_is_used_when_unset():
    """An unset base must not reach docker as None."""
    for base in (None, ""):
        sess = ContainerSourceSession(
            "https://github.com/lammps/lammps", base_image=base, runner=FakeDocker()
        )
        assert sess.base_image == DEFAULT_BASE_IMAGE, base


def test_base_image_default_is_declared():
    sess = ContainerSourceSession(
        "https://github.com/lammps/lammps", runner=FakeDocker()
    )
    assert sess.base_image == DEFAULT_BASE_IMAGE


def test_clone_happens_inside_the_container():
    fake = FakeDocker()
    sess = ContainerSourceSession(
        "https://github.com/lammps/lammps", runner=fake, ref="stable"
    )
    with sess as backend:
        assert backend is not None
        # the clone is a docker exec, not a host command
        clones = [e for e in fake.execs() if "clone" in e]
        assert len(clones) == 1, fake.execs()
        assert "--branch" in clones[0] and "stable" in clones[0]
        assert sess.clone_dir in clones[0]
        assert sess.commit == "abc1234"
        assert sess.root == sess.clone_dir  # reader addresses the clone, not /
    assert ["docker", "rm", "-f", "sandbox123"] in fake.calls  # sandbox torn down


def test_missing_git_in_base_is_a_clear_skip():
    fake = FakeDocker(has_git=False)
    sess = ContainerSourceSession("https://github.com/lammps/lammps", runner=fake)
    try:
        with sess:
            assert False, "should have raised SkipRepo"
    except SkipRepo as e:
        assert "no git" in str(e)
        assert DEFAULT_BASE_IMAGE in str(e)
    # and it must not leave the sandbox running
    assert ["docker", "rm", "-f", "sandbox123"] in fake.calls


def test_failed_clone_reports_and_tears_down():
    fake = FakeDocker(clone_ok=False)
    sess = ContainerSourceSession("https://github.com/nope/nope", runner=fake)
    try:
        with sess:
            assert False, "should have raised SkipRepo"
    except SkipRepo as e:
        assert "clone failed" in str(e)
    assert ["docker", "rm", "-f", "sandbox123"] in fake.calls


def test_existing_path_in_container_is_not_cloned():
    fake = FakeDocker()
    sess = ContainerSourceSession("/opt/lammps", runner=fake)
    assert sess.will_clone is False
    with sess:
        assert not [e for e in fake.execs() if "clone" in e], "must not clone a path"
        assert sess.root == "/opt/lammps"


def test_host_session_locates_existing_tree(tmp_path):
    (tmp_path / "src").mkdir()
    sess = SourceSession(str(tmp_path))
    assert sess.will_clone is False
    with sess as insp:
        assert sess.root == "/"  # host Target is the tree itself
        assert any(e["name"] == "src" for e in insp.list_dir("/"))


def test_host_session_rejects_missing_path():
    try:
        with SourceSession("/definitely/not/here"):
            assert False, "should have raised SkipRepo"
    except SkipRepo as e:
        assert "no such directory" in str(e)


def test_looks_like_url():
    assert looks_like_url("https://github.com/lammps/lammps")
    assert looks_like_url("git@github.com:lammps/lammps.git")
    assert not looks_like_url("/opt/lammps")
    assert not looks_like_url("./lammps")


def test_commit_is_pinned_and_tree_marked_safe():
    """The revision has to be captured and the tree marked safe first, since git
    refuses trees it thinks have dubious ownership."""
    fake = FakeDocker()
    sess = ContainerSourceSession("https://github.com/lammps/lammps", runner=fake)
    with sess:
        assert sess.commit == "abc1234", sess.commit
        assert sess.commit_error == ""
    # safe.directory goes in inline with -c, which needs no HOME. git config
    # global fails outright without one, which happens under docker exec
    safe = [e for e in fake.execs() if any("safe.directory" in x for x in e)]
    assert safe, "should pass safe.directory inline"
    assert any(sess.clone_dir in x for x in safe[0])


def test_failed_rev_parse_is_reported_not_swallowed():
    """A failed pin must surface a reason instead of an empty commit."""
    fake = FakeDocker(revparse_ok=False)
    notes = []
    sess = ContainerSourceSession(
        "https://github.com/lammps/lammps", runner=fake, on_progress=notes.append
    )
    with sess:
        assert sess.commit == ""
        # the report has to say what was tried and what git said, not just failed
        assert "rev-parse" in sess.commit_error
        assert "ambiguous argument" in sess.commit_error
        assert "log" in sess.commit_error, "should have tried more than one strategy"
    assert any("could not resolve commit" in n for n in notes), notes
