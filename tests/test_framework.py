import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary import Inspector, ProfileTask, Target, ToolSpec, run_task
from secretary.framework.tools import inspection_tools
from secretary.model.manifest import ManifestLookup
from behalf.runner.sdk import _to_sdk_tool


class FakeSession:
    """Stands in for ContainerSession: yields an Inspector over a temp dir that
    contains a real ELF binary, and exposes digest/pulled_by_us."""

    def __init__(self, root, digest, pulled):
        self._root, self.digest, self.pulled_by_us = root, digest, pulled

    def __enter__(self):
        return Inspector(Target(self._root))

    def __exit__(self, *a):
        return False


class FakeRunner:
    def __init__(self, manifest):
        self.manifest = manifest

    async def converse(self, task):
        return self.manifest

    async def run_agent(self, system_prompt, user_prompt, tools, confirm_fn):
        byname = {t.name: t for t in tools}
        info = (await byname["inspect_elf"].handler({"path": "/lmp"}))["content"][0][
            "text"
        ]
        import json as _j

        needed = _j.loads(info)["needed"]
        await byname["record_artifact"].handler(
            {
                "application": "lammps",
                "binary": "/lmp",
                "arch": "amd64",
                "needed": needed,
                "provenance": {"build_system": "cmake"},
            }
        )


def test_profile_flow_direct_manifest():
    tmp = tempfile.mkdtemp()
    shutil.copy("/bin/ls", os.path.join(tmp, "lmp"))  # a real ELF to characterize
    factory = lambda ref, keep: FakeSession(tmp, "sha256:deadbeef", True)
    task = ProfileTask(session_factory=factory)
    manifest = {
        "catalog": ["myreg/lammps:latest"],
        "goal": "profile",
        "keep_images": False,
    }

    outcome = asyncio.run(run_task(task, FakeRunner(manifest), manifest=manifest))
    lk = outcome.result
    assert isinstance(lk, ManifestLookup)
    assert "sha256:deadbeef" in lk.entries, lk.entries.keys()
    e = lk.entries["sha256:deadbeef"]
    assert (
        e.reproduce.pulled_by_us
        and e.artifacts
        and e.artifacts[0].application == "lammps"
    )
    # round-trips through JSON
    import json

    reloaded = json.loads(lk.to_json())
    assert reloaded["version"].startswith("artifact-lookup/")
    print(
        "OK profile flow -> lookup keyed by digest, artifact recorded, JSON round-trips"
    )


def test_confirm_gate_on_action_tool():
    calls = []

    async def do(args):
        calls.append(args)
        return {"content": [{"type": "text", "text": "did it"}]}

    action = ToolSpec(
        "submit", "submit something", {"x": int}, do, kind="action", confirm=True
    )

    async def run(confirm_value):
        wrapped = _to_sdk_tool(action, lambda name, args: confirm_value)
        return await wrapped.handler({"x": 1})

    denied = asyncio.run(run(False))
    assert (
        "cancelled" in denied["content"][0]["text"] and not calls
    ), "action ran despite denial!"
    approved = asyncio.run(run(True))
    assert calls == [{"x": 1}], "approved action did not run"
    print("OK action tool gates on confirmation (deny blocks, approve runs)")


def test_setup_approval_can_abort():
    task = ProfileTask(session_factory=lambda r, k: None)
    task.requires_setup_approval = True  # simulate a task that needs approval
    outcome = asyncio.run(
        run_task(
            task,
            FakeRunner({"catalog": []}),
            manifest={"catalog": []},
            approve_fn=lambda m: False,
        )
    )
    assert outcome.approved is False and outcome.result is None
    print("OK setup approval can abort before execution")


if __name__ == "__main__":
    test_profile_flow_direct_manifest()
    test_confirm_gate_on_action_tool()
    test_setup_approval_can_abort()
    print("all framework tests passed")
