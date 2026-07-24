"""Embedded device code is the only GPU evidence a statically linked build has.

test_platform.py covers derive_capability, the pure end of that path. These
cover the two ends that actually touch a binary and the agent: the inspector
reporting the section, and record_artifact carrying it through.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretary.framework.tools import inspection_tools
from secretary.inspection.inspector import Inspector
from secretary.inspection.target import Target
from secretary.model.artifact import _DEVICE_SECTIONS


def _elf_with_sections(root, name, sections):
    """Copy a real ELF and append named sections. Returns the path, or None if
    objcopy is unavailable."""
    if not shutil.which("objcopy"):
        return None
    payload = os.path.join(root, "payload.bin")
    with open(payload, "wb") as fd:
        fd.write(b"device kernel image")
    src, dst = "/bin/ls", os.path.join(root, name)
    argv = ["objcopy"]
    for sec in sections:
        argv += [
            "--add-section",
            "%s=%s" % (sec, payload),
            "--set-section-flags",
            "%s=noload,readonly" % sec,
        ]
    argv += [src, dst]
    if subprocess.run(argv, capture_output=True).returncode != 0:
        return None
    return dst


def test_inspect_elf_reports_device_code():
    """The section has to survive the read, otherwise the agent never sees it."""
    with tempfile.TemporaryDirectory() as d:
        binary = _elf_with_sections(d, "kripke", [".nv_fatbin", ".nvFatBinSegment"])
        if binary is None:
            print("SKIP inspect_elf device code (no objcopy)")
            return
        info = Inspector(Target("/")).inspect_elf(binary)
        assert info["is_elf"], info
        assert info["device_code"] == [".nvFatBinSegment", ".nv_fatbin"], info
        # sorted and de-duplicated, so the manifest is stable across runs
        assert info["device_code"] == sorted(set(info["device_code"])), info
        print("OK inspect_elf reports device code:", info["device_code"])


def test_inspect_elf_reports_no_device_code_for_cpu_binary():
    """A plain CPU binary must not acquire a phantom GPU requirement."""
    info = Inspector(Target("/")).inspect_elf("/bin/ls")
    assert info["device_code"] == [], info
    print("OK plain binary reports no device code")


def test_record_artifact_carries_device_code():
    """The agent hands device_code back through the tool; a static CUDA build
    has no gpu library, so dropping it here loses the GPU requirement."""
    sink = []
    tools = {t.name: t for t in inspection_tools(object(), sink)}

    record = tools["record_artifact"]
    assert "device_code" in record.input_schema, record.input_schema

    asyncio.run(
        record.handler(
            {
                "application": "kripke",
                "binary": "/opt/kripke/build/kripke.exe",
                "arch": "amd64",
                "needed": ["libmpi.so.40", "libc.so.6"],
                "device_code": [".nv_fatbin"],
            }
        )
    )
    assert len(sink) == 1, sink
    cap = sink[0].capability
    assert cap.accelerator == "cuda", cap
    assert cap.mpi == "openmpi", cap
    assert sink[0].to_requires()["gpu"] == "cuda", sink[0].to_requires()
    print("OK record_artifact derives cuda from device code alone")


def test_record_artifact_without_device_code_stays_cpu():
    """Omitting the key entirely must behave the same as passing nothing."""
    sink = []
    tools = {t.name: t for t in inspection_tools(object(), sink)}
    asyncio.run(
        tools["record_artifact"].handler(
            {
                "application": "lammps",
                "binary": "/opt/lammps/build/lmp",
                "arch": "amd64",
                "needed": ["libmpi.so.40"],
            }
        )
    )
    assert sink[0].capability.accelerator == "none", sink[0].capability
    assert "gpu" not in sink[0].to_requires(), sink[0].to_requires()
    print("OK missing device_code stays cpu")


def test_every_known_section_maps_to_a_vendor():
    """The inspector collects on key membership and derive_capability resolves
    the value, so an entry with an unknown vendor would be silently ignored."""
    assert set(_DEVICE_SECTIONS.values()) == {"cuda", "rocm"}, _DEVICE_SECTIONS
    with tempfile.TemporaryDirectory() as d:
        for i, (section, vendor) in enumerate(sorted(_DEVICE_SECTIONS.items())):
            binary = _elf_with_sections(d, "bin%d" % i, [section])
            if binary is None:
                print("SKIP section sweep (no objcopy)")
                return
            info = Inspector(Target("/")).inspect_elf(binary)
            assert info["device_code"] == [section], (section, info)
    print("OK all", len(_DEVICE_SECTIONS), "sections read and map to a vendor")


if __name__ == "__main__":
    for fn in [
        test_inspect_elf_reports_device_code,
        test_inspect_elf_reports_no_device_code_for_cpu_binary,
        test_record_artifact_carries_device_code,
        test_record_artifact_without_device_code_stays_cpu,
        test_every_known_section_maps_to_a_vendor,
    ]:
        fn()
    print("\ndevice code tests passed")
