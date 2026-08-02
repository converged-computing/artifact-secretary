"""The container's libc decides which flux view can be mounted into it."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from secretary.inspection.inspector import Inspector
from secretary.inspection.target import Target
from secretary.model.manifest import LookupEntry, Platform, Reproduce


def test_platform_reports_libc_and_os():
    p = Inspector(Target("/")).platform()
    assert p["libc_flavor"] and p["libc_version"], p
    # a version we can compare, not a marketing string
    major, _, minor = p["libc_version"].partition(".")
    assert major.isdigit() and minor.split(".")[0].isdigit(), p
    print("OK platform:", p)


def test_platform_travels_on_the_entry():
    e = LookupEntry(
        reproduce=Reproduce(reference="x"),
        platform=Platform(libc_flavor="glibc", libc_version="2.35", os_id="ubuntu"),
    )
    assert e.platform.libc_version == "2.35"
    print("OK platform recorded on the manifest entry")


def test_static_cuda_is_detected_from_device_code():
    """A statically linked runtime leaves no library to find.

    Kripke links libcudart_static.a: no DT_NEEDED for libcuda, but the binary
    carries .nv_fatbin and 85 CUDA symbols. Dynamic linkage alone reports it as
    CPU only, which silently drops the GPU requirement from its jobspec.
    """
    from secretary.model.artifact import derive_capability

    cap = derive_capability(["libmpi.so.40"], None, [".nv_fatbin", ".nvFatBinSegment"])
    assert cap.accelerator == "cuda", cap
    assert any("static" in x for x in cap.gpu_libs), cap
    assert cap.mpi == "openmpi", cap

    # dynamic linkage still wins, and reports the real library
    cap = derive_capability(["libcuda.so.1"], None, [".nv_fatbin"])
    assert cap.accelerator == "cuda" and cap.gpu_libs == ["libcuda.so.1"], cap

    # hip too
    assert derive_capability([], None, [".hip_fatbin"]).accelerator == "rocm"

    # and a genuinely CPU build stays CPU
    assert derive_capability(["libmpi.so.40"], None, []).accelerator == "none"
    print("OK static device code detected")


if __name__ == "__main__":
    test_platform_reports_libc_and_os()
    test_platform_travels_on_the_entry()
    test_static_cuda_is_detected_from_device_code()
    print("\nplatform tests passed")
