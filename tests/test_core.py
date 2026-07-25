import os
import pathlib
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary import Artifact, Inspector, Target, derive_capability
from secretary.inspection.target import OutsideTargetError


def test_inspect_real_elf():
    t = Target("/")
    insp = Inspector(t)
    info = insp.inspect_elf("/bin/ls")
    assert info["is_elf"] and info["arch"] in ("amd64", "arm64"), info
    assert any(n.startswith("libc.so") for n in info["needed"]), info["needed"]
    assert info["interpreter"], "expected a PT_INTERP"
    print("OK inspect_elf /bin/ls:", info["arch"], info["needed"][:3])


def test_find_elf():
    t = Target("/")
    hits = Inspector(t).find("/bin", kind="elf", limit=5)
    assert hits, "expected some ELF binaries under /bin"
    print("OK find elf:", hits[:3])


def test_containment():
    with tempfile.TemporaryDirectory() as d:
        sub = pathlib.Path(d) / "root"
        sub.mkdir()
        (sub / "ok.txt").write_text("hi")
        t = Target(sub)
        assert t.resolve("/ok.txt").name == "ok.txt"
        try:
            t.resolve("../../etc/passwd")
            assert False, "should have blocked escape"
        except OutsideTargetError:
            print("OK containment blocks escape")


def test_capability_derivation():
    cap = derive_capability(
        [
            "libcudart.so.12",
            "libfabric.so.1",
            "libefa.so.1",
            "libmpi.so.40",
            "libc.so.6",
        ]
    )
    assert cap.accelerator == "cuda", cap
    assert cap.fabric_libfabric and cap.fabric_efa
    assert cap.mpi == "openmpi", cap
    # ROCm case
    cap2 = derive_capability(["libamdhip64.so.5", "libmpich.so.12"])
    assert cap2.accelerator == "rocm" and cap2.mpi == "mpich"
    print("OK capability derivation (cuda/efa/openmpi ; rocm/mpich)")


def test_provenance_cmake_general():
    # detector must be app-agnostic: read compiler + raw flags, not project keys
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "build").mkdir()
        (root / "build" / "CMakeCache.txt").write_text(
            "CMAKE_CXX_COMPILER:FILEPATH=/opt/nvidia/nvcc_wrapper\n"
            "CMAKE_CXX_FLAGS:STRING=-O3 -fopenmp\n"
            "Kokkos_ENABLE_CUDA:BOOL=ON\n"
        )
        prov = Inspector(Target(root)).detect_provenance("/build")
        assert prov["build_system"] == "cmake", prov
        assert "nvcc_wrapper" in prov["compiler"], prov
        assert "-fopenmp" in prov["flags"], prov["flags"]
        # no hard-coded project keys leaked into flags
        assert not any("Kokkos" in f for f in prov["flags"]), prov["flags"]
        print("OK provenance cmake (general): ", prov["compiler"], prov["flags"])


def test_artifact_to_requires():
    a = Artifact(
        application="lammps",
        binary="/opt/lammps/build/lmp",
        arch="arm64",
        capability=derive_capability(
            ["libcudart.so.12", "libfabric.so.1", "libefa.so.1", "libmpi.so.40"]
        ),
    )
    req = a.to_requires()
    assert req == {
        "arch": "arm64",
        "gpu": "cuda",
        "fabric": "efa",
        "mpi": "openmpi",
    }, req
    print("OK to_requires:", req)


if __name__ == "__main__":
    for fn in [
        test_inspect_real_elf,
        test_find_elf,
        test_containment,
        test_capability_derivation,
        test_provenance_cmake_general,
        test_artifact_to_requires,
    ]:
        fn()
    print("\nall core tests passed")


def test_save_tree_skips_ungeneratable_manifests():
    """A container we couldn't characterize (e.g. an arch we can't inspect)
    records a skip reason and no artifacts. save_tree must not emit a
    manifest.json for it, and it must not be counted among the written paths."""
    from secretary.model import LookupEntry, ManifestLookup, Reproduce

    lk = ManifestLookup()

    good = LookupEntry(
        reproduce=Reproduce(reference="ghcr.io/org/app:amd64", digest="sha256:aaa")
    )
    good.artifacts = [Artifact(application="lammps", binary="/opt/lammps/bin/lmp")]

    skipped = LookupEntry(
        reproduce=Reproduce(reference="ghcr.io/org/app:arm64", digest="sha256:bbb")
    )
    skipped.skipped = (
        "no usable python3 in image — could not fetch a standalone python (arm64)"
    )

    lk.add(good)
    lk.add(skipped)

    with tempfile.TemporaryDirectory() as d:
        paths = lk.save_tree(d)
        on_disk = sorted(str(p) for p in pathlib.Path(d).rglob("manifest.json"))
        # only the characterizable image yields a manifest
        assert len(paths) == 1, paths
        assert len(on_disk) == 1, on_disk
        assert "amd64" in paths[0] and "arm64" not in paths[0], paths
        print("OK save_tree omits ungeneratable manifest:", paths[0])
