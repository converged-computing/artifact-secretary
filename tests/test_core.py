import os, tempfile, pathlib, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary import Target, Inspector, derive_capability, Artifact
from secretary.inspection.target import OutsideTargetError

def test_inspect_real_elf():
    t = Target("/")
    insp = Inspector(t)
    info = insp.inspect_elf("/bin/ls")
    assert info["is_elf"] and info["arch"] in ("amd64","arm64"), info
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
        sub = pathlib.Path(d)/"root"; sub.mkdir()
        (sub/"ok.txt").write_text("hi")
        t = Target(sub)
        assert t.resolve("/ok.txt").name == "ok.txt"
        try:
            t.resolve("../../etc/passwd"); assert False, "should have blocked escape"
        except OutsideTargetError:
            print("OK containment blocks escape")

def test_capability_derivation():
    cap = derive_capability(
        ["libcudart.so.12", "libfabric.so.1", "libefa.so.1", "libmpi.so.40", "libc.so.6"])
    assert cap.accelerator == "cuda", cap
    assert cap.fabric_libfabric and cap.fabric_efa
    assert cap.mpi == "openmpi", cap
    # ROCm case
    cap2 = derive_capability(["libamdhip64.so.5", "libmpich.so.12"])
    assert cap2.accelerator == "rocm" and cap2.mpi == "mpich"
    print("OK capability derivation (cuda/efa/openmpi ; rocm/mpich)")

def test_provenance_cmake_kokkos():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root/"build").mkdir()
        (root/"build"/"CMakeCache.txt").write_text(
            "CMAKE_CXX_COMPILER:FILEPATH=/opt/nvidia/nvcc_wrapper\n"
            "CMAKE_CXX_FLAGS:STRING=-O3 -fopenmp\n"
            "Kokkos_ENABLE_CUDA:BOOL=ON\n"
            "PKG_REAXFF:BOOL=ON\n")
        prov = Inspector(Target(root)).detect_provenance("/build")
        assert prov["build_system"] == "cmake", prov
        assert "nvcc_wrapper" in prov["compiler"], prov
        assert any("CUDA" in f for f in prov["flags"]), prov["flags"]
        print("OK provenance cmake+kokkos-cuda:", prov["compiler"], prov["flags"][:3])

def test_artifact_to_requires():
    a = Artifact(application="lammps", binary="/opt/lammps/build/lmp",
                 arch="arm64", capability=derive_capability(["libcudart.so.12","libfabric.so.1","libefa.so.1","libmpi.so.40"]))
    req = a.to_requires()
    assert req == {"arch":"arm64","gpu":"cuda","fabric":"efa","mpi":"openmpi"}, req
    print("OK to_requires:", req)

if __name__ == "__main__":
    for fn in [test_inspect_real_elf,test_find_elf,test_containment,test_capability_derivation,test_provenance_cmake_kokkos,test_artifact_to_requires]:
        fn()
    print("\nall core tests passed")
