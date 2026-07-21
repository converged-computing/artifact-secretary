"""The Artifact: a compiled application discovered on a Target, characterized
into what it IS and what it can RUN ON.

Unlike a resource "provider" (which detects an enumerable manager like spack or
conda), an artifact is idiosyncratic — it may have been hand-compiled somewhere
with no manager to ask. So we characterize it directly from its own footprint:
the binary, what it links against, and whatever build residue we can find.

Capability is derived deterministically from linkage (libcudart => needs a GPU,
libfabric+efa => wants EFA). Provenance (how it was built) is softer — inferred
from build files/strings — so it carries evidence and is clearly separable from
the hard capability facts.

A single source tree often yields MULTIPLE builds (LAMMPS: a KOKKOS/OpenMP CPU
variant, a CUDA variant, a ROCm variant), so an Artifact holds a list of
capability VARIANTS, not one answer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Capability:
    accelerator: str = "none"  # none|cuda|rocm
    gpu_libs: list[str] = field(default_factory=list)
    fabric_libfabric: bool = False
    fabric_efa: bool = False
    fabric_verbs: bool = False
    mpi: str = "none"  # none|openmpi|mpich|spectrum|intelmpi
    mpi_libs: list[str] = field(default_factory=list)


@dataclass
class Provenance:
    build_system: str = "unknown"  # cmake|autotools|make|ninja|spack|conda|unknown
    compiler: str = ""  # e.g. gcc, nvcc, clang, cray
    compiler_version: str = ""
    flags: list[str] = field(default_factory=list)
    evidence: list[str] = field(
        default_factory=list
    )  # files/strings that support the above


@dataclass
class Variant:
    """One hardware target the artifact was built to exploit."""

    arch: str
    accelerator: str = "none"
    fabric: str = "none"  # none|efa|verbs|libfabric
    note: str = ""


@dataclass
class Artifact:
    application: str  # e.g. "lammps"
    binary: str  # path (target-relative)
    arch: str = "unknown"
    interpreter: str = ""
    needed: list[str] = field(default_factory=list)
    rpath: list[str] = field(default_factory=list)
    runpath: list[str] = field(default_factory=list)
    capability: Capability = field(default_factory=Capability)
    provenance: Provenance = field(default_factory=Provenance)
    variants: list[Variant] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(
        default_factory=dict
    )  # field -> supporting paths
    confidence: str = "medium"  # low|medium|high

    def to_requires(self) -> dict:
        """Project the artifact into a fleetq `requires` block (hardware the
        matcher must satisfy). Capability facts only — never a cluster name."""
        req: dict = {"arch": self.arch}
        if self.capability.accelerator != "none":
            req["gpu"] = self.capability.accelerator
        if self.capability.fabric_efa:
            req["fabric"] = "efa"
        elif self.capability.fabric_verbs:
            req["fabric"] = "verbs"
        elif self.capability.fabric_libfabric:
            req["fabric"] = "libfabric"
        if self.capability.mpi != "none":
            req["mpi"] = self.capability.mpi
        return req

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --- deterministic capability derivation from linkage -------------------------

_GPU_CUDA = ("libcudart", "libcuda.so", "libcublas", "libcudnn", "libnccl")
_GPU_ROCM = ("libamdhip64", "librocm", "libhsa-runtime", "librocblas", "librccl")
_MPI = {
    "openmpi": ("libmpi.so", "libopen-rte", "libopen-pal"),
    "mpich": ("libmpich", "libmpifort"),
    "spectrum": ("libmpi_ibm",),
}


def derive_capability(
    needed: list[str], rpath_libs: list[str] | None = None
) -> Capability:
    """Map linked libraries to hardware capability. Deterministic: the same
    inputs always give the same Capability."""
    libs = [x.lower() for x in list(needed) + list(rpath_libs or [])]
    cap = Capability()

    gpu_hits = [l for l in libs if any(l.startswith(p) for p in _GPU_CUDA)]
    rocm_hits = [l for l in libs if any(l.startswith(p) for p in _GPU_ROCM)]
    if gpu_hits:
        cap.accelerator = "cuda"
        cap.gpu_libs = sorted(set(gpu_hits))
    elif rocm_hits:
        cap.accelerator = "rocm"
        cap.gpu_libs = sorted(set(rocm_hits))

    for l in libs:
        if l.startswith("libfabric.so"):
            cap.fabric_libfabric = True
        if l.startswith("libefa"):
            cap.fabric_efa = True
        if l.startswith("libibverbs") or l.startswith("librdmacm"):
            cap.fabric_verbs = True

    for flavor, prefixes in _MPI.items():
        matched = [l for l in libs if any(l.startswith(p) for p in prefixes)]
        if matched:
            cap.mpi = flavor
            cap.mpi_libs = sorted(set(matched))
            break
    return cap
