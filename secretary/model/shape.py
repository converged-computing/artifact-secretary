"""The ScheduleShape: how a source repository WANTS TO RUN, characterized so a
scheduler can derive the resource shape it should place.

This is the source-level sibling of the Artifact. Where an Artifact reads a
*compiled* binary's linkage to say what hardware it CAN run on (a GPU exists,
this fabric is present), a ScheduleShape reads a *repository and its entrypoint*
to say what a run will DEMAND of a scheduler: is it distributed (MPI) and thus
in need of communication; does it use shared memory and thus need its ranks on
one node; does it pin cores and thus want the cache to itself.

The same discipline as the Artifact path holds: the FACTS are deterministic.
`derive_shape` maps a bag of evidence markers (which MPI calls appear, which
pragmas, which launcher flags) to a closed vocabulary, the same way
`derive_capability` maps linked libraries to a Capability. The agent supplies
only *where* to look and the soft judgments (communication intensity) that no
regex can settle -- and it carries those with evidence and a confidence.

The closed vocabularies below ARE the grammar's terminals. Keeping them
enumerable is what lets a downstream scheduler consume a shape deterministically
rather than parsing prose. `to_shape_hint()` is the projection toward that
consumer -- the analog of Artifact.to_requires() -- and, like it, names hardware
shape only, never a cluster or a jobspec.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

# --- the marker library: evidence -> category (the deterministic ground truth)-
#
# Each entry is a list of regexes we grep the source/entrypoint tree for. A hit
# is a fact ("MPI_Alltoall appears at src/comm.c:412"); the interpretation of
# that fact into the grammar is done, deterministically, in derive_shape.

PARALLELISM_MARKERS: dict[str, list[str]] = {
    "mpi": [r"\bMPI_Init\b", r"#include\s*<mpi\.h>", r"\buse\s+mpi\b", r"\bmpi4py\b"],
    "threaded": [
        r"#pragma\s+omp\b",
        r"\bomp_get_\w+",
        r"\bOMP_NUM_THREADS\b",
        r"\bpthread_create\b",
        r"\btbb::",
    ],
    "gpu": [
        r"__global__\b",
        r"\bcudaMalloc\b",
        r"\bhipMalloc\b",
        r"#include\s*<cuda_runtime",
        r"\bsycl::",
        r"#pragma\s+acc\b",
        r"\bKokkos::Cuda\b",
    ],
    "task": [
        r"\bimport\s+dask\b",
        r"\bray\.init\b",
        r"\bSparkContext\b",
        r"\bimport\s+multiprocessing\b",
        r"\blegion\b",
    ],
}

# MPI call families -> (comm.model, comm.pattern). Order matters: earlier, more
# specific families win when several match.
COMM_MARKERS: list[tuple[str, str, str]] = [
    (r"\bMPI_Alltoall\w*", "collective", "all_to_all"),
    (
        r"\bMPI_(All)?[Rr]educe\b|\bMPI_Allgather\w*|\bMPI_Scan\b",
        "collective",
        "reduction",
    ),
    (r"\bMPI_Bcast\b|\bMPI_Scatter\w*|\bMPI_Gather\w*", "collective", "broadcast"),
    (r"\bMPI_Neighbor_\w+", "collective", "neighbor"),
    (
        r"\bMPI_Put\b|\bMPI_Get\b|\bMPI_Win_(?!allocate_shared)\w+",
        "point_to_point",
        "rma",
    ),
    (
        r"\bMPI_I?[Ss]end\b|\bMPI_I?[Rr]ecv\b|\bMPI_Sendrecv\b",
        "point_to_point",
        "neighbor",
    ),
]

SHARED_MEM_MARKERS = [
    r"\bMPI_Win_allocate_shared\b",
    r"\bshm_open\b",
    r"\bshmget\b",
    r"/dev/shm",
    r"\bMAP_SHARED\b",
]
NUMA_MARKERS = [
    r"\bnumactl\b",
    r"\bhwloc\b",
    r"\bKMP_AFFINITY\b",
    r"--bind-to\b",
    r"--map-by\b",
    r"\bGOMP_CPU_AFFINITY\b",
    r"\bsched_setaffinity\b",
]
HUGEPAGE_MARKERS = [r"\bhugetlb\b", r"\bMADV_HUGEPAGE\b", r"\bHUGETLB\b"]

ACCEL_MARKERS: dict[str, list[str]] = {
    "cuda": [
        r"\bcudaMalloc\b",
        r"__global__\b",
        r"#include\s*<cuda_runtime",
        r"\bKokkos::Cuda\b",
    ],
    "rocm": [r"\bhipMalloc\b", r"#include\s*<hip/", r"\bKokkos::HIP\b"],
    "sycl": [r"\bsycl::", r"#include\s*<CL/sycl"],
    "opencl": [r"\bclCreateKernel\b", r"#include\s*<CL/cl\.h>"],
}
GPU_COMM_MARKERS: dict[str, list[str]] = {
    "nccl": [r"\bnccl\w*", r"\bncclAllReduce\b"],
    "rccl": [r"\brccl\w*"],
    "gpu_aware_mpi": [r"\bMPIX_Query_cuda\w*", r"cuda[_-]aware", r"rocm[_-]aware"],
}

LAUNCHER_MARKERS: dict[str, str] = {
    "mpirun": r"\bmpirun\b",
    "mpiexec": r"\bmpiexec\b",
    "srun": r"\bsrun\b",
    "flux": r"\bflux\s+(?:run|submit|mini)\b",
    "jsrun": r"\bjsrun\b",
    "aprun": r"\baprun\b",
    "torchrun": r"\btorchrun\b|torch\.distributed\.(?:run|launch)",
}
# launcher flag -> ScheduleShape.launch field. Parsed-or-null; never guessed.
LAUNCH_FLAGS: dict[str, str] = {
    "tasks": r"(?:-np|--ntasks)[= ](\d+)",
    "tasks_per_node": r"--ntasks-per-node[= ](\d+)",
    "cpus_per_task": r"--cpus-per-task[= ](\d+)",
    "gpus_per_task": r"--gpus-per-task[= ](\d+)",
    "nodes": r"(?:-N|--nodes)[= ](\d+)",
}

IO_MARKERS: dict[str, list[str]] = {
    "mpiio": [r"\bMPI_File_\w+"],
    "hdf5": [r"\bH5Fcreate\b", r"#include\s*<hdf5", r"\bH5Pset_fapl_mpio\b"],
    "adios": [r"\badios2\b", r"\badios_\w+"],
    "netcdf": [r"\bnc_create_par\b", r"\bpnetcdf\b"],
}
CHECKPOINT_MARKERS = [r"\bSCR_\w+", r"\bcheckpoint\b", r"\brestart\b", r"\.chkpt\b"]

# The flat pattern list handed to the scan primitive, plus a reverse index the
# host uses to categorize hits (so regexes run in the deterministic Inspector,
# whether local or in-container, and only the labeling happens here).
ALL_MARKER_PATTERNS: list[str] = sorted(
    {
        p
        for group in (
            *PARALLELISM_MARKERS.values(),
            [c[0] for c in COMM_MARKERS],
            SHARED_MEM_MARKERS,
            NUMA_MARKERS,
            HUGEPAGE_MARKERS,
            *ACCEL_MARKERS.values(),
            *GPU_COMM_MARKERS.values(),
            list(LAUNCHER_MARKERS.values()),
            *IO_MARKERS.values(),
            CHECKPOINT_MARKERS,
        )
        for p in group
    }
)


def categorize_hits(hits: dict[str, list]) -> dict[str, list[str]]:
    """Turn raw scan_tree output ({regex: [locations]}) into the category keys
    derive_shape consumes ({'mpi': [...], 'comm:all_to_all': [...], ...}). The
    mapping lives here so the vocabulary and its evidence rules stay together."""
    out: dict[str, list[str]] = {}

    def add(cat: str, locs: list) -> None:
        ev = out.setdefault(cat, [])
        for loc in locs:
            if isinstance(loc, dict):
                ev.append(f"{loc.get('path', '')}:{loc.get('line', '')}")
            else:
                ev.append(str(loc))

    for pat, locs in (hits or {}).items():
        if not locs:
            continue
        for kind, pats in PARALLELISM_MARKERS.items():
            if pat in pats:
                add(kind, locs)
        for regex, _model, pattern in COMM_MARKERS:
            if pat == regex:
                add(f"comm:{pattern}", locs)
        if pat in SHARED_MEM_MARKERS:
            add("shared_memory", locs)
        if pat in NUMA_MARKERS:
            add("numa", locs)
        if pat in HUGEPAGE_MARKERS:
            add("hugepages", locs)
        for kind, pats in ACCEL_MARKERS.items():
            if pat in pats:
                add(f"accel:{kind}", locs)
        for flavor, pats in GPU_COMM_MARKERS.items():
            if pat in pats:
                add(f"gpu_comm:{flavor}", locs)
        for flavor, regex in LAUNCHER_MARKERS.items():
            if pat == regex:
                add(f"launcher:{flavor}", locs)
        for fmt, pats in IO_MARKERS.items():
            if pat in pats:
                add(f"io:{fmt}", locs)
        if pat in CHECKPOINT_MARKERS:
            add("checkpoint", locs)
    return out


# --- the grammar (closed-vocabulary terminals) --------------------------------


@dataclass
class Communication:
    model: str = "none"  # none|point_to_point|collective|mixed
    pattern: str = "none"  # none|neighbor|all_to_all|reduction|broadcast|rma|unknown
    intensity: str = "unknown"  # none|low|moderate|high|unknown  (agent judgment)
    latency_sensitive: bool = False
    bandwidth_sensitive: bool = False


@dataclass
class Memory:
    shared_memory: bool = False  # ranks touch the same node's memory -> co-locate
    numa_sensitive: bool = False  # pins/binds -> placement & core exclusivity matter
    hugepages: bool = False
    cpu_exclusive: bool = False  # "don't share the cache" -> whole cores, no SMT peers


@dataclass
class Accelerator:
    kind: str = "none"  # none|cuda|rocm|sycl|opencl
    multi_gpu: bool = False
    gpu_comm: str = "none"  # none|nccl|rccl|gpu_aware_mpi
    gpus_per_process: Optional[int] = None


@dataclass
class Launch:
    launcher: str = "none"  # none|mpirun|mpiexec|srun|flux|jsrun|aprun|torchrun|custom
    nodes: Optional[int] = None
    tasks: Optional[int] = None
    tasks_per_node: Optional[int] = None
    cpus_per_task: Optional[int] = None
    gpus_per_task: Optional[int] = None
    threads_per_task: Optional[int] = None
    elastic: bool = False  # spawns/resizes at runtime (MPI_Comm_spawn, torch elastic)


@dataclass
class IO:
    parallel_io: str = "none"  # none|mpiio|hdf5|adios|netcdf
    checkpoint: bool = False
    scratch: bool = False


@dataclass
class Topology:
    """The derived scheduler-facing preference -- the 'shape' proper."""

    span: str = "either"  # single_node|multi_node|either
    placement: str = "any"  # any|pack|spread|neighbor
    co_location: str = "none"  # none|preferred|required


@dataclass
class ScheduleShape:
    parallelism: list[str] = field(
        default_factory=lambda: ["serial"]
    )  # subset of grammar
    communication: Communication = field(default_factory=Communication)
    memory: Memory = field(default_factory=Memory)
    accelerator: Accelerator = field(default_factory=Accelerator)
    launch: Launch = field(default_factory=Launch)
    io: IO = field(default_factory=IO)
    topology: Topology = field(default_factory=Topology)
    confidence: str = "medium"  # low|medium|high
    evidence: dict[str, list[str]] = field(
        default_factory=dict
    )  # category -> file:line

    def to_shape_hint(self) -> dict:
        """Project the shape into a scheduler-facing hint (the grammar instance
        a resource matcher consumes). Shape only -- never a cluster or jobspec.
        The consumer is free to override; these are demands, not commands."""
        hint: dict = {
            "parallelism": self.parallelism,
            "span": self.topology.span,
            "placement": self.topology.placement,
            "co_location": self.topology.co_location,
        }
        if self.communication.model != "none":
            hint["communication"] = {
                "model": self.communication.model,
                "pattern": self.communication.pattern,
                "latency_sensitive": self.communication.latency_sensitive,
                "bandwidth_sensitive": self.communication.bandwidth_sensitive,
            }
        if self.accelerator.kind != "none":
            hint["gpu"] = self.accelerator.kind
            if self.accelerator.gpu_comm != "none":
                hint["gpu_comm"] = self.accelerator.gpu_comm
        if self.memory.shared_memory:
            hint["shared_memory"] = True
        if self.memory.cpu_exclusive:
            hint["cpu_exclusive"] = True
        for k in ("nodes", "tasks", "tasks_per_node", "cpus_per_task", "gpus_per_task"):
            v = getattr(self.launch, k)
            if v is not None:
                hint.setdefault("resources", {})[k] = v
        return hint

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# --- deterministic derivation (the derive_capability analog) ------------------


def _match_any(patterns: list[str], hay: str) -> bool:
    return any(re.search(p, hay) for p in patterns)


def derive_shape(
    markers: dict[str, list[str]],
    launch: Optional[dict] = None,
    env: Optional[dict] = None,
) -> ScheduleShape:
    """Map categorized evidence markers to a ScheduleShape. Deterministic: the
    same markers always give the same shape.

    `markers` is {category: [evidence strings]} as produced by
    scan_shape_markers (categories are the marker-library keys, e.g. 'mpi',
    'comm:all_to_all', 'shared_memory', 'numa', 'accel:cuda', 'gpu_comm:nccl',
    'launcher:srun', 'io:hdf5', 'checkpoint'). `launch`/`env` carry any
    parsed-or-null counts and environment the agent lifted from the entrypoint.
    """
    markers = markers or {}
    launch = launch or {}
    env = env or {}
    present = {k: v for k, v in markers.items() if v}
    shape = ScheduleShape(parallelism=[], evidence=present)

    # parallelism -------------------------------------------------------------
    for kind in ("mpi", "threaded", "gpu", "task"):
        if present.get(kind):
            shape.parallelism.append(kind)
    if not shape.parallelism:
        shape.parallelism = ["serial"]

    # communication (only meaningful when distributed) ------------------------
    comm = shape.communication
    models = set()
    # comm:* categories carry the (model, pattern) the marker mapped to
    for cat in present:
        if cat.startswith("comm:"):
            _, _, pat = cat.partition(":")
            for pattern_regex, model, pattern in COMM_MARKERS:
                if pattern == pat:
                    models.add(model)
                    if comm.pattern in ("none", pattern):
                        comm.pattern = pattern
                    break
    if models:
        comm.model = "mixed" if len(models) > 1 else next(iter(models))
    elif "mpi" in shape.parallelism:
        comm.model = comm.model or "point_to_point"
        comm.pattern = comm.pattern if comm.pattern != "none" else "unknown"
    comm.bandwidth_sensitive = comm.pattern in ("all_to_all", "reduction") or bool(
        present.get("gpu_comm:nccl") or present.get("gpu_comm:rccl")
    )
    comm.latency_sensitive = (
        comm.pattern in ("neighbor", "rma") or comm.model == "point_to_point"
    )

    # memory ------------------------------------------------------------------
    mem = shape.memory
    mem.shared_memory = bool(present.get("shared_memory"))
    mem.numa_sensitive = bool(present.get("numa"))
    mem.hugepages = bool(present.get("hugepages"))
    # pinning implies the app wants its cores (and their cache) to itself
    mem.cpu_exclusive = mem.numa_sensitive

    # accelerator -------------------------------------------------------------
    acc = shape.accelerator
    for kind in ("cuda", "rocm", "sycl", "opencl"):
        if present.get(f"accel:{kind}"):
            acc.kind = kind
            break
    for flavor in ("nccl", "rccl", "gpu_aware_mpi"):
        if present.get(f"gpu_comm:{flavor}"):
            acc.gpu_comm = flavor
            break
    if launch.get("gpus_per_task"):
        acc.gpus_per_process = int(launch["gpus_per_task"])
        acc.multi_gpu = acc.multi_gpu or int(launch["gpus_per_task"]) > 1
    acc.multi_gpu = acc.multi_gpu or acc.gpu_comm != "none"

    # launch ------------------------------------------------------------------
    lz = shape.launch
    for name in ("nodes", "tasks", "tasks_per_node", "cpus_per_task", "gpus_per_task"):
        if launch.get(name) is not None:
            setattr(lz, name, int(launch[name]))
    for flavor in LAUNCHER_MARKERS:
        if present.get(f"launcher:{flavor}"):
            lz.launcher = flavor
            break
    omp = env.get("OMP_NUM_THREADS")
    if omp and str(omp).isdigit():
        lz.threads_per_task = int(omp)

    # io ----------------------------------------------------------------------
    io = shape.io
    for fmt in ("mpiio", "hdf5", "adios", "netcdf"):
        if present.get(f"io:{fmt}"):
            io.parallel_io = fmt
            break
    io.checkpoint = bool(present.get("checkpoint"))

    # topology (the derived shape) --------------------------------------------
    topo = shape.topology
    if "mpi" in shape.parallelism and (lz.nodes is None or lz.nodes > 1):
        topo.span = "multi_node"
    elif set(shape.parallelism) <= {"serial", "threaded", "gpu"}:
        topo.span = "single_node"
    if mem.shared_memory:
        topo.co_location = "required"
        topo.span = "single_node" if lz.nodes in (None, 1) else topo.span
    elif comm.model != "none" or acc.gpu_comm != "none":
        topo.co_location = "preferred"
    if comm.pattern == "neighbor":
        topo.placement = "neighbor"
    elif mem.shared_memory or comm.bandwidth_sensitive or acc.gpu_comm != "none":
        topo.placement = "pack"
    elif "task" in shape.parallelism and comm.model == "none":
        topo.placement = "spread"

    return shape


# --- the result: a per-repo report and a commit-keyed lookup ------------------

SHAPE_SCHEMA_VERSION = "schedule-shape/v1"


@dataclass
class Entrypoint:
    """The observed launch surface -- evidence, not derivation."""

    command: str = ""  # resolved ENTRYPOINT/CMD or run-script command
    scripts: list[str] = field(default_factory=list)  # files that make it up
    env: dict[str, str] = field(default_factory=dict)  # relevant vars it sets/reads


@dataclass
class ShapeReport:
    repo: str  # url or path the shape was derived from
    commit: str = ""  # resolved commit, when cloned
    entrypoint: Entrypoint = field(default_factory=Entrypoint)
    shapes: list[ScheduleShape] = field(default_factory=list)  # >1 if multiple entries
    skipped: str = ""
    notes: str = ""

    def key(self) -> str:
        return f"{self.repo}@{self.commit}" if self.commit else self.repo


@dataclass
class ShapeLookup:
    version: str = SHAPE_SCHEMA_VERSION
    entries: dict[str, ShapeReport] = field(default_factory=dict)

    def add(self, report: ShapeReport) -> None:
        self.entries[report.key()] = report

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "entries": {k: asdict(v) for k, v in self.entries.items()},
            },
            indent=2,
            sort_keys=True,
        )
