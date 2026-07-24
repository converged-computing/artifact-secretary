import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from secretary.inspection.inspector import Inspector
from secretary.inspection.target import Target
from secretary.model.shape import (
    ALL_MARKER_PATTERNS,
    categorize_hits,
    derive_shape,
)


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(text)


def test_scan_tree_and_categorize(tmp_path):
    root = str(tmp_path)
    _write(
        root,
        "src/solver.c",
        "#include <mpi.h>\nint main(){ MPI_Init(0,0); MPI_Alltoall(); }\n#pragma omp parallel\n",
    )
    _write(root, "src/shm.c", "int x = shm_open();\n")
    _write(root, "run.sh", "srun --ntasks 128 --cpus-per-task 4 ./solver\n")

    insp = Inspector(Target(root))
    raw = insp.scan_tree("/", ALL_MARKER_PATTERNS)
    cats = categorize_hits(raw)

    assert cats.get("mpi"), "should detect MPI"
    assert cats.get("threaded"), "should detect OpenMP"
    assert "comm:all_to_all" in cats
    assert cats.get("shared_memory")
    assert cats.get("launcher:srun")
    # evidence carries file:line, not just a boolean
    assert all(":" in e for e in cats["mpi"])


def test_derive_shape_all_to_all_is_bandwidth_bound():
    markers = {"mpi": ["x"], "comm:all_to_all": ["y"], "threaded": ["z"]}
    launch = {"tasks": 128, "cpus_per_task": 4}
    shape = derive_shape(markers, launch=launch)

    assert set(shape.parallelism) == {"mpi", "threaded"}
    assert shape.communication.model == "collective"
    assert shape.communication.pattern == "all_to_all"
    assert shape.communication.bandwidth_sensitive is True
    assert shape.topology.span == "multi_node"
    assert shape.topology.placement == "pack"
    assert shape.launch.tasks == 128
    assert shape.to_shape_hint()["resources"]["cpus_per_task"] == 4


def test_shared_memory_forces_co_location():
    shape = derive_shape({"mpi": ["x"], "shared_memory": ["y"]})
    assert shape.memory.shared_memory is True
    assert shape.topology.co_location == "required"
    assert shape.topology.span == "single_node"


def test_pinning_implies_cpu_exclusive():
    shape = derive_shape({"threaded": ["x"], "numa": ["--bind-to core"]})
    assert shape.memory.cpu_exclusive is True
    assert shape.topology.span == "single_node"


def test_neighbor_exchange_is_latency_bound():
    shape = derive_shape({"mpi": ["x"], "comm:neighbor": ["y"]})
    assert shape.communication.pattern == "neighbor"
    assert shape.communication.latency_sensitive is True
    assert shape.topology.placement == "neighbor"


def test_serial_default():
    shape = derive_shape({})
    assert shape.parallelism == ["serial"]
    assert shape.topology.span == "single_node"
