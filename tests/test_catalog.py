import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import secretary.catalog as catalog

_PAGE = '''
<a href="/orgs/converged-computing/packages/container/package/metric-lammps-cpu">x</a>
<a href="/orgs/converged-computing/packages/container/package/rdma-infiniband">y</a>
<a href="/orgs/converged-computing/packages/container/package/metrics-operator-experiments%2Fperformance">z</a>
'''

def test_list_packages_parses_html(monkeypatch=None):
    catalog._text = lambda url: _PAGE if "page=1" in url else ""  # one page then empty
    names = catalog.list_packages("converged-computing", "performance-study")
    assert names == ["metric-lammps-cpu", "metrics-operator-experiments/performance", "rdma-infiniband"], names
    print("OK list_packages parses names (incl. slashed package)")


def test_arch_filter_drops_arm():
    catalog._text = lambda url: _PAGE if "page=1" in url else ""
    # fake registry: two tags, zen4=amd64, hpc7g=arm64
    catalog._registry_token = lambda repo_path: "tok"
    catalog._tags = lambda repo_path, token: ["zen4", "hpc7g"]
    catalog.tag_arches = lambda repo_path, tag, token: {
        "zen4": ["linux/amd64"], "hpc7g": ["linux/arm64"]}[tag]
    refs = catalog.list_ghcr("converged-computing", "performance-study", arch="amd64")
    assert all(":zen4" in r for r in refs), refs
    assert not any("hpc7g" in r for r in refs), "arm64 tag should have been dropped"
    assert any("metric-lammps-cpu:zen4" in r for r in refs)
    print("OK arch filter keeps amd64, drops arm64 (by real arch, not tag name)")


if __name__ == "__main__":
    test_list_packages_parses_html()
    test_arch_filter_drops_arm()
    print("all catalog tests passed")
