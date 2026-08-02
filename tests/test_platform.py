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


if __name__ == "__main__":
    test_platform_reports_libc_and_os()
    test_platform_travels_on_the_entry()
    print("\nplatform tests passed")
