"""The deterministic inspection library: the trustworthy primitives the agent
navigates with. Reading an ELF's linkage, scanning strings, reading build files
are all deterministic — same input, same output — so the FACTS never depend on
the model. The agent supplies only judgment about *where* to look and *what it
means*; these functions supply the ground truth it reasons over.

All access goes through a Target, so every path is contained and read-only.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import asdict
from pathlib import Path

from elftools.elf.dynamic import DynamicSection
from elftools.elf.elffile import ELFFile

from ..model.artifact import _DEVICE_SECTIONS, Provenance
from .target import Target

_ARCH = {
    "x64": "amd64",
    "x86": "386",
    "AArch64": "arm64",
    "ARM": "arm",
    "64-bit PowerPC": "ppc64le",
    "RISC-V": "riscv64",
    "IBM S/390": "s390x",
}


def is_elf(p) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


class Inspector:
    """Read-only inspection over one Target. Methods are what the agent calls."""

    def __init__(self, target: Target):
        self.t = target

    # --- navigation ---------------------------------------------------------

    def platform(self) -> dict:
        """The container's libc and OS.

        This decides which flux view can be mounted into the image: the view's
        binaries link against the CONTAINER's libc, so a view built on a newer
        glibc will not load (`version GLIBC_2.38 not found`). It is a property of
        the image, like arch, so it belongs in the manifest rather than being
        guessed per cluster.
        """
        import os as _os

        out = {
            "libc_flavor": "",
            "libc_version": "",
            "os_id": "",
            "os_version_id": "",
            "os_codename": "",
        }
        try:
            # CS_GNU_LIBC_VERSION is "glibc 2.35". No subprocess, no ldd.
            flavor, _, version = (_os.confstr("CS_GNU_LIBC_VERSION") or "").partition(
                " "
            )
            out["libc_flavor"], out["libc_version"] = flavor, version
        except (ValueError, OSError):
            pass
        if not out["libc_version"]:
            try:
                import platform as _platform

                out["libc_flavor"], out["libc_version"] = _platform.libc_ver()
            except Exception:  # noqa: BLE001 - musl and friends report nothing
                pass
        try:
            for line in self.read_text("/etc/os-release", 64 * 1024).splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if k == "ID":
                    out["os_id"] = v
                elif k == "VERSION_ID":
                    out["os_version_id"] = v
                elif k == "VERSION_CODENAME":
                    out["os_codename"] = v
        except Exception:  # noqa: BLE001 - not every image has os-release
            pass
        return out

    def list_dir(self, path: str = "/") -> list[dict]:
        d = self.t.resolve(path)
        out = []
        if not d.is_dir():
            return out
        for e in sorted(d.iterdir(), key=lambda x: x.name):
            try:
                kind = "dir" if e.is_dir() else ("elf" if is_elf(e) else "file")
                size = e.stat().st_size if e.is_file() else 0
            except OSError:
                kind, size = "file", 0
            out.append(
                {"name": e.name, "kind": kind, "size": size, "path": self.t.display(e)}
            )
        return out

    def find(
        self,
        root: str = "/",
        name_glob: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[str]:
        """Walk under `root`, returning target-relative paths. kind: elf|file|dir."""
        base = self.t.resolve(root)
        hits: list[str] = []
        skip = {"/proc", "/sys", "/dev", "/run"}
        for cur, dirs, files in os.walk(base):
            if self.t.display(Path(cur)) in skip:
                dirs[:] = []
                continue
            names = (
                dirs
                if kind == "dir"
                else files if kind in ("file", "elf") else dirs + files
            )
            for n in names:
                if name_glob and not fnmatch.fnmatch(n, name_glob):
                    continue
                p = Path(cur) / n
                if kind == "elf" and not is_elf(p):
                    continue
                hits.append(self.t.display(p))
                if len(hits) >= limit:
                    return hits
        return hits

    # --- inspection ---------------------------------------------------------

    def inspect_elf(self, path: str) -> dict:
        """The core 'what is this compiled against' call: arch, interpreter, and
        the dynamic linkage (NEEDED libraries, RPATH/RUNPATH, SONAME)."""
        p = self.t.resolve(path)
        if not is_elf(p):
            return {"path": self.t.display(p), "is_elf": False}
        with open(p, "rb") as f:
            elf = ELFFile(f)
            arch = _ARCH.get(elf.get_machine_arch(), elf.get_machine_arch())
            interp = ""
            for seg in elf.iter_segments():
                if seg.header.p_type == "PT_INTERP":
                    interp = seg.get_interp_name()
                    break
            needed, rpath, runpath, soname = [], [], [], ""
            device_code = []
            for sec in elf.iter_sections():
                # Embedded device code. A CUDA or HIP build linked against the
                # static runtime has no DT_NEEDED for libcuda, so dynamic
                # linkage alone reports it as CPU only. The fatbin section is
                # the evidence in that case.
                if sec.name in _DEVICE_SECTIONS:
                    device_code.append(sec.name)
                if isinstance(sec, DynamicSection):
                    for tag in sec.iter_tags():
                        t = tag.entry.d_tag
                        if t == "DT_NEEDED":
                            needed.append(tag.needed)
                        elif t == "DT_RPATH":
                            rpath += [x for x in tag.rpath.split(":") if x]
                        elif t == "DT_RUNPATH":
                            runpath += [x for x in tag.runpath.split(":") if x]
                        elif t == "DT_SONAME":
                            soname = tag.soname
            return {
                "path": self.t.display(p),
                "is_elf": True,
                "arch": arch,
                "type": str(elf.header.e_type),
                "interpreter": interp,
                "needed": needed,
                "rpath": rpath,
                "runpath": runpath,
                "soname": soname,
                "device_code": sorted(set(device_code)),
            }

    def scan_strings(
        self,
        path: str,
        patterns: list[str],
        max_hits: int = 40,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> dict:
        """Grep printable strings in a binary/file for regexes (build flags,
        compiler banners, CUDA arch, package names)."""
        p = self.t.resolve(path)
        try:
            data = p.read_bytes()[:max_bytes]
        except OSError as e:
            return {"error": str(e)}
        text = data.decode("latin-1", "ignore")
        res: dict[str, list[str]] = {}
        for pat in patterns:
            rx = re.compile(pat)
            found = []
            for m in rx.finditer(text):
                s = m.group(0)
                if s not in found:
                    found.append(s)
                if len(found) >= max_hits:
                    break
            if found:
                res[pat] = found
        return res

    def scan_tree(
        self,
        root: str,
        patterns: list[str],
        max_hits_per_pattern: int = 25,
        max_file_bytes: int = 2 * 1024 * 1024,
    ) -> dict:
        """Grep text files under `root` for regexes, returning per-pattern hits
        with target-relative path + line + the matched text. Deterministic, and
        the source-tree analog of scan_strings (which reads one binary): the
        ground truth the shape agent reasons over. Binary/oversized files and
        the usual pseudo-filesystems are skipped so this stays cheap and read-only.
        """
        base = self.t.resolve(root)
        rxs = [(p, re.compile(p)) for p in patterns]
        out: dict[str, list[dict]] = {p: [] for p in patterns}
        skip_dirs = {".git", "node_modules", "__pycache__", ".spack"}
        for cur, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for n in files:
                p = Path(cur) / n
                try:
                    if p.stat().st_size > max_file_bytes or is_elf(p):
                        continue
                    text = p.read_bytes().decode("utf-8", "ignore")
                except OSError:
                    continue
                disp = self.t.display(p)
                for pat, rx in rxs:
                    if len(out[pat]) >= max_hits_per_pattern:
                        continue
                    for i, line in enumerate(text.splitlines(), 1):
                        if rx.search(line):
                            out[pat].append(
                                {"path": disp, "line": i, "text": line.strip()[:200]}
                            )
                            if len(out[pat]) >= max_hits_per_pattern:
                                break
        return {p: hits for p, hits in out.items() if hits}

    def read_text(self, path: str, max_bytes: int = 256 * 1024) -> str:
        p = self.t.resolve(path)
        try:
            return p.read_bytes()[:max_bytes].decode("utf-8", "replace")
        except OSError as e:
            return f"<error: {e}>"

    # --- provenance (deterministic detectors the agent can lean on) ---------

    BUILD_MARKERS = {
        "CMakeCache.txt": "cmake",
        "build.ninja": "ninja",
        "config.log": "autotools",
        "configure": "autotools",
        "Makefile": "make",
        "conda-meta": "conda",
    }

    def detect_provenance(self, dir_path: str) -> dict:
        """Inspect a build/install directory for how it was built. Returns a
        Provenance dict with evidence. Softer than linkage — flagged as such."""
        d = self.t.resolve(dir_path)
        prov = Provenance()
        if not d.is_dir():
            return asdict(prov)

        entries = {e.name: e for e in d.iterdir()} if d.is_dir() else {}
        # spack leaves a .spack dir or spack-build-* trees
        if any(n == ".spack" or n.startswith("spack-build") for n in entries):
            prov.build_system = "spack"
            prov.evidence.append(self.t.display(d) + " (spack build residue)")
        for marker, system in self.BUILD_MARKERS.items():
            if marker in entries:
                if prov.build_system in ("unknown",):
                    prov.build_system = system
                prov.evidence.append(self.t.display(entries[marker]))

        cc = entries.get("CMakeCache.txt")
        if cc:
            prov.build_system = "cmake"
            txt = cc.read_bytes()[: 512 * 1024].decode("utf-8", "replace")
            comp = re.search(r"CMAKE_CXX_COMPILER:\w+=(.+)", txt)
            if comp:
                prov.compiler = comp.group(1).strip()
            # raw compile flags only; app/package-specific interpretation is the
            # consumer's job, so we don't hard-code any project's cache keys here.
            flags = re.search(r"CMAKE_CXX_FLAGS:\w+=(.+)", txt)
            if flags and flags.group(1).strip():
                prov.flags += flags.group(1).split()
        return asdict(prov)


# convenience for tests / non-agent use
def inspect_binary(target: Target, path: str) -> dict:
    return Inspector(target).inspect_elf(path)
