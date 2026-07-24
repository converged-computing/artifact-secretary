"""The traversal the shape agent navigates with.

Sits over any backend with the Inspector surface, local or in a container. Reads
are cached so a file is fetched once and re-reads are free, a token budget charges
only for new lines so a big repo cannot run away, and cursors let the agent keep
reading or jump elsewhere.

Token counting is a rough four characters each. It bounds work, it does not bill.
"""

from __future__ import annotations

from dataclasses import dataclass


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class TokenBudget:
    """A ceiling on source served into context over a whole walk. Charged once
    per line the first time it goes out, so re-reads are free."""

    limit: int = 60_000
    spent: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def charge(self, text: str) -> int:
        n = approx_tokens(text)
        self.spent += n
        return n


class SourceReader:
    def __init__(self, backend, budget: TokenBudget, root: str = "/", context: int = 8):
        self.backend = backend
        self.budget = budget
        self.root = root.rstrip("/") or "/"
        self.context = context  # default lines above/below a search hit
        self._files: dict[str, list[str]] = {}  # path -> lines (read once)
        self._served: dict[str, set[int]] = {}  # path -> line numbers already charged

    # --- paths ---------------------------------------------------------------
    # the clone sits at an absolute path in the container. that must not reach the
    # agent or the report, nobody can resolve it once the sandbox is gone

    def to_backend(self, path: str) -> str:
        """Turn a repo relative path into the one the backend wants."""
        p = (path or "").strip()
        if self.root == "/":
            return p if p.startswith("/") else "/" + p.lstrip("./")
        if p.startswith(self.root):
            return p
        return f"{self.root}/{p.lstrip('./').lstrip('/')}"

    def to_repo(self, path: str) -> str:
        """Turn a backend path back into the repo relative one we show."""
        p = (path or "").strip()
        if self.root != "/" and p.startswith(self.root):
            p = p[len(self.root) :]
        return p.lstrip("/")

    # --- cache ---------------------------------------------------------------

    def _load(self, path: str) -> list[str]:
        if path not in self._files:
            txt = self.backend.read_text(path)
            self._files[path] = (txt if isinstance(txt, str) else "").splitlines()
        return self._files[path]

    def _render_range(self, path: str, start: int, end: int) -> dict:
        """Serve a line range, charging only lines not served before and
        shrinking the window when the budget cannot fit the new ones."""
        backend_path = self.to_backend(path)
        shown = self.to_repo(backend_path)
        path = backend_path
        lines = self._load(path)
        total = len(lines)
        if total == 0:
            return {"path": shown, "empty": True, "total_lines": 0}
        start = max(1, start)
        end = min(total, max(start, end))
        served = self._served.setdefault(path, set())

        def new_text(s: int, e: int) -> str:
            return "\n".join(lines[n - 1] for n in range(s, e + 1) if n not in served)

        truncated = False
        if new_text(start, end).strip():  # there is unseen content in the window
            if self.budget.exhausted:
                # no budget left for new lines so serve only what was already seen
                visible = [n for n in range(start, end + 1) if n in served]
                if not visible:
                    return {
                        "path": shown,
                        "start": start,
                        "requested_end": end,
                        "total_lines": total,
                        "content": "",
                        "next": None,
                        "truncated": True,
                        "budget_exhausted": True,
                    }
                start, end = visible[0], visible[-1]
            else:
                # shrink end until the new lines fit the remaining budget
                while (
                    end > start
                    and approx_tokens(new_text(start, end)) > self.budget.remaining
                ):
                    end -= 1
                    truncated = True
                self.budget.charge(new_text(start, end))
                for n in range(start, end + 1):
                    served.add(n)

        body = "\n".join(f"{n:>6}\t{lines[n - 1]}" for n in range(start, end + 1))
        return {
            "path": shown,
            "start": start,
            "end": end,
            "total_lines": total,
            "content": body,
            "next": (end + 1 if end < total else None),
            "truncated": truncated,
            "budget": self._budget_view(),
        }

    def _budget_view(self) -> dict:
        b = self.budget
        return {
            "spent": b.spent,
            "limit": b.limit,
            "remaining": b.remaining,
            "exhausted": b.exhausted,
        }

    # --- the primitives the tools expose -------------------------------------

    def read(self, path: str, start: int = 1, count: int = 40) -> dict:
        """A line window plus a cursor. Call again with it to keep reading the
        same file. Charged once per line, so re-reads are free."""
        return self._render_range(path, start, start + max(1, count) - 1)

    def search(self, pattern: str, context: int | None = None, limit: int = 30) -> dict:
        """Find a regex the agent chose, not one from a fixed table, and give
        back each hit with surrounding lines and a cursor to expand. This is how
        it searches for a signature it saw called and then reads around it."""
        ctx = self.context if context is None else context
        raw = self.backend.scan_tree(self.root, [pattern], max_hits_per_pattern=limit)
        hits = []
        for loc in raw.get(pattern, []):
            path, line = loc.get("path", ""), loc.get("line", 0)
            win = self._render_range(path, line - ctx, line + ctx)
            hits.append(
                {
                    "path": self.to_repo(self.to_backend(path)),
                    "line": line,
                    "context": win.get("content", ""),
                    "read_more": win.get("next"),
                    "total_lines": win.get("total_lines"),
                }
            )
            if self.budget.exhausted:
                break
        return {"pattern": pattern, "hits": hits, "budget": self._budget_view()}

    def extract(
        self, path: str, start: int, end: int = 0, pad: int = 3, max_lines: int = 24
    ) -> list[tuple[int, str]]:
        """Lines around a span to store in the report. Not charged to the budget,
        this goes into the artifact and never into the model context. Recording it
        now is the only way a citation stays auditable once the sandbox is gone.
        """
        try:
            lines = self._load(self.to_backend(path))
        except Exception:
            return []
        if not lines or start < 1:
            return []
        lo = max(1, start - pad)
        hi = min(len(lines), (end or start) + pad)
        if hi < lo:
            return []
        if hi - lo + 1 > max_lines:
            hi = lo + max_lines - 1
        return [(n, lines[n - 1]) for n in range(lo, hi + 1)]

    def seen(self) -> dict:
        """What has been read this run, plus what is left of the budget. Doubles
        as the stop signal and, dumped at the end, as the run provenance."""
        files = {}
        for path, served in self._served.items():
            if served:
                lo, hi = min(served), max(served)
                files[self.to_repo(path)] = {
                    "lines_read": len(served),
                    "total_lines": len(self._files.get(path, [])),
                    "span": f"{lo}-{hi}",
                }
        return {"files": files, "budget": self._budget_view()}
