"""Saved discovery runs, one folder each (see ``pipeline``)."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path


class DiscoveryStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, run_id: str) -> Path:
        if not run_id.isalnum():
            raise KeyError(run_id)
        return self.root / run_id

    def list(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for d in self.root.iterdir():
            meta = d / "meta.json"
            if d.is_dir() and meta.is_file():
                try:
                    m = json.loads(meta.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                out.append(self._summary(d.name, m))
        return sorted(out, key=lambda r: r["created"] or 0, reverse=True)

    @staticmethod
    def _summary(run_id: str, m: dict) -> dict:
        return {"id": run_id, "created": m.get("created"), "dataset": m.get("dataset", {}),
                "counts": m.get("counts", {}), "psms": m.get("psms"),
                "n_features": m.get("n_features"), "notes": m.get("notes", [])}

    def meta(self, run_id: str) -> dict | None:
        p = self.path(run_id) / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def candidates(self, run_id: str) -> list[dict]:
        p = self.path(run_id) / "candidates.json"
        if not p.is_file():
            return []
        return _load(str(p), p.stat().st_mtime_ns)

    def candidate(self, run_id: str, candidate_id: int) -> dict | None:
        return next((c for c in self.candidates(run_id) if c["candidate_id"] == candidate_id),
                    None)

    def delete(self, run_id: str) -> bool:
        p = self.path(run_id)
        if not p.is_dir():
            return False
        shutil.rmtree(p, ignore_errors=True)
        _load.cache_clear()
        return True


@lru_cache(maxsize=8)
def _load(path: str, _mtime: int) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
