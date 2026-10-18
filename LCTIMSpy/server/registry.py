"""The datasets the user has added, persisted across server restarts.

After a ``.d`` is added the browser refers to it only by ``dataset_id``. The id
is derived from the real path, so adding the same folder twice yields the same
dataset rather than a duplicate.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.config import ExtractionConfig
from ..core.io.bruker import read_dataset_meta
from ..core.io.cache import Cache, CacheEntry
from ..core.io.paths import is_network_path, validate_d_folder
from ..core.results import DatasetMeta


def dataset_id_for(path: Path) -> str:
    real = os.path.normcase(os.path.realpath(path))
    return hashlib.blake2b(real.encode(), digest_size=8).hexdigest()


@dataclass
class Dataset:
    id: str
    path: str
    name: str
    meta: DatasetMeta
    group: str | None = None
    local_path: str | None = None          # set after a copy-to-local-scratch job
    added: float = field(default_factory=time.time)

    @property
    def source(self) -> Path:
        """The folder extraction reads: the local copy if one exists."""
        if self.local_path and Path(self.local_path).exists():
            return Path(self.local_path)
        return Path(self.path)

    def to_json(self) -> dict:
        d = asdict(self)
        d["meta"] = self.meta.to_dict()
        return d

    @classmethod
    def from_json(cls, d: dict) -> Dataset:
        return cls(**{**d, "meta": DatasetMeta.from_dict(d["meta"])})


class DatasetRegistry:
    def __init__(self, state_dir: Path, cache: Cache):
        self.state_file = Path(state_dir) / "datasets.json"
        self.cache = cache
        self._lock = threading.RLock()
        self._datasets: dict[str, Dataset] = {}
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            self._datasets = {d["id"]: Dataset.from_json(d) for d in raw}
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, TypeError):
            # A corrupt state file costs the dataset list, not the server.
            self.state_file.replace(self.state_file.with_suffix(".corrupt.json"))

    def _save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps([d.to_json() for d in self._datasets.values()], indent=1),
                       encoding="utf-8")
        os.replace(tmp, self.state_file)

    # ---- CRUD ---------------------------------------------------------------

    def add(self, path: str, name: str | None = None, group: str | None = None) -> Dataset:
        d = validate_d_folder(path)
        meta = read_dataset_meta(d)
        ds_id = dataset_id_for(d)
        with self._lock:
            existing = self._datasets.get(ds_id)
            if existing is not None:
                existing.meta = meta
                if name:
                    existing.name = name
                if group is not None:
                    existing.group = group
                self._save()
                return existing
            ds = Dataset(id=ds_id, path=str(d), name=name or meta.name, meta=meta, group=group)
            self._datasets[ds_id] = ds
            self._save()
            return ds

    def get(self, ds_id: str) -> Dataset | None:
        return self._datasets.get(ds_id)

    def list(self) -> list[Dataset]:
        return sorted(self._datasets.values(), key=lambda d: d.added)

    def update(self, ds_id: str, **changes) -> Dataset | None:
        with self._lock:
            ds = self._datasets.get(ds_id)
            if ds is None:
                return None
            for key, value in changes.items():
                setattr(ds, key, value)
            self._save()
            return ds

    def remove(self, ds_id: str) -> bool:
        with self._lock:
            if self._datasets.pop(ds_id, None) is None:
                return False
            self._save()
            return True

    # ---- cache status -------------------------------------------------------

    def cache_entry(self, ds: Dataset, config: ExtractionConfig | None = None) -> CacheEntry | None:
        """The valid cache entry for this dataset, if it has been extracted."""
        config = config or ExtractionConfig()
        try:
            return self.cache.lookup(validate_d_folder(ds.source), config)
        except Exception:
            return None

    def describe(self, ds: Dataset) -> dict:
        entry = self.cache_entry(ds)
        return {
            "id": ds.id, "name": ds.name, "group": ds.group, "path": ds.path,
            "local_path": ds.local_path, "is_network": is_network_path(ds.path),
            "available": Path(ds.source).exists(), "meta": ds.meta.to_dict(),
            "cache": ({"status": "cached", "key": entry.key, "rows": entry.rows,
                       "bytes": entry.bytes, "created": entry.created,
                       "frames_failed": entry.stats.get("frames_failed", 0)}
                      if entry else {"status": "missing"}),
        }
