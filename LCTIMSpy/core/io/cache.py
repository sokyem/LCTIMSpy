"""On-disk extraction cache: one mz-sorted Parquet file per (``.d``, ExtractionConfig).

Layout::

    <root>/<key>/meta.json          # authoritative: config, DatasetMeta, stats
    <root>/<key>/data.parquet       # mz-sorted, ZSTD, row-group statistics on
    <root>/<key>/frame_index.json   # frame -> retention_time, polarity
    <root>/<key>/last_used          # touched on every read; mtime drives LRU

The key hashes the source's identity (real path plus size/mtime of
``analysis.tdf`` and ``analysis.tdf_bin``) and the ExtractionConfig -- never the
target m/z list, so one extraction serves every later query.

Writes happen in ``<key>.tmp-*`` and are moved into place with ``os.replace``,
so a killed job never leaves a half-written entry that looks valid. Because SMB
mtimes are coarse, validation is a heuristic; callers should always offer a
"re-extract" that passes ``force=True``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExtractionConfig
from ..errors import CacheError
from ..results import DatasetMeta
from .paths import REQUIRED_FILES, cache_root

SCHEMA_VERSION = 1
# Bump when extraction output changes for the same input and config.
# 2: also caches DDA-PASEF precursors and MS/MS spectra.
EXTRACTOR_VERSION = 2

DATA_FILE = "data.parquet"
META_FILE = "meta.json"
FRAMES_FILE = "frame_index.json"
PRECURSORS_FILE = "precursors.parquet"
MSMS_FILE = "msms.parquet"
LAST_USED_FILE = "last_used"
HEARTBEAT_FILE = "heartbeat"
TMP_MARK = ".tmp-"
OLD_MARK = ".old-"

DEFAULT_MAX_BYTES = 50 * 1024**3

# Entries written before the mobility coordinate was recorded are Bruker TDF.
DEFAULT_MOBILITY = {"present": True, "coordinate": "1/K0", "unit": "V·s/cm²",
                    "representation": "TDF scan number via Bruker TDF-SDK", "binned": True}


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def source_fingerprint(d_path: Path) -> dict[str, list[int]]:
    """Size and mtime of the files whose change invalidates a cache entry."""
    if d_path.is_file():                        # mzML: the file itself
        st = d_path.stat()
        return {d_path.name: [st.st_size, st.st_mtime_ns]}
    out = {}
    for name in REQUIRED_FILES:
        st = (d_path / name).stat()
        out[name] = [st.st_size, st.st_mtime_ns]
    return out


def cache_key(d_path: Path, config: ExtractionConfig) -> str:
    identity = {
        "schema": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "source": os.path.normcase(os.path.realpath(d_path)),
        "fingerprint": source_fingerprint(d_path),
        "config": config.to_dict(),
    }
    return hashlib.blake2b(_canonical(identity).encode(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    path: Path
    dataset: DatasetMeta
    config: ExtractionConfig
    rows: int
    bytes: int
    created: float
    last_used: float
    stats: dict
    # Which mobility coordinate the cache holds (see extract.TDF_MOBILITY).
    mobility: dict = field(default_factory=lambda: dict(DEFAULT_MOBILITY))

    @property
    def data_path(self) -> Path:
        return self.path / DATA_FILE


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _rmtree(path: Path) -> None:
    # Windows refuses to delete files a reader still has mapped; whatever is
    # left is retried by sweep_partials().
    shutil.rmtree(path, ignore_errors=True)


class Cache:
    def __init__(self, root: str | os.PathLike | None = None, max_bytes: int | None = None):
        self.root = Path(root) if root is not None else cache_root()
        if max_bytes is None:
            env = os.environ.get("LCTIMSPY_CACHE_MAX_GB")
            max_bytes = int(float(env) * 1024**3) if env else DEFAULT_MAX_BYTES
        self.max_bytes = max_bytes

    # ---- lookup -------------------------------------------------------------

    def key_for(self, d_path: Path, config: ExtractionConfig) -> str:
        return cache_key(d_path, config)

    def _load_entry(self, path: Path) -> CacheEntry | None:
        meta_path, data_path = path / META_FILE, path / DATA_FILE
        if not (meta_path.is_file() and data_path.is_file() and (path / FRAMES_FILE).is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("schema") != SCHEMA_VERSION:
                return None
            last_used_path = path / LAST_USED_FILE
            last_used = (
                last_used_path.stat().st_mtime if last_used_path.exists() else meta["created"]
            )
            return CacheEntry(
                key=path.name,
                path=path,
                dataset=DatasetMeta.from_dict(meta["dataset"]),
                config=ExtractionConfig.from_dict(meta["config"]),
                rows=meta["rows"],
                bytes=meta["bytes"],
                created=meta["created"],
                last_used=last_used,
                stats=meta.get("stats", {}),
                mobility=meta.get("mobility", dict(DEFAULT_MOBILITY)),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def get(self, key: str, *, touch: bool = True) -> CacheEntry | None:
        """The entry for ``key``, or None if absent or unreadable."""
        path = self.root / key
        entry = self._load_entry(path)
        if entry is not None and touch:
            (path / LAST_USED_FILE).touch()
        return entry

    def lookup(self, d_path: Path, config: ExtractionConfig) -> CacheEntry | None:
        return self.get(self.key_for(d_path, config))

    def entries(self) -> list[CacheEntry]:
        if not self.root.is_dir():
            return []
        out = []
        for child in self.root.iterdir():
            if child.is_dir() and TMP_MARK not in child.name and OLD_MARK not in child.name:
                entry = self._load_entry(child)
                if entry is not None:
                    out.append(entry)
        return sorted(out, key=lambda e: e.last_used, reverse=True)

    def total_bytes(self) -> int:
        return sum(e.bytes for e in self.entries())

    # ---- writing ------------------------------------------------------------

    @contextmanager
    def writer(self, key: str) -> Iterator[Path]:
        """Yield a private temp dir; on clean exit it atomically becomes ``<root>/<key>``.

        Any exception -- including KeyboardInterrupt and CancelledError --
        deletes the temp dir, so no partial entry can survive.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / f"{key}{TMP_MARK}{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            yield tmp
            self._commit(tmp, self.root / key)
        except BaseException:
            _rmtree(tmp)
            raise

    def _commit(self, tmp: Path, final: Path) -> None:
        old = None
        if final.exists():
            old = final.with_name(f"{final.name}{OLD_MARK}{uuid.uuid4().hex[:8]}")
            try:
                os.replace(final, old)
            except OSError as exc:
                raise CacheError(
                    f"Cannot replace cache entry {final.name}; is it open in another process? {exc}"
                ) from exc
        os.replace(tmp, final)
        if old is not None:
            _rmtree(old)
        _frame_index.cache_clear()

    @staticmethod
    def heartbeat(tmp: Path) -> None:
        """Mark a temp dir as live; sweep_partials() spares fresh heartbeats."""
        (tmp / HEARTBEAT_FILE).touch()

    # ---- maintenance ----------------------------------------------------------

    def sweep_partials(self, max_age_s: float = 3600) -> list[Path]:
        """Delete temp dirs left by killed jobs, and superseded entries."""
        if not self.root.is_dir():
            return []
        removed = []
        now = time.time()
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            if OLD_MARK in child.name:
                _rmtree(child)
                removed.append(child)
            elif TMP_MARK in child.name:
                beat = child / HEARTBEAT_FILE
                stamp = beat.stat().st_mtime if beat.exists() else child.stat().st_mtime
                if now - stamp > max_age_s:
                    _rmtree(child)
                    removed.append(child)
        return removed

    def remove(self, key: str) -> bool:
        path = self.root / key
        if not path.is_dir():
            return False
        _rmtree(path)
        _frame_index.cache_clear()
        return not path.exists()

    def clear(self) -> int:
        n = 0
        for entry in self.entries():
            n += self.remove(entry.key)
        self.sweep_partials(max_age_s=0)
        return n

    def evict(self, max_bytes: int | None = None, keep: Iterable[str] = ()) -> list[str]:
        """Remove least-recently-used entries until the cache fits in ``max_bytes``."""
        limit = self.max_bytes if max_bytes is None else max_bytes
        keep = set(keep)
        entries = self.entries()          # most recently used first
        total = sum(e.bytes for e in entries)
        evicted = []
        for entry in reversed(entries):
            if total <= limit:
                break
            if entry.key in keep:
                continue
            if self.remove(entry.key):
                total -= entry.bytes
                evicted.append(entry.key)
        return evicted

    # ---- reading ------------------------------------------------------------

    def frame_index(self, entry: CacheEntry) -> pd.DataFrame:
        frames, rts, polarity = _frame_index(str(entry.path / FRAMES_FILE))
        return pd.DataFrame({"frame": frames, "retention_time": rts, "polarity": polarity})

    def load_mz_window(
        self,
        entry: CacheEntry,
        ranges: Iterable[tuple[float, float]],
        columns: Iterable[str] = ("mz", "intensity", "mobility", "frame"),
        with_retention_time: bool = True,
    ) -> pd.DataFrame:
        """Rows whose m/z falls in any of ``ranges`` (inclusive), read by pushdown.

        The file is mz-sorted, so each narrow window touches only the one or two
        row groups whose statistics overlap it. Overlapping ranges are merged
        into a single read.
        """
        import pyarrow.parquet as pq

        merged = _merge_ranges(ranges)
        cols = list(columns)
        if with_retention_time and "frame" not in cols:
            cols.append("frame")
        if not merged:
            return pd.DataFrame({c: [] for c in cols})
        filters = [[("mz", ">=", lo), ("mz", "<=", hi)] for lo, hi in merged]
        try:
            table = pq.read_table(entry.data_path, columns=cols, filters=filters)
        except OSError as exc:
            raise CacheError(f"Cannot read cache entry {entry.key}: {exc}") from exc
        df = table.to_pandas()
        # Stored narrow to halve the file; widened here so every query computes
        # in float64, as the legacy in-memory extractor did. Mobility is snapped
        # back to the exact float64 bin centres the extractor rounded to.
        if "intensity" in df:
            df["intensity"] = df["intensity"].astype(np.float64)
        if "mobility" in df:
            values = df["mobility"].to_numpy(np.float64)
            if entry.mobility.get("binned", True):
                # Snap back to the exact bin centres the extractor rounded 1/K0 to.
                width = entry.config.mobility_bin_width
                values = np.round(values / width) * width
            df["mobility"] = values
        if with_retention_time:
            frames, rts, _ = _frame_index(str(entry.path / FRAMES_FILE))
            lookup = np.full(int(frames.max()) + 1, np.nan)
            lookup[frames] = rts
            df["retention_time"] = lookup[df["frame"].to_numpy()]
        return df


    def load_precursors(self, entry: CacheEntry) -> pd.DataFrame:
        """DDA-PASEF precursors of this run (empty if it has none)."""
        import pyarrow.parquet as pq

        path = entry.path / PRECURSORS_FILE
        if not path.is_file():
            return pd.DataFrame()
        return pq.read_table(path).to_pandas()

    def load_msms(self, entry: CacheEntry, precursor_ids: Iterable[int]) -> dict[int, tuple]:
        """``{precursor_id: (mz, intensity)}`` for the requested precursors, by pushdown."""
        import pyarrow.parquet as pq

        path = entry.path / MSMS_FILE
        ids = sorted({int(i) for i in precursor_ids})
        if not path.is_file() or not ids:
            return {}
        df = pq.read_table(path, filters=[("precursor_id", "in", ids)]).to_pandas()
        return {int(pid): (g["mz"].to_numpy(np.float64), g["intensity"].to_numpy(np.float64))
                for pid, g in df.groupby("precursor_id", sort=False)}


def _merge_ranges(ranges: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for lo, hi in sorted((float(a), float(b)) for a, b in ranges):
        if hi < lo:
            raise ValueError(f"empty m/z range ({lo}, {hi})")
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


@lru_cache(maxsize=16)
def _frame_index(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        np.asarray(data["frame"], dtype=np.int64),
        np.asarray(data["retention_time"], dtype=np.float64),
        np.asarray(data["polarity"], dtype=object),
    )
