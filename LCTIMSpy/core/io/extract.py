"""Full extraction of a ``.d`` folder into the Parquet cache.

The cache must be mz-sorted (that is what makes every later query a pushdown
read), but a full extraction is 1-10 GB -- too big to sort in memory. So it runs
in two passes with memory bounded by one bucket rather than the whole dataset:

1. **Spill.** Frames are read in batches; each batch's rows are split into
   fixed m/z buckets and appended to one Arrow IPC file per bucket.
2. **Finalize.** Buckets are read back in m/z order, sorted, and appended to
   ``data.parquet``. Concatenating sorted, non-overlapping buckets yields a
   globally sorted file.

Both passes report progress and poll the CancelToken; any exit other than
success leaves nothing behind (see ``Cache.writer``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ExtractionConfig
from ..errors import DatasetError
from ..progress import CancelToken, ProgressEvent, ProgressReporter, StageTimer, null_reporter
from .bruker import FrameSource, open_frame_source, read_dataset_meta, read_frames
from .cache import (
    DATA_FILE,
    EXTRACTOR_VERSION,
    FRAMES_FILE,
    META_FILE,
    SCHEMA_VERSION,
    Cache,
    CacheEntry,
    source_fingerprint,
)
from .paths import validate_d_folder

FRAME_BATCH = 50
N_BUCKETS = 256
ROW_GROUP_ROWS = 1 << 20          # ~20 MB per row group at 20 bytes/row
DEFAULT_MZ_RANGE = (0.0, 4000.0)  # used when GlobalMetadata has no acquisition range


def _schema():
    import pyarrow as pa

    return pa.schema([
        ("mz", pa.float64()),
        ("intensity", pa.float32()),
        ("mobility", pa.float32()),
        ("frame", pa.int32()),
    ])


@dataclass
class ExtractStats:
    rows: int = 0
    frames_read: int = 0
    frames_empty: int = 0
    frames_failed: int = 0
    first_error: str | None = None
    mz_min: float | None = None
    mz_max: float | None = None
    seconds: float = 0.0
    failed_frames: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["failed_frames"] = self.failed_frames[:100]
        return d


def extract_to_cache(
    d_path: str | Path,
    config: ExtractionConfig | None = None,
    *,
    cache: Cache | None = None,
    force: bool = False,
    reporter: ProgressReporter = null_reporter,
    cancel: CancelToken | None = None,
    frame_source: FrameSource | None = None,
    on_temp_dir: Callable[[Path], None] | None = None,
) -> CacheEntry:
    """Extract ``d_path`` into the cache and return the entry.

    Returns the existing entry immediately if one is valid, unless ``force``.
    ``frame_source`` replaces the TDF-SDK reader; tests use it to run the whole
    pipeline without the SDK. ``on_temp_dir`` is told the private temp folder
    as soon as it exists, so a supervising process can remove it if this one
    is killed before it can clean up after itself.
    """
    config = config or ExtractionConfig()
    cache = cache or Cache()
    cancel = cancel or CancelToken()
    d = validate_d_folder(d_path)
    key = cache.key_for(d, config)

    if not force:
        existing = cache.get(key)
        if existing is not None:
            reporter(ProgressEvent("cached", 1, 1, f"Using cached extraction {key}"))
            return existing

    cache.sweep_partials()
    dataset = read_dataset_meta(d)
    frames = read_frames(d)
    if config.ms1_only:
        frames = frames[frames["MsMsType"] == 0]
    if frames.empty:
        raise DatasetError(f"{d} has no {'MS1 ' if config.ms1_only else ''}frames to extract.")
    fingerprint = source_fingerprint(d)

    with ExitStack() as stack:
        # Open the source before creating the temp dir, so a missing SDK fails
        # fast without touching the cache.
        source = (
            frame_source if frame_source is not None
            else stack.enter_context(open_frame_source(d))
        )
        tmp = stack.enter_context(cache.writer(key))
        if on_temp_dir is not None:
            on_temp_dir(tmp)
        mz_range = (
            (dataset.mz_acq_lower, dataset.mz_acq_upper)
            if dataset.mz_acq_lower is not None and dataset.mz_acq_upper is not None
            else DEFAULT_MZ_RANGE
        )
        stats = _extract(source, frames, config, tmp, mz_range, reporter, cancel)

        _write_json(tmp / FRAMES_FILE, {
            "frame": frames["Id"].astype(int).tolist(),
            "retention_time": (frames["Time"] / 60.0).astype(float).tolist(),
            "polarity": frames["Polarity"].astype(str).tolist(),
        })
        data_bytes = sum(f.stat().st_size for f in tmp.iterdir() if f.is_file())
        _write_json(tmp / META_FILE, {
            "schema": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "key": key,
            "source_path": str(d),
            "source_fingerprint": fingerprint,
            "config": config.to_dict(),
            "dataset": dataset.to_dict(),
            "rows": stats.rows,
            "bytes": data_bytes,
            "created": time.time(),
            "stats": stats.to_dict(),
        })

    entry = cache.get(key)
    if entry is None:
        raise DatasetError(f"Extraction of {d} finished but the cache entry is unreadable.")
    cache.evict(keep={key})
    return entry


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def _extract(
    source: FrameSource,
    frames: pd.DataFrame,
    config: ExtractionConfig,
    tmp: Path,
    mz_range: tuple[float, float],
    reporter: ProgressReporter,
    cancel: CancelToken,
) -> ExtractStats:
    import pyarrow as pa

    start = time.perf_counter()
    stats = ExtractStats()
    schema = _schema()
    edges = np.linspace(mz_range[0], mz_range[1], N_BUCKETS + 1)
    spill_dir = tmp / "spill"
    spill_dir.mkdir()
    # bucket -> (writer, sink). Sinks are owned explicitly: a writer opened
    # from a path does not close its file, which keeps it locked on Windows.
    writers: dict[int, tuple[pa.ipc.RecordBatchFileWriter, pa.NativeFile]] = {}
    width = config.mobility_bin_width

    ids = frames["Id"].to_numpy(dtype=np.int64)
    num_scans = frames["NumScans"].to_numpy(dtype=np.int64)
    timer = StageTimer("extract", len(ids), reporter)
    timer.update(0, "Reading frames")

    try:
        for batch_start in range(0, len(ids), FRAME_BATCH):
            mz_parts, int_parts, mob_parts, frame_parts = [], [], [], []
            for frame_id, n_scans in zip(
                ids[batch_start:batch_start + FRAME_BATCH],
                num_scans[batch_start:batch_start + FRAME_BATCH],
            ):
                # Per frame, not per batch: a batch of slow frames (network
                # drive, large frames) must not outlast the server's grace period.
                cancel.raise_if_cancelled()
                try:
                    spectrum = source.read_frame(int(frame_id), int(n_scans))
                except Exception as exc:  # one bad frame should not lose an hour of work
                    stats.frames_failed += 1
                    stats.failed_frames.append(int(frame_id))
                    if stats.first_error is None:
                        stats.first_error = f"frame {frame_id}: {type(exc).__name__}: {exc}"
                    continue
                stats.frames_read += 1
                if spectrum is None:
                    stats.frames_empty += 1
                    continue
                mz, intensity, mobility = spectrum
                mz_parts.append(np.asarray(mz, dtype=np.float64))
                int_parts.append(np.asarray(intensity, dtype=np.float32))
                mob_parts.append(np.asarray(mobility, dtype=np.float64))
                frame_parts.append(np.full(len(mz), frame_id, dtype=np.int32))

            if mz_parts:
                _spill_batch(
                    np.concatenate(mz_parts),
                    np.concatenate(int_parts),
                    (np.round(np.concatenate(mob_parts) / width) * width).astype(np.float32),
                    np.concatenate(frame_parts),
                    edges, spill_dir, writers, schema, stats,
                )
            Cache.heartbeat(tmp)
            done = min(batch_start + FRAME_BATCH, len(ids))
            timer.update(done, f"{stats.rows:,} points")
    finally:
        for writer, sink in writers.values():
            writer.close()
            sink.close()

    if stats.rows == 0:
        detail = f" First error: {stats.first_error}" if stats.first_error else ""
        raise DatasetError(f"No spectra could be extracted.{detail}")

    _finalize(spill_dir, tmp / DATA_FILE, sorted(writers), schema, reporter, cancel)
    spill_dir.rmdir()
    stats.seconds = time.perf_counter() - start
    return stats


def _spill_batch(mz, intensity, mobility, frame, edges, spill_dir, writers, schema, stats):
    import pyarrow as pa

    bucket = np.clip(np.searchsorted(edges, mz, side="right") - 1, 0, len(edges) - 2)
    order = np.argsort(bucket, kind="stable")
    bucket = bucket[order]
    bounds = np.flatnonzero(np.diff(bucket)) + 1
    starts = np.concatenate(([0], bounds))
    stops = np.concatenate((bounds, [len(bucket)]))
    for s, e in zip(starts, stops):
        b = int(bucket[s])
        idx = order[s:e]
        batch = pa.record_batch(
            [pa.array(mz[idx]), pa.array(intensity[idx]), pa.array(mobility[idx]),
             pa.array(frame[idx])],
            schema=schema,
        )
        if b not in writers:
            sink = pa.OSFile(str(spill_dir / f"b{b:04d}.arrow"), "wb")
            writers[b] = (pa.ipc.new_file(sink, schema), sink)
        writers[b][0].write_batch(batch)

    stats.rows += len(mz)
    lo, hi = float(mz.min()), float(mz.max())
    stats.mz_min = lo if stats.mz_min is None else min(stats.mz_min, lo)
    stats.mz_max = hi if stats.mz_max is None else max(stats.mz_max, hi)


def _finalize(spill_dir, out_path, buckets, schema, reporter, cancel):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    timer = StageTimer("finalize", len(buckets), reporter)
    timer.update(0, "Sorting by m/z")
    with pq.ParquetWriter(
        str(out_path), schema, compression="zstd", compression_level=3, write_statistics=True
    ) as writer:
        for i, b in enumerate(buckets, start=1):
            cancel.raise_if_cancelled()
            path = spill_dir / f"b{b:04d}.arrow"
            # Read through Python bytes: pyarrow file readers keep the handle
            # open, which on Windows makes the unlink below fail.
            table = pa.ipc.open_file(pa.py_buffer(path.read_bytes())).read_all()
            table = table.take(pc.sort_indices(table["mz"]))
            writer.write_table(table, row_group_size=ROW_GROUP_ROWS)
            del table
            path.unlink()
            timer.update(i)
