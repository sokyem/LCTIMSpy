"""Extraction into the Parquet cache, driven by FakeFrameSource (no SDK)."""

import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from LCTIMSpy import cli
from LCTIMSpy.core.config import ExtractionConfig
from LCTIMSpy.core.errors import CancelledError, DatasetError
from LCTIMSpy.core.io import extract as extract_mod
from LCTIMSpy.core.io.cache import Cache, cache_key
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.core.progress import CancelToken
from tests.conftest import (
    ISOMERS,
    N_FRAMES,
    SECONDS_PER_FRAME,
    TARGET_MZ,
    FakeFrameSource,
    is_ms1,
    make_fake_d,
)

MS1_FRAMES = [i for i in range(1, N_FRAMES + 1) if is_ms1(i)]


def expected_frame(source, frame_id, width=0.0005):
    mz, intensity, mobility = source.spectrum(frame_id)
    return pd.DataFrame({
        "mz": mz,
        "intensity": intensity.astype(np.float32),
        "mobility": (np.round(mobility / width) * width).astype(np.float32),
        "frame": np.int32(frame_id),
    })


def expected_all(source, frames=MS1_FRAMES):
    return pd.concat([expected_frame(source, f) for f in frames], ignore_index=True)


def cache_dirs(cache):
    return sorted(p.name for p in cache.root.iterdir()) if cache.root.exists() else []


@pytest.fixture
def small_row_groups(monkeypatch):
    # Force several row groups and several buckets out of a small dataset.
    monkeypatch.setattr(extract_mod, "ROW_GROUP_ROWS", 2000)
    monkeypatch.setattr(extract_mod, "FRAME_BATCH", 7)


def test_extract_writes_complete_mz_sorted_cache(fake_d, cache, small_row_groups):
    source = FakeFrameSource()
    entry = extract_to_cache(fake_d, cache=cache, frame_source=source)

    assert source.calls == len(MS1_FRAMES)          # MS/MS frames never read
    table = pq.read_table(entry.data_path).to_pandas()
    assert entry.rows == len(table) == len(expected_all(source))
    assert table["mz"].is_monotonic_increasing

    # Same multiset of rows as the source produced.
    key = ["mz", "frame", "intensity", "mobility"]
    got = table.sort_values(key).reset_index(drop=True)
    want = expected_all(source).sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)

    # Row-group statistics are ordered and non-overlapping: this is what lets
    # a narrow m/z query skip almost every row group.
    md = pq.ParquetFile(entry.data_path).metadata
    assert md.num_row_groups > 1
    col = md.schema.to_arrow_schema().get_field_index("mz")
    bounds = [(md.row_group(i).column(col).statistics.min,
               md.row_group(i).column(col).statistics.max) for i in range(md.num_row_groups)]
    assert all(a[1] <= b[0] for a, b in zip(bounds, bounds[1:]))

    assert cache_dirs(cache) == [entry.key]           # no temp or spill leftovers
    assert not (entry.path / "spill").exists()


def test_mobility_is_binned(fake_d, cache):
    entry = extract_to_cache(fake_d, ExtractionConfig(mobility_bin_width=0.01), cache=cache,
                             frame_source=FakeFrameSource())
    mob = pq.read_table(entry.data_path, columns=["mobility"]).column(0).to_numpy()
    assert np.allclose(mob / 0.01, np.round(mob / 0.01), atol=1e-3)


def test_metadata_and_frame_index(fake_d, cache):
    entry = extract_to_cache(fake_d, cache=cache, frame_source=FakeFrameSource())
    meta = json.loads((entry.path / "meta.json").read_text())
    assert meta["config"] == ExtractionConfig().to_dict()
    assert meta["stats"]["frames_read"] == len(MS1_FRAMES)
    assert entry.dataset.n_frames == N_FRAMES
    frames = cache.frame_index(entry)
    assert frames["frame"].tolist() == MS1_FRAMES
    assert frames["retention_time"].iloc[0] == pytest.approx(SECONDS_PER_FRAME / 60)


def test_load_mz_window_matches_brute_force(fake_d, cache, small_row_groups):
    source = FakeFrameSource()
    entry = extract_to_cache(fake_d, cache=cache, frame_source=source)
    everything = expected_all(source)

    lo, hi = TARGET_MZ - 0.01, TARGET_MZ + 0.01
    got = cache.load_mz_window(entry, [(lo, hi)])
    want = everything[(everything.mz >= lo) & (everything.mz <= hi)]
    assert len(got) == len(want) > 0
    assert set(got["frame"]) == set(want["frame"])
    # Every hit belongs to one of the two isomers' mobility envelopes.
    near = [(got["mobility"] - m).abs() <= 0.01 for _, m in ISOMERS]
    assert (near[0] | near[1]).all()
    # retention_time is joined from the frame index.
    rt = dict(zip(want["frame"], want["frame"] * SECONDS_PER_FRAME / 60))
    assert np.allclose(got["retention_time"], got["frame"].map(rt))

    # Overlapping windows are merged, not double-counted.
    two = cache.load_mz_window(entry, [(lo, hi), (lo + 0.005, hi + 0.005)])
    assert len(two) == len(everything[(everything.mz >= lo) & (everything.mz <= hi + 0.005)])

    assert cache.load_mz_window(entry, []).empty


def test_second_run_uses_cache_and_force_reextracts(fake_d, cache):
    first = extract_to_cache(fake_d, cache=cache, frame_source=FakeFrameSource())
    source = FakeFrameSource()
    again = extract_to_cache(fake_d, cache=cache, frame_source=source)
    assert source.calls == 0
    assert again.key == first.key

    forced = extract_to_cache(fake_d, cache=cache, force=True, frame_source=source)
    assert source.calls == len(MS1_FRAMES)
    assert forced.key == first.key
    assert cache_dirs(cache) == [first.key]


def test_cache_key_tracks_config_and_source(fake_d):
    base = cache_key(fake_d, ExtractionConfig())
    assert cache_key(fake_d, ExtractionConfig()) == base
    assert cache_key(fake_d, ExtractionConfig(mobility_bin_width=0.001)) != base
    assert cache_key(fake_d, ExtractionConfig(ms1_only=False)) != base
    st = (fake_d / "analysis.tdf_bin").stat()
    os.utime(fake_d / "analysis.tdf_bin", ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert cache_key(fake_d, ExtractionConfig()) != base


def test_cancel_leaves_no_partial_cache(fake_d, cache):
    token = CancelToken()

    def cancel_midway(frame_id):
        if frame_id == 61:  # an MS1 frame; multiples of 5 are never read
            token.cancel()

    with pytest.raises(CancelledError):
        extract_to_cache(fake_d, cache=cache, cancel=token,
                         frame_source=FakeFrameSource(on_frame=cancel_midway))
    assert cache_dirs(cache) == []


def test_keyboard_interrupt_leaves_no_partial_cache(fake_d, cache):
    def interrupt(frame_id):
        if frame_id == 31:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        extract_to_cache(fake_d, cache=cache, frame_source=FakeFrameSource(on_frame=interrupt))
    assert cache_dirs(cache) == []


def test_failed_and_empty_frames_are_counted_not_fatal(fake_d, cache):
    entry = extract_to_cache(fake_d, cache=cache,
                             frame_source=FakeFrameSource(fail_frames={7}, empty_frames={8, 9}))
    assert entry.stats["frames_failed"] == 1
    assert entry.stats["frames_empty"] == 2
    assert "frame 7" in entry.stats["first_error"]
    frames = set(pq.read_table(entry.data_path, columns=["frame"]).column(0).to_pylist())
    assert frames == set(MS1_FRAMES) - {7, 8, 9}


def test_all_frames_failing_is_an_error(fake_d, cache):
    with pytest.raises(DatasetError, match="corrupt frame"):
        extract_to_cache(fake_d, cache=cache,
                         frame_source=FakeFrameSource(fail_frames=set(MS1_FRAMES)))
    assert cache_dirs(cache) == []


def test_lru_eviction_keeps_most_recent(tmp_path, cache):
    a = extract_to_cache(make_fake_d(tmp_path, "a.d"), cache=cache, frame_source=FakeFrameSource())
    b = extract_to_cache(make_fake_d(tmp_path, "b.d"), cache=cache, frame_source=FakeFrameSource())
    old = time.time() - 1000
    os.utime(a.path / "last_used", (old, old))
    evicted = cache.evict(max_bytes=b.bytes)
    assert evicted == [a.key]
    assert [e.key for e in cache.entries()] == [b.key]


def test_sweep_removes_only_stale_temp_dirs(cache):
    cache.root.mkdir(parents=True)
    stale = cache.root / "abc.tmp-11111111"
    fresh = cache.root / "def.tmp-22222222"
    for d in (stale, fresh):
        d.mkdir()
        Cache.heartbeat(d)
    old = time.time() - 7200
    os.utime(stale / "heartbeat", (old, old))
    assert cache.sweep_partials() == [stale]
    assert fresh.exists()


def test_cli_doctor_and_cache_list(tmp_path, capsys):
    root = str(tmp_path / "cache")
    assert cli.main(["--cache-dir", root, "doctor"]) == 0
    out = capsys.readouterr().out
    assert "Bruker TDF-SDK" in out and root in out
    assert cli.main(["--cache-dir", root, "cache", "list"]) == 0
    assert "No cached datasets" in capsys.readouterr().out


def test_cli_extract_without_sdk_explains_fix(fake_d, tmp_path, capsys, monkeypatch):
    from LCTIMSpy.core import sdk

    monkeypatch.setattr(sdk, "_library", None)
    monkeypatch.setenv("LCTIMSPY_TIMSDATA", str(tmp_path / "missing.dll"))
    assert cli.main(["--cache-dir", str(tmp_path / "cache"), "extract", str(fake_d)]) == 1
    err = capsys.readouterr().err
    assert "not found" in err and "fix:" in err
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())
