"""Shared test helpers.

``fake_d`` builds a minimal timsTOF ``.d`` folder: a real ``analysis.tdf``
sqlite database with ``Frames`` and ``GlobalMetadata`` tables and an empty
``analysis.tdf_bin``. ``FakeFrameSource`` stands in for the TDF-SDK reader, so
the whole extract -> cache -> query pipeline runs without the Bruker SDK.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from LCTIMSpy import get_fragmentMz
from LCTIMSpy.core.io.cache import Cache

PROTON = 1.00727646677

# Synthetic acquisition: every 5th frame is MS/MS; RT is 0.6 s per frame.
N_FRAMES = 120
NUM_SCANS = 10
SECONDS_PER_FRAME = 0.6
# Two fragment ions, each present as an isomer pair: isomer A elutes at frame
# 40 (RT 0.40 min), isomer B at frame 80 (RT 0.80 min) with a higher mobility.
# The second target's B isomer is shifted a further 0.004 so the two targets'
# mobility differences differ, which the stats tests rely on.
TARGETS = (500.25, 700.40)
ISOMERS = ((40, 0.9123), (80, 0.9500))      # (apex frame, mobility)
SECOND_TARGET_EXTRA_SHIFT = 0.004
TARGET_MZ = TARGETS[0]
TARGET_MOBILITY = ISOMERS[0][1]
WINDOWS = ((0.30, 0.50), (0.70, 0.90))      # RT windows (min) around the two isomers


def is_ms1(frame_id: int) -> bool:
    return frame_id % 5 != 0


def make_fake_d(root: Path, name: str = "sample.d", n_frames: int = N_FRAMES) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "analysis.tdf_bin").write_bytes(b"\0" * 16)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute(
        "CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL, Polarity TEXT, "
        "MsMsType INTEGER, NumScans INTEGER)"
    )
    conn.executemany(
        "INSERT INTO Frames VALUES (?, ?, ?, ?, ?)",
        [(i, i * SECONDS_PER_FRAME, "+", 0 if is_ms1(i) else 8, NUM_SCANS)
         for i in range(1, n_frames + 1)],
    )
    conn.execute("CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT)")
    conn.executemany(
        "INSERT INTO GlobalMetadata VALUES (?, ?)",
        [("MzAcqRangeLower", "100"), ("MzAcqRangeUpper", "1700"),
         ("OneOverK0AcqRangeLower", "0.6"), ("OneOverK0AcqRangeUpper", "1.4"),
         ("InstrumentName", "timsTOF Fake"), ("AcquisitionSoftware", "pytest")],
    )
    conn.commit()
    conn.close()
    return d


class FakeFrameSource:
    """Deterministic spectra: noise across 100-1700 m/z plus the eluting isomer peaks.

    ``seed`` varies peak intensities between replicates. ``fail_frames`` raise,
    ``empty_frames`` return None, and ``on_frame`` is called before every read
    (tests use it to cancel or interrupt mid-run).
    """

    def __init__(self, fail_frames=(), empty_frames=(), on_frame=None, seed=0):
        self.fail_frames = set(fail_frames)
        self.empty_frames = set(empty_frames)
        self.on_frame = on_frame
        self.seed = seed
        self.calls = 0

    def spectrum(self, frame_id: int):
        rng = np.random.default_rng([self.seed, frame_id])
        n = 200
        mz = list(rng.uniform(100, 1700, n))
        intensity = list(rng.uniform(10, 100, n).round())
        mobility = list(rng.uniform(0.6, 1.4, n))
        for t_idx, target in enumerate(TARGETS):
            for i_idx, (apex, mob) in enumerate(ISOMERS):
                amp = 1000 * (1 + 0.5 * t_idx) * np.exp(-0.5 * ((frame_id - apex) / 4) ** 2)
                if amp < 1:
                    continue
                centre = mob + (SECOND_TARGET_EXTRA_SHIFT if t_idx == 1 and i_idx == 1 else 0)
                for k in range(-3, 4):
                    mz.append(target + rng.normal(0, 0.002))
                    weight = np.exp(-0.25 * k * k) * rng.uniform(0.8, 1.2)
                    intensity.append(round(amp * weight) + 1)
                    mobility.append(centre + 0.003 * k)
        mz, intensity, mobility = map(np.asarray, (mz, intensity, mobility))
        order = np.argsort(mz)
        return mz[order], intensity[order], mobility[order]

    def read_frame(self, frame_id: int, num_scans: int):
        self.calls += 1
        if self.on_frame is not None:
            self.on_frame(frame_id)
        if frame_id in self.fail_frames:
            raise RuntimeError(f"corrupt frame {frame_id}")
        if frame_id in self.empty_frames:
            return None
        return self.spectrum(frame_id)


@pytest.fixture
def fake_d(tmp_path):
    return make_fake_d(tmp_path / "data")


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "cache")


@pytest.fixture(scope="session")
def fragments():
    return get_fragmentMz


class SlowFakeFrameSource(FakeFrameSource):
    """FakeFrameSource that takes ``delay`` seconds per frame (cancellation tests)."""

    def __init__(self, delay: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay

    def read_frame(self, frame_id: int, num_scans: int):
        import time

        time.sleep(self.delay)
        return super().read_frame(frame_id, num_scans)


class CrashingFrameSource(FakeFrameSource):
    """Prints like a chatty native library, then dies the way an access violation does."""

    def read_frame(self, frame_id: int, num_scans: int):
        if frame_id == 11:
            import os

            print("timsdata: something went badly wrong", flush=True)
            os._exit(3)
        return super().read_frame(frame_id, num_scans)
