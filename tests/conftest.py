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
TARGET_MZ = 500.25          # a fragment that elutes around frame 40
TARGET_MOBILITY = 0.9123


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
    """Deterministic spectra: noise across 100-1700 m/z plus an eluting target peak.

    ``fail_frames`` raise, ``empty_frames`` return None, and ``on_frame`` is
    called before every read (tests use it to cancel or interrupt mid-run).
    """

    def __init__(self, fail_frames=(), empty_frames=(), on_frame=None):
        self.fail_frames = set(fail_frames)
        self.empty_frames = set(empty_frames)
        self.on_frame = on_frame
        self.calls = 0

    def spectrum(self, frame_id: int):
        rng = np.random.default_rng(frame_id)
        n = 200
        mz = rng.uniform(100, 1700, n)
        intensity = rng.uniform(10, 100, n).round()
        mobility = rng.uniform(0.6, 1.4, n)
        peak = 1000 * np.exp(-0.5 * ((frame_id - 40) / 4) ** 2)
        mz = np.append(mz, TARGET_MZ + rng.normal(0, 0.002))
        intensity = np.append(intensity, round(peak) + 1)
        mobility = np.append(mobility, TARGET_MOBILITY)
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
