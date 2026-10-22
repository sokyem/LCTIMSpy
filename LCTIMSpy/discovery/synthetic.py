"""Synthetic LC-TIMS-MS runs with known ground truth, for testing and validation.

A scenario is a list of ``Species``: a neutral monoisotopic mass observed at
one charge state, eluting as a Gaussian in RT and drifting as a Gaussian in
1/K0, with an averagine-like isotope envelope. Two species with the same mass
and different RT and/or mobility model a putative isomer pair; a different
sequence of the same mass, an Na+ adduct, a second charge state, or a coeluting
pair can be added the same way.

These runs test the *software* -- detection limits, grouping, classification
logic. They are not evidence about real isomers, and must never be reported as
such (spec §15).
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROTON = 1.007276466812
NEUTRON = 1.0033548378           # 13C - 12C, the dominant isotope spacing
# Expected number of +1 Da heavy atoms per Da of averagine peptide
# (C 4.9384, H 7.7583, N 1.3577, O 1.4773, S 0.0417 per 111.1254 Da).
AVERAGINE_LAMBDA_PER_DA = 0.000536


def averagine_pattern(neutral_mass: float, n: int = 5) -> np.ndarray:
    """Relative isotope intensities (Poisson approximation), normalized to max = 1."""
    lam = AVERAGINE_LAMBDA_PER_DA * neutral_mass
    p = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in range(n)])
    return p / p.max()


@dataclass(frozen=True)
class Species:
    name: str
    neutral_mass: float
    charge: int
    rt: float                      # apex, minutes
    mobility: float                # apex, 1/K0
    abundance: float = 1e5         # apex intensity of the most abundant isotope
    rt_sigma: float = 0.04         # minutes
    mobility_sigma: float = 0.006  # 1/K0
    n_isotopes: int = 4
    sequence: str | None = None    # ground-truth identity, if any

    @property
    def mono_mz(self) -> float:
        return self.neutral_mass / self.charge + PROTON


@dataclass
class Scenario:
    species: list[Species]
    n_frames: int = 300
    seconds_per_frame: float = 1.2
    msms_every: int = 0             # every n-th frame is MS/MS (0 = MS1 only)
    mobility_range: tuple[float, float] = (0.6, 1.6)
    scan_step: float = 0.001        # 1/K0 per TIMS scan
    mz_jitter_ppm: float = 2.0
    noise_points: int = 150         # random background points per frame
    noise_level: float = 30.0
    min_signal: float = 5.0         # points below this are not recorded
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    def rt_of(self, frame: int) -> float:
        return frame * self.seconds_per_frame / 60.0

    def is_ms1(self, frame: int) -> bool:
        return not self.msms_every or frame % self.msms_every != 0


class SyntheticFrameSource:
    """A FrameSource producing the scenario's MS1 points (and a linear TIMS calibration)."""

    def __init__(self, scenario: Scenario):
        self.s = scenario
        lo, hi = scenario.mobility_range
        self._grid = np.arange(lo, hi, scenario.scan_step)        # one value per TIMS scan

    def read_frame(self, frame_id: int, num_scans: int):
        s = self.s
        rng = np.random.default_rng([s.seed, frame_id])
        rt = s.rt_of(frame_id)
        mz_parts, int_parts, mob_parts = [], [], []
        for sp in s.species:
            elution = math.exp(-0.5 * ((rt - sp.rt) / sp.rt_sigma) ** 2)
            if elution * sp.abundance < s.min_signal:
                continue
            drift = np.exp(-0.5 * ((self._grid - sp.mobility) / sp.mobility_sigma) ** 2)
            scans = self._grid[drift * elution * sp.abundance >= s.min_signal]
            weights = np.exp(-0.5 * ((scans - sp.mobility) / sp.mobility_sigma) ** 2)
            pattern = averagine_pattern(sp.neutral_mass, sp.n_isotopes)
            for k, rel in enumerate(pattern):
                inten = sp.abundance * elution * rel * weights
                keep = inten >= s.min_signal
                if not keep.any():
                    continue
                n = int(keep.sum())
                mz = sp.mono_mz + k * NEUTRON / sp.charge
                jitter = rng.normal(0, s.mz_jitter_ppm * 1e-6 * mz, n)
                mz_parts.append(mz + jitter)
                int_parts.append(np.round(inten[keep] * rng.uniform(0.85, 1.15, n)))
                mob_parts.append(scans[keep])
        if s.noise_points:
            mz_parts.append(rng.uniform(100, 1700, s.noise_points))
            int_parts.append(np.round(rng.exponential(s.noise_level, s.noise_points)) + 1)
            mob_parts.append(rng.choice(self._grid, s.noise_points))
        mz = np.concatenate(mz_parts)
        order = np.argsort(mz)
        return mz[order], np.concatenate(int_parts)[order], np.concatenate(mob_parts)[order]

    def scan_to_mobility(self, frame_id: int, scans):
        lo, hi = self.s.mobility_range
        return hi - self.s.scan_step * np.asarray(scans, dtype=float)


def write_synthetic_d(root: Path, name: str, scenario: Scenario) -> Path:
    """A minimal ``.d`` folder whose Frames table matches the scenario."""
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "analysis.tdf_bin").write_bytes(b"\0" * 16)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL, Polarity TEXT, "
                 "MsMsType INTEGER, NumScans INTEGER)")
    n_scans = int(round((scenario.mobility_range[1] - scenario.mobility_range[0])
                        / scenario.scan_step))
    conn.executemany("INSERT INTO Frames VALUES (?, ?, ?, ?, ?)", [
        (f, f * scenario.seconds_per_frame, "+", 0 if scenario.is_ms1(f) else 8, n_scans)
        for f in range(1, scenario.n_frames + 1)])
    conn.execute("CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT)")
    conn.executemany("INSERT INTO GlobalMetadata VALUES (?, ?)", [
        ("MzAcqRangeLower", "100"), ("MzAcqRangeUpper", "1700"),
        ("OneOverK0AcqRangeLower", str(scenario.mobility_range[0])),
        ("OneOverK0AcqRangeUpper", str(scenario.mobility_range[1])),
        ("InstrumentName", "synthetic"), ("AcquisitionSoftware", "LCTIMSpy.discovery.synthetic")])
    conn.commit()
    conn.close()
    return d
