"""DDA-PASEF precursors and MS/MS spectra from a Bruker ``.d`` folder.

Tables (read with sqlite3, no SDK needed):

* ``Precursors`` -- one row per selected precursor: ``Id, LargestPeakMz,
  AverageMz, MonoisotopicMz, Charge, ScanNumber, Intensity, Parent``.
  ``MonoisotopicMz`` and ``Charge`` are NULL when Bruker's acquisition
  software could not determine them; they are kept as missing, never guessed.
  ``ScanNumber`` is the precursor's (fractional) TIMS scan in the parent MS1
  frame -- a *scan coordinate*, not a mobility.
* ``PasefFrameMsMsInfo`` -- one row per PASEF MS/MS frame segment:
  ``Frame, ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth,
  CollisionEnergy, Precursor``. A precursor is usually fragmented in several
  frames; Bruker's SDK sums those segments into one MS/MS spectrum.

Scan numbers become 1/K0 only through the SDK, with the dataset's own
calibration. In timsTOF data scan number *increases as 1/K0 decreases*, so a
scan range maps to a 1/K0 interval with its bounds swapped.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .bruker import _connect
from .paths import validate_d_folder

PRECURSOR_COLUMNS = ("Id", "LargestPeakMz", "AverageMz", "MonoisotopicMz", "Charge",
                     "ScanNumber", "Intensity", "Parent")
PASEF_COLUMNS = ("Frame", "ScanNumBegin", "ScanNumEnd", "IsolationMz", "IsolationWidth",
                 "CollisionEnergy", "Precursor")
MSMS_BATCH = 500


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def read_pasef_tables(d_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(precursors, pasef_frames)``; both empty when the run is not DDA-PASEF."""
    d = validate_d_folder(d_path)
    with closing(_connect(d / "analysis.tdf")) as conn:
        tables = _tables(conn)
        if not {"Precursors", "PasefFrameMsMsInfo"} <= tables:
            return (pd.DataFrame(columns=PRECURSOR_COLUMNS), pd.DataFrame(columns=PASEF_COLUMNS))
        precursors = pd.read_sql_query(
            f"SELECT {', '.join(PRECURSOR_COLUMNS)} FROM Precursors ORDER BY Id", conn)
        pasef = pd.read_sql_query(
            f"SELECT {', '.join(PASEF_COLUMNS)} FROM PasefFrameMsMsInfo ORDER BY Precursor, Frame",
            conn)
    return precursors, pasef


def summarize_precursors(precursors: pd.DataFrame, pasef: pd.DataFrame,
                         frames: pd.DataFrame) -> pd.DataFrame:
    """One row per precursor with acquisition context; mobility columns filled later.

    Columns: precursor_id, parent_frame, rt (min, of the parent MS1 frame),
    mz_largest, mz_average, mz_monoisotopic, charge (nullable), intensity,
    scan_number, n_msms_frames, msms_frame_first, msms_rt_first, msms_rt_last,
    scan_begin, scan_end, isolation_mz, isolation_width, collision_energy.
    """
    rt_of = pd.Series(frames["Time"].to_numpy() / 60.0, index=frames["Id"].to_numpy())
    out = pd.DataFrame({
        "precursor_id": precursors["Id"].astype("int64"),
        "parent_frame": precursors["Parent"].astype("int64"),
        "mz_largest": precursors["LargestPeakMz"].astype(float),
        "mz_average": precursors["AverageMz"].astype(float),
        "mz_monoisotopic": pd.to_numeric(precursors["MonoisotopicMz"], errors="coerce"),
        "charge": pd.to_numeric(precursors["Charge"], errors="coerce").astype("Int64"),
        "intensity": precursors["Intensity"].astype(float),
        "scan_number": precursors["ScanNumber"].astype(float),
    })
    out["rt"] = out["parent_frame"].map(rt_of)
    if pasef.empty:
        for col in ("n_msms_frames", "msms_frame_first", "msms_rt_first", "msms_rt_last",
                    "scan_begin", "scan_end", "isolation_mz", "isolation_width",
                    "collision_energy"):
            out[col] = np.nan
        return out
    p = pasef.assign(rt=pasef["Frame"].map(rt_of))
    agg = p.groupby("Precursor").agg(
        n_msms_frames=("Frame", "size"),
        msms_frame_first=("Frame", "min"),
        msms_rt_first=("rt", "min"),
        msms_rt_last=("rt", "max"),
        scan_begin=("ScanNumBegin", "min"),
        scan_end=("ScanNumEnd", "max"),
        isolation_mz=("IsolationMz", "first"),
        isolation_width=("IsolationWidth", "first"),
        collision_energy=("CollisionEnergy", "mean"),
    )
    return out.join(agg, on="precursor_id")


def add_mobility(summary: pd.DataFrame, scan_to_mobility) -> pd.DataFrame:
    """Convert scan coordinates to 1/K0 (V·s/cm²) with the dataset's calibration.

    ``scan_to_mobility(frame_id, scans) -> ndarray`` is the SDK conversion.
    Adds ``mobility`` (precursor apex) and ``isolation_mobility_low/high`` (the
    1/K0 interval the quadrupole isolated over -- what decides whether an MS/MS
    spectrum can belong to one mobility population or several).
    """
    out = summary.copy()
    out["mobility"] = np.nan
    out["isolation_mobility_low"] = np.nan
    out["isolation_mobility_high"] = np.nan
    for frame_id, idx in out.groupby("parent_frame").groups.items():
        rows = out.loc[idx]
        out.loc[idx, "mobility"] = scan_to_mobility(int(frame_id), rows["scan_number"].to_numpy(float))
    has_window = out["msms_frame_first"].notna()
    for frame_id, idx in out[has_window].groupby("msms_frame_first").groups.items():
        rows = out.loc[idx]
        begin = scan_to_mobility(int(frame_id), rows["scan_begin"].to_numpy(float))
        end = scan_to_mobility(int(frame_id), rows["scan_end"].to_numpy(float))
        out.loc[idx, "isolation_mobility_low"] = np.minimum(begin, end)
        out.loc[idx, "isolation_mobility_high"] = np.maximum(begin, end)
    return out
