"""mzML input, with ion mobility taken only from what the file actually contains.

Converters preserve mobility in different ways, or not at all. Three shapes
are recognized (and reported by ``MzmlIndex.mobility``):

* **per-spectrum scan parameter** -- one spectrum per TIMS scan, each with an
  ``inverse reduced ion mobility`` (MS:1002815) value; also drift time
  (MS:1002476) and FAIMS compensation voltage (MS:1001581);
* **per-peak binary array** -- e.g. msconvert ``--combineIonMobilitySpectra``
  writes ``mean inverse reduced ion mobility array`` (MS:1003006) beside m/z;
* **none** -- conventional LC-MS/MS, or a conversion that dropped mobility.

mzML carries no TIMS calibration, so 1/K0 values are taken as converted by the
exporting software; a CCS is never derived from them here.

MS1 spectra that share a retention time (one per TIMS scan) are grouped into
one *frame*, so the cache layout matches Bruker data. MS/MS spectra become the
precursor table with their isolation window, charge and precursor mobility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..errors import DatasetError
from ..results import DatasetMeta

MOBILITY_SCALARS = {
    "inverse reduced ion mobility": ("1/K0", "V·s/cm²"),
    "ion mobility drift time": ("drift time", "ms"),
    "FAIMS compensation voltage": ("FAIMS CV", "V"),
}
MOBILITY_ARRAYS = {
    "mean inverse reduced ion mobility array": ("1/K0", "V·s/cm²"),
    "raw inverse reduced ion mobility array": ("1/K0", "V·s/cm²"),
    "mean drift time array": ("drift time", "ms"),
    "raw drift time array": ("drift time", "ms"),
    "mean ion mobility array": ("ion mobility (unspecified)", None),
    "raw ion mobility array": ("ion mobility (unspecified)", None),
}
RT_TOLERANCE_MIN = 1e-6      # spectra closer than this in RT belong to one frame


def is_mzml(path: str | Path) -> bool:
    p = Path(path)
    return p.suffix.lower() == ".mzml" or p.name.lower().endswith(".mzml.gz")


def _minutes(value) -> float:
    unit = str(getattr(value, "unit_info", "") or "").lower()
    v = float(value)
    return v / 60.0 if unit.startswith("second") or unit in ("s", "sec") else v


def _scan(spectrum: dict) -> dict:
    scans = (spectrum.get("scanList") or {}).get("scan") or [{}]
    return scans[0]


def _scalar_mobility(d: dict) -> tuple[float | None, str | None]:
    for name in MOBILITY_SCALARS:
        if name in d:
            return float(d[name]), name
    return None, None


@dataclass
class MzmlIndex:
    """Everything known about an mzML file after one metadata pass."""

    path: Path
    frames: pd.DataFrame                    # Id, Time (s), Polarity, MsMsType, NumScans
    frame_spectra: dict[int, list[str]]     # frame id -> MS1 spectrum ids
    precursors: pd.DataFrame                # same columns as the TDF precursor summary
    msms_ids: dict[int, str]                # precursor id -> MS/MS spectrum id
    mobility: dict                          # coordinate, unit, representation, cv name
    mz_range: tuple[float, float] | None
    counts: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def dataset_meta(self) -> DatasetMeta:
        rt = self.frames["Time"] / 60.0 if len(self.frames) else pd.Series([0.0])
        return DatasetMeta(
            source_path=str(self.path), name=self.path.name.split(".")[0],
            n_frames=int(self.counts.get("spectra", 0)), n_ms1_frames=len(self.frames),
            rt_min=float(rt.min()), rt_max=float(rt.max()),
            mz_acq_lower=self.mz_range[0] if self.mz_range else None,
            mz_acq_upper=self.mz_range[1] if self.mz_range else None,
        )


def index_mzml(path: str | Path, reporter=None, cancel=None) -> MzmlIndex:
    """One pass over spectrum metadata (binary data not decoded)."""
    from pyteomics import mzml

    path = Path(path)
    frame_rows, frame_spectra = [], {}
    precursor_rows, msms_ids = [], {}
    counts = {"spectra": 0, "MS1": 0, "MS2": 0, "other": 0}
    representation = cv_name = None
    coordinate = unit = None
    mz_lo, mz_hi = np.inf, -np.inf
    frame_id, last_rt = 0, None
    try:
        reader = mzml.MzML(str(path), read_schema=False, decode_binary=False, use_index=True)
    except Exception as exc:                                  # lxml/pyteomics parse errors
        raise DatasetError(f"{path.name}: not a readable mzML file: {exc}") from exc
    with reader:
        for spectrum in reader:
            counts["spectra"] += 1
            if cancel is not None and counts["spectra"] % 2000 == 0:
                cancel.raise_if_cancelled()
            if reporter is not None and counts["spectra"] % 5000 == 0:
                from ..progress import ProgressEvent

                reporter(ProgressEvent("index", counts["spectra"], 0,
                                       f"{counts['spectra']:,} spectra indexed"))
            level = int(spectrum.get("ms level", 1))
            scan = _scan(spectrum)
            if "scan start time" not in scan:
                continue
            rt = _minutes(scan["scan start time"])
            if "lowest observed m/z" in spectrum and "highest observed m/z" in spectrum:
                mz_lo = min(mz_lo, float(spectrum["lowest observed m/z"]))
                mz_hi = max(mz_hi, float(spectrum["highest observed m/z"]))
            array_names = [k for k in spectrum if k in MOBILITY_ARRAYS]
            scalar, scalar_name = _scalar_mobility(scan)
            if representation is None:
                if array_names:
                    representation, cv_name = "per-peak array", array_names[0]
                    coordinate, unit = MOBILITY_ARRAYS[cv_name]
                elif scalar is not None:
                    representation, cv_name = "per-spectrum scan parameter", scalar_name
                    coordinate, unit = MOBILITY_SCALARS[scalar_name]
            if level == 1:
                counts["MS1"] += 1
                if last_rt is None or abs(rt - last_rt) > RT_TOLERANCE_MIN:
                    frame_id += 1
                    last_rt = rt
                    polarity = "-" if "negative scan" in spectrum else "+"
                    frame_rows.append([frame_id, rt * 60.0, polarity, 0, 0])
                    frame_spectra[frame_id] = []
                frame_spectra[frame_id].append(spectrum["id"])
                frame_rows[-1][4] += 1
            elif level == 2:
                counts["MS2"] += 1
                pid = len(precursor_rows) + 1
                msms_ids[pid] = spectrum["id"]
                precursor_rows.append(_precursor_row(pid, spectrum, scan, rt, frame_id))
            else:
                counts["other"] += 1

    frames = pd.DataFrame(frame_rows, columns=["Id", "Time", "Polarity", "MsMsType", "NumScans"])
    precursors = pd.DataFrame(precursor_rows, columns=PRECURSOR_SUMMARY_COLUMNS)
    precursors["charge"] = precursors["charge"].astype("Int64")
    notes = ["mzML carries no TIMS calibration: mobility is used as converted by the "
             "exporting software and CCS is not derived"]
    if representation is None:
        notes.append("no ion mobility found in the file; conventional LC-MS/MS processing")
    if counts["MS2"] and precursors["mobility"].isna().all() and representation:
        notes.append("MS/MS spectra carry no precursor mobility; mobility-resolved MS/MS "
                     "association will be ambiguous")
    return MzmlIndex(
        path=path, frames=frames, frame_spectra=frame_spectra, precursors=precursors,
        msms_ids=msms_ids,
        mobility={"present": representation is not None, "coordinate": coordinate, "unit": unit,
                  "representation": representation or "none", "cv_name": cv_name},
        mz_range=(mz_lo, mz_hi) if np.isfinite(mz_lo) else None, counts=counts, notes=notes,
    )


PRECURSOR_SUMMARY_COLUMNS = [
    "precursor_id", "parent_frame", "mz_largest", "mz_average", "mz_monoisotopic", "charge",
    "intensity", "scan_number", "rt", "n_msms_frames", "msms_frame_first", "msms_rt_first",
    "msms_rt_last", "scan_begin", "scan_end", "isolation_mz", "isolation_width",
    "collision_energy", "mobility", "isolation_mobility_low", "isolation_mobility_high",
]


def _precursor_row(pid: int, spectrum: dict, scan: dict, rt: float, parent_frame: int) -> list:
    precursor = ((spectrum.get("precursorList") or {}).get("precursor") or [{}])[0]
    ion = ((precursor.get("selectedIonList") or {}).get("selectedIon") or [{}])[0]
    window = precursor.get("isolationWindow") or {}
    activation = precursor.get("activation") or {}
    mz = ion.get("selected ion m/z")
    mobility, _ = _scalar_mobility(ion)
    if mobility is None:
        mobility, _ = _scalar_mobility(scan)
    lower = window.get("isolation window lower offset")
    upper = window.get("isolation window upper offset")
    width = float(lower) + float(upper) if lower is not None and upper is not None else np.nan
    charge = ion.get("charge state")
    return [
        pid, parent_frame or np.nan, mz, np.nan,
        mz if charge is not None else np.nan,              # exporter's monoisotopic pick
        int(charge) if charge is not None else pd.NA,
        ion.get("peak intensity", np.nan), np.nan, rt, 1, np.nan, rt, rt, np.nan, np.nan,
        window.get("isolation window target m/z", mz), width,
        activation.get("collision energy", np.nan),
        mobility if mobility is not None else np.nan, np.nan, np.nan,
    ]


class MzmlFrameSource:
    """Frame-at-a-time access to MS1 points and MS/MS spectra through the mzML index."""

    def __init__(self, index: MzmlIndex):
        from pyteomics import mzml

        self.index = index
        self._reader = mzml.MzML(str(index.path), read_schema=False, use_index=True)
        self._array = index.mobility.get("cv_name") \
            if index.mobility.get("representation") == "per-peak array" else None

    def read_frame(self, frame_id: int, num_scans: int):
        mz_parts, int_parts, mob_parts = [], [], []
        for sid in self.index.frame_spectra.get(int(frame_id), []):
            s = self._reader.get_by_id(sid)
            mz = np.asarray(s.get("m/z array", ()), dtype=np.float64)
            if mz.size == 0:
                continue
            intensity = np.asarray(s["intensity array"], dtype=np.float64)
            if self._array and self._array in s:
                mobility = np.asarray(s[self._array], dtype=np.float64)
            else:
                value, _ = _scalar_mobility(_scan(s))
                mobility = np.full(mz.size, np.nan if value is None else value)
            mz_parts.append(mz)
            int_parts.append(intensity)
            mob_parts.append(mobility)
        if not mz_parts:
            return None
        return np.concatenate(mz_parts), np.concatenate(int_parts), np.concatenate(mob_parts)

    def read_pasef_msms(self, precursor_ids):
        out = {}
        for pid in precursor_ids:
            sid = self.index.msms_ids.get(int(pid))
            if sid is None:
                continue
            s = self._reader.get_by_id(sid)
            out[int(pid)] = (np.asarray(s.get("m/z array", ()), dtype=np.float64),
                             np.asarray(s.get("intensity array", ()), dtype=np.float64))
        return out

    def close(self) -> None:
        self._reader.close()

