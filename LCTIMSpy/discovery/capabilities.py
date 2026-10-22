"""What a dataset can actually support, determined from its metadata -- never assumed.

Mobility coordinates are not interchangeable: Bruker TIMS reports inverse
reduced mobility 1/K0 (V·s/cm²) derived from a scan number through the run's
calibration; drift-tube instruments report drift time; FAIMS reports a
compensation voltage; CCS (Å²) is a *derived* quantity that needs a valid
calibration and the Mason–Schamp relation. The report records which one a
dataset has, where it came from, and whether CCS can be derived from it.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.io.bruker import _connect, read_frames, read_global_metadata
from ..core.io.paths import validate_d_folder

# Bruker Frames.MsMsType codes.
MSMS_TYPES = {0: "MS1", 2: "MRM", 8: "DDA-PASEF MS/MS", 9: "diaPASEF MS/MS"}
# Standard timsTOF proteomics methods start the 1/K0 ramp around 0.6-0.7.
# A much lower start is typical of custom TIMS tuning (e.g. CID-TIMS, where
# ions are fragmented before mobility separation) for which a standard CCS
# calibration does not hold.
NONSTANDARD_MOBILITY_LOW = 0.45


@dataclass(frozen=True)
class MobilityInfo:
    present: bool
    coordinate: str | None = None          # e.g. "1/K0"
    unit: str | None = None                # e.g. "V·s/cm²"
    source: str | None = None              # how the values are obtained
    acquisition_range: tuple[float, float] | None = None
    calibration: str | None = None         # what calibration information exists
    ccs: str = "unavailable"               # "convertible" | "unreliable" | "unavailable"
    ccs_detail: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Capabilities:
    source_format: str                     # "bruker-tdf" | "mzml"
    acquisition: str                       # "MS1 only" | "DDA-PASEF" | "diaPASEF" | "mixed" | ...
    spectrum_counts: dict[str, int]
    has_msms: bool
    msms_precursor_info: bool              # precursor m/z, charge, isolation known per MS/MS
    mobility: MobilityInfo
    mz_range: tuple[float, float] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _float(meta: dict, key: str) -> float | None:
    try:
        return float(meta[key])
    except (KeyError, TypeError, ValueError):
        return None


def _tables(d: Path) -> set[str]:
    with closing(_connect(d / "analysis.tdf")) as conn:
        try:
            return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        except sqlite3.DatabaseError:
            return set()


def _calibration_rows(d: Path) -> int:
    with closing(_connect(d / "analysis.tdf")) as conn:
        try:
            return int(conn.execute("SELECT COUNT(*) FROM TimsCalibration").fetchone()[0])
        except sqlite3.DatabaseError:
            return 0


def inspect(path: str | Path, sdk_available: bool | None = None) -> Capabilities:
    """Capabilities of a ``.d`` folder or an mzML file."""
    from ..core.io.mzml import is_mzml

    return inspect_mzml(path) if is_mzml(path) else inspect_tdf(path, sdk_available)


def inspect_mzml(path: str | Path, index=None) -> Capabilities:
    """Capabilities of an mzML file, from one metadata pass (or a prebuilt index)."""
    from ..core.io.mzml import index_mzml

    index = index or index_mzml(path)
    m = index.mobility
    counts = {"MS1": index.counts["MS1"], "MS2": index.counts["MS2"]}
    if index.counts.get("other"):
        counts["MSn (n>2)"] = index.counts["other"]
    acquisition = ("MS1 only" if not index.counts["MS2"]
                   else "DDA/MS2" if not index.precursors["isolation_width"].gt(20).any()
                   else "MS2 with wide isolation (DIA-like)")
    warnings = []
    if acquisition.startswith("MS2 with wide"):
        warnings.append("isolation windows wider than 20 m/z: MS/MS spectra may be multiplexed")
    if m["present"] and m["coordinate"] == "ion mobility (unspecified)":
        warnings.append("the mobility array does not say which coordinate it holds; "
                        "values are used as-is and cannot be compared across instruments")
    mobility = MobilityInfo(
        present=bool(m["present"]), coordinate=m["coordinate"], unit=m["unit"],
        source=(f"mzML {m['representation']} ({m['cv_name']})" if m["present"] else None),
        acquisition_range=None, calibration="not recorded in mzML",
        ccs="unavailable",
        ccs_detail="mzML does not carry the TIMS calibration needed to derive CCS",
        notes=list(index.notes),
    )
    return Capabilities(
        source_format="mzml", acquisition=acquisition, spectrum_counts=counts,
        has_msms=index.counts["MS2"] > 0,
        msms_precursor_info=bool(len(index.precursors))
        and index.precursors["mz_largest"].notna().all(),
        mobility=mobility, mz_range=index.mz_range, warnings=warnings,
    )


def inspect_tdf(d_path: str | Path, sdk_available: bool | None = None) -> Capabilities:
    """Capabilities of a Bruker timsTOF ``.d`` folder, from ``analysis.tdf`` alone."""
    from ..core.sdk import probe_sdk

    d = validate_d_folder(d_path)
    frames = read_frames(d)
    meta = read_global_metadata(d)
    tables = _tables(d)
    counts = {MSMS_TYPES.get(int(k), f"MsMsType {int(k)}"): int(v)
              for k, v in frames["MsMsType"].value_counts().sort_index().items()}
    kinds = set(frames["MsMsType"].astype(int))
    warnings: list[str] = []

    if kinds <= {0}:
        acquisition = "MS1 only"
    elif kinds <= {0, 8}:
        acquisition = "DDA-PASEF"
    elif kinds <= {0, 9}:
        acquisition = "diaPASEF"
    else:
        acquisition = "mixed"
        warnings.append(f"unusual mix of frame types: {counts}")
    has_msms = bool(kinds - {0})
    precursor_info = {"Precursors", "PasefFrameMsMsInfo"} <= tables
    if 8 in kinds and not precursor_info:
        warnings.append("DDA-PASEF frames without Precursors/PasefFrameMsMsInfo tables; "
                        "MS/MS cannot be linked to precursors")
    if 9 in kinds:
        warnings.append("diaPASEF: MS/MS spectra are multiplexed; precursor-specific MS/MS "
                        "association needs deconvolution and is not supported yet")

    lo = _float(meta, "OneOverK0AcqRangeLower")
    hi = _float(meta, "OneOverK0AcqRangeUpper")
    acq_range = (lo, hi) if lo is not None and hi is not None else None
    cal_rows = _calibration_rows(d) if "TimsCalibration" in tables else 0
    calibration = (f"TimsCalibration table ({cal_rows} entr{'y' if cal_rows == 1 else 'ies'})"
                   if cal_rows else "no TimsCalibration entries found")
    if sdk_available is None:
        sdk_available = probe_sdk().available

    notes = ["1/K0 is computed from TIMS scan numbers with the run's own calibration; the "
             "measured coordinate is kept as 1/K0 and never relabelled as CCS"]
    if not sdk_available:
        ccs, ccs_detail = "unavailable", ("the Bruker TDF-SDK is not installed, so neither "
                                          "scan->1/K0 nor 1/K0->CCS conversion is available")
        notes.append("mobility of cached datasets remains usable; new extraction needs the SDK")
    elif not cal_rows:
        ccs, ccs_detail = "unreliable", "no TIMS calibration recorded in the run"
    elif lo is not None and lo < NONSTANDARD_MOBILITY_LOW:
        ccs, ccs_detail = "unreliable", (
            f"1/K0 acquisition range starts at {lo:g} V·s/cm², typical of custom TIMS tuning "
            "(e.g. CID-TIMS); a standard CCS calibration should not be assumed")
        warnings.append("non-standard TIMS mobility range; treat 1/K0 as run-specific")
    else:
        ccs, ccs_detail = "convertible", (
            "via the Bruker TDF-SDK (Mason–Schamp, N2 drift gas, the run's calibration); "
            "values are derived, not measured")

    mobility = MobilityInfo(
        present=True, coordinate="1/K0", unit="V·s/cm²",
        source="TIMS scan number converted with the run calibration (Bruker TDF-SDK)",
        acquisition_range=acq_range, calibration=calibration, ccs=ccs, ccs_detail=ccs_detail,
        notes=notes,
    )
    mz_lo, mz_hi = _float(meta, "MzAcqRangeLower"), _float(meta, "MzAcqRangeUpper")
    return Capabilities(
        source_format="bruker-tdf", acquisition=acquisition, spectrum_counts=counts,
        has_msms=has_msms, msms_precursor_info=precursor_info, mobility=mobility,
        mz_range=(mz_lo, mz_hi) if mz_lo is not None and mz_hi is not None else None,
        warnings=warnings,
    )
