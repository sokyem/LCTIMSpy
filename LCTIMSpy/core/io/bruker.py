"""Reading Bruker timsTOF ``.d`` folders.

Two levels of access:

* **Metadata** (``read_frames``, ``read_dataset_meta``) reads ``analysis.tdf``
  with plain sqlite3. No SDK needed, and fast even on a 50 GB dataset.
* **Spectra** (``open_frame_source``) goes through the TDF-SDK and yields a
  ``FrameSource``. The context manager closes the native handle on exit; the
  legacy code never closed it, so ``.d`` files stayed locked until GC.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..errors import DatasetError
from ..results import DatasetMeta
from .paths import validate_d_folder

FRAME_COLUMNS = ("Id", "Time", "Polarity", "MsMsType", "NumScans")

Spectrum = tuple[np.ndarray, np.ndarray, np.ndarray]   # mz, intensity, mobility (1/K0)


def _connect(tdf: Path) -> sqlite3.Connection:
    # Read-only so instrument drives and read-only shares work. sqlite rejects
    # URIs with an authority, so UNC paths fall back to a plain connect.
    try:
        return sqlite3.connect(f"{tdf.as_uri()}?mode=ro", uri=True)
    except (sqlite3.OperationalError, ValueError):
        return sqlite3.connect(str(tdf))


def read_frames(d_path: str | Path) -> pd.DataFrame:
    """The ``Frames`` table: one row per frame with RT in seconds (``Time``)."""
    d = validate_d_folder(d_path)
    try:
        with closing(_connect(d / "analysis.tdf")) as conn:
            return pd.read_sql_query(
                f"SELECT {', '.join(FRAME_COLUMNS)} FROM Frames ORDER BY Id", conn
            )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError) as exc:
        raise DatasetError(f"Could not read the Frames table of {d}: {exc}") from exc


def read_global_metadata(d_path: str | Path) -> dict[str, str]:
    d = validate_d_folder(d_path)
    try:
        with closing(_connect(d / "analysis.tdf")) as conn:
            return dict(conn.execute("SELECT Key, Value FROM GlobalMetadata").fetchall())
    except sqlite3.DatabaseError:
        return {}


def _float_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_dataset_meta(d_path: str | Path) -> DatasetMeta:
    d = validate_d_folder(d_path)
    frames = read_frames(d)
    if frames.empty:
        raise DatasetError(f"{d} contains no frames.")
    meta = read_global_metadata(d)
    ms1 = frames[frames["MsMsType"] == 0]
    rt = (ms1 if not ms1.empty else frames)["Time"] / 60.0
    return DatasetMeta(
        source_path=str(d),
        name=d.stem,
        n_frames=len(frames),
        n_ms1_frames=len(ms1),
        rt_min=float(rt.min()),
        rt_max=float(rt.max()),
        mz_acq_lower=_float_or_none(meta.get("MzAcqRangeLower")),
        mz_acq_upper=_float_or_none(meta.get("MzAcqRangeUpper")),
        mobility_acq_lower=_float_or_none(meta.get("OneOverK0AcqRangeLower")),
        mobility_acq_upper=_float_or_none(meta.get("OneOverK0AcqRangeUpper")),
        instrument=meta.get("InstrumentName"),
        acquisition_software=meta.get("AcquisitionSoftware"),
        acquisition_datetime=meta.get("AcquisitionDateTime"),
    )


class FrameSource(Protocol):
    """Reads the 3D spectrum of one frame. Implemented by the SDK and by test fakes."""

    def read_frame(self, frame_id: int, num_scans: int) -> Spectrum | None: ...


class BrukerFrameSource:
    def __init__(self, tdf_data):
        self._tdf = tdf_data
        from pyTDFSDK.tims import extract_3d_tdf_spectrum

        self._extract = extract_3d_tdf_spectrum

    def read_frame(self, frame_id: int, num_scans: int) -> Spectrum | None:
        mz, intensity, mobility = self._extract(self._tdf, int(frame_id), 0, int(num_scans))
        if mz is None or len(mz) == 0:
            return None
        return mz, intensity, mobility

    def close(self) -> None:
        tdf = self._tdf
        if getattr(tdf, "handle", None) is not None:
            tdf.api.tims_close(tdf.handle)
            # TdfData.__del__ closes again if handle is set; clear it so it won't.
            tdf.handle = None


@contextmanager
def open_frame_source(d_path: str | Path) -> Iterator[BrukerFrameSource]:
    """Open a ``.d`` through the TDF-SDK. Raises SdkUnavailableError without it."""
    from ..sdk import load_sdk

    d = validate_d_folder(d_path)
    sdk = load_sdk()
    from pyTDFSDK.classes import TdfData
    from pyTDFSDK.ctypes_data_structures import PressureCompensationStrategy

    try:
        tdf = TdfData(
            bruker_d_folder_name=str(d),
            tdf_sdk=sdk,
            pressure_compensation_strategy=PressureCompensationStrategy.PerFramePressureCompensation,
        )
    except Exception as exc:
        raise DatasetError(f"TDF-SDK could not open {d}: {exc}") from exc
    source = BrukerFrameSource(tdf)
    try:
        yield source
    finally:
        source.close()
