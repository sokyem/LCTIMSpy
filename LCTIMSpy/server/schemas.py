"""Request and response models. The frontend's TypeScript types are generated from these."""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from ..core.config import MzTolerance, QueryConfig, RtWindow, StatsConfig

# ---- shared pieces ------------------------------------------------------------


class Tolerance(BaseModel):
    value: float = Field(0.02, gt=0)
    unit: Literal["Da", "ppm"] = "Da"

    def core(self) -> MzTolerance:
        return MzTolerance(self.value, self.unit)


class Target(BaseModel):
    mz: float = Field(gt=0)
    label: str | None = None


class Window(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def _ordered(self):
        if not self.end > self.start:
            raise ValueError("window end must be after start")
        return self

    def core(self) -> RtWindow:
        return RtWindow(self.start, self.end)


class Query(BaseModel):
    tolerance: Tolerance = Tolerance()
    mobilogram_sigma: float | None = Field(None, gt=0)
    baseline_correction: bool = True
    intensity_threshold: float | None = 100
    threshold_on: Literal["corrected", "raw"] = "corrected"

    def core(self) -> QueryConfig:
        return QueryConfig(self.tolerance.core(), self.mobilogram_sigma, self.baseline_correction,
                           self.intensity_threshold, self.threshold_on)


class Stats(BaseModel):
    metric: Literal["mobility_difference", "peak_mobility_difference"] = "mobility_difference"
    correction: Literal["bonferroni", "holm", "tukey", "none"] = "bonferroni"
    alpha: float = Field(0.05, gt=0, lt=1)
    min_samples: int = Field(3, ge=2)
    equal_var: bool = True

    def core(self) -> StatsConfig:
        return StatsConfig(self.metric, self.correction, self.alpha, self.min_samples,
                           self.equal_var)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as JSON-safe dicts (NaN -> None)."""
    out = df.to_dict(orient="records")
    for row in out:
        for k, v in row.items():
            if isinstance(v, float) and math.isnan(v):
                row[k] = None
    return out


# ---- system -------------------------------------------------------------------


class SdkStatusOut(BaseModel):
    available: bool
    version: str | None = None
    path: str | None = None
    error: str | None = None
    remediation: str | None = None


class CapabilitiesOut(BaseModel):
    version: str
    platform: str
    sdk: SdkStatusOut
    cache_dir: str
    browse_enabled: bool
    max_concurrent_jobs: int


class FsEntry(BaseModel):
    name: str
    path: str
    kind: Literal["dir", "d_folder", "mzml", "file"]
    size: int | None = None


class FsListing(BaseModel):
    path: str
    parent: str | None
    is_network: bool
    entries: list[FsEntry]


# ---- datasets -----------------------------------------------------------------


class DatasetAdd(BaseModel):
    path: str = Field(min_length=1)
    name: str | None = None
    group: str | None = None


class DatasetPatch(BaseModel):
    name: str | None = Field(None, min_length=1)
    group: str | None = None


class CacheStatus(BaseModel):
    status: Literal["cached", "missing"]
    key: str | None = None
    rows: int | None = None
    bytes: int | None = None
    created: float | None = None
    frames_failed: int | None = None


class DatasetOut(BaseModel):
    id: str
    name: str
    group: str | None
    path: str
    local_path: str | None
    is_network: bool
    available: bool
    meta: dict
    cache: CacheStatus


# ---- jobs ---------------------------------------------------------------------


class ExtractJobIn(BaseModel):
    dataset_id: str
    force: bool = False
    mobility_bin_width: float = Field(0.0005, gt=0)


class CopyJobIn(BaseModel):
    dataset_id: str


class JobOut(BaseModel):
    id: str
    kind: str
    dataset_id: str | None
    state: Literal["queued", "running", "cancelling", "done", "failed", "cancelled"]
    created: float
    started: float | None
    finished: float | None
    progress: dict | None
    result: dict | None
    error: dict | None
    log_tail: list[str] | None = None


# ---- analysis -----------------------------------------------------------------


class FragmentsIn(BaseModel):
    peptide: str = Field(min_length=1)
    charges: list[int] = Field([1, 2], min_length=1)
    ions: list[str] | None = None


class FragmentsOut(BaseModel):
    targets: list[Target]
    precursors: dict[str, float]


class ChromatogramsIn(BaseModel):
    dataset_ids: list[str] = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)
    tolerance: Tolerance = Tolerance()
    max_points: int | None = Field(None, ge=10, description="LTTB-downsample each series")


class SeriesOut(BaseModel):
    dataset_id: str
    target_mz: float
    label: str | None = None
    x: list[float]
    y: list[float]


class ChromatogramsOut(BaseModel):
    series: list[SeriesOut]


class PeaksIn(BaseModel):
    dataset_id: str
    targets: list[Target] = Field(min_length=1)
    tolerance: Tolerance = Tolerance()
    n: int = Field(2, ge=1, le=10)


class PeakOut(BaseModel):
    apex_rt: float
    start_rt: float
    end_rt: float
    height: float
    prominence: float


class PeaksOut(BaseModel):
    peaks: list[PeakOut]


class MobilogramsIn(BaseModel):
    dataset_ids: list[str] = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)
    windows: list[Window] = Field(min_length=1)
    query: Query = Query()


class MobilogramOut(SeriesOut):
    window: int
    rt_start: float
    rt_end: float
    max_intensity_raw: float


class MobilogramsOut(BaseModel):
    series: list[MobilogramOut]


class MobilityDiffIn(BaseModel):
    dataset_ids: list[str] = Field(min_length=1)
    targets: list[Target] = Field(min_length=1)
    windows: list[Window] = Field(min_length=2)
    query: Query = Query()
    rt_offsets: dict[str, float] = Field(default_factory=dict,
                                         description="dataset_id -> RT shift in minutes")


class MobilityDiffOut(BaseModel):
    per_window: list[dict]
    differences: list[dict]
    excluded: list[dict]


class CompareIn(MobilityDiffIn):
    stats: Stats = Stats()
    include_figure: bool = True


class CompareOut(BaseModel):
    differences: MobilityDiffOut
    pairs: list[dict]
    data: list[dict]
    figure: dict | None = None


class ExportIn(BaseModel):
    kind: Literal["chromatograms", "mobilograms", "mobility-difference", "compare"]
    format: Literal["csv", "html", "json"]
    request: dict
