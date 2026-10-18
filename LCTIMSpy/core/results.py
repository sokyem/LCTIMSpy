"""Typed result objects returned by the core and serialized by the server."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetMeta:
    """What is known about a ``.d`` folder from ``analysis.tdf`` alone.

    Read without the SDK in a few hundred ms even for a 50 GB dataset, so the UI
    can show RT bounds before anything expensive runs.
    """

    source_path: str
    name: str
    n_frames: int
    n_ms1_frames: int
    rt_min: float                      # minutes
    rt_max: float                      # minutes
    mz_acq_lower: float | None = None
    mz_acq_upper: float | None = None
    mobility_acq_lower: float | None = None
    mobility_acq_upper: float | None = None
    instrument: str | None = None
    acquisition_software: str | None = None
    acquisition_datetime: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetMeta:
        return cls(**data)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ChromatogramSeries:
    """Extracted-ion chromatogram: summed intensity per MS1 frame for one target."""

    target_mz: float
    retention_time: np.ndarray        # minutes, ascending
    intensity: np.ndarray
    label: str | None = None          # e.g. "y3 (1+)"

    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class MobilogramSeries:
    """Intensity per mobility bin for one target inside one RT window."""

    target_mz: float
    window: int                       # index into the RT window list
    rt_start: float
    rt_end: float
    mobility: np.ndarray              # 1/K0, ascending
    intensity: np.ndarray             # after smoothing and baseline correction
    max_intensity_raw: float          # before baseline correction

    def to_dict(self) -> dict:
        return {k: _jsonable(v) for k, v in self.__dict__.items()}


@dataclass(frozen=True)
class MobilityDiffResult:
    """Mobility statistics per window, and the differences between windows.

    per_window:  sample_id, target_mz, window, rt_start, rt_end, avg_mobility,
                 peak_mobility, max_intensity, max_intensity_raw, total_intensity, n_bins
    differences: sample_id, target_mz, window_a, window_b, avg_mobility_a,
                 avg_mobility_b, mobility_difference, peak_mobility_a,
                 peak_mobility_b, peak_mobility_difference
    excluded:    sample_id, target_mz, reason
    """

    per_window: pd.DataFrame
    differences: pd.DataFrame
    excluded: pd.DataFrame

    def to_legacy_frame(self) -> pd.DataFrame:
        """The two-window wide format of ``extract_and_calculate_mobility_difference``."""
        d = self.differences[(self.differences.window_a == 0) & (self.differences.window_b == 1)]
        return d.rename(columns={
            "avg_mobility_a": "avg_mobility_rt_range1",
            "avg_mobility_b": "avg_mobility_rt_range2",
            "peak_mobility_a": "peak_mobility_rt_range1",
            "peak_mobility_b": "peak_mobility_rt_range2",
        })[["sample_id", "target_mz", "avg_mobility_rt_range1", "avg_mobility_rt_range2",
            "mobility_difference", "peak_mobility_rt_range1", "peak_mobility_rt_range2",
            "peak_mobility_difference"]].reset_index(drop=True)


@dataclass(frozen=True)
class StatsResult:
    """Pairwise comparison of a metric between targets, across samples.

    data:  the rows that entered the test (targets with >= min_samples samples)
    pairs: group_a, group_b, n_a, n_b, statistic, raw_p_value, adjusted_p_value, reject
    """

    metric: str
    correction: str
    alpha: float
    data: pd.DataFrame
    pairs: pd.DataFrame
