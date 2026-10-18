"""Typed configuration objects.

``ExtractionConfig`` is part of the cache key: two extractions share a cache
entry only if every field here is equal. The target m/z list is deliberately
*not* part of it -- extraction is always full, so one cache serves any target
chosen later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class ExtractionConfig:
    # Mobility values are rounded to the nearest multiple of this width. The
    # legacy default was None (no binning), which made the per-mobility groupby
    # in mobilogram extraction aggregate almost nothing. 0.0005 is the value
    # the README and notebook already used.
    mobility_bin_width: float = 0.0005
    # Keep only MS1 frames (MsMsType == 0), as the legacy extractor did.
    ms1_only: bool = True

    def __post_init__(self) -> None:
        if not self.mobility_bin_width > 0:
            raise ValueError(f"mobility_bin_width must be > 0, got {self.mobility_bin_width!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionConfig:
        return cls(**data)


@dataclass(frozen=True)
class MzTolerance:
    """One m/z tolerance for the whole app, in Da or ppm.

    Replaces the five different ``mz_tolerance`` defaults (0.01, 0.02, 0.05,
    0.1) the legacy functions used for the same concept.
    """

    value: float = 0.02
    unit: Literal["Da", "ppm"] = "Da"

    def __post_init__(self) -> None:
        if self.unit not in ("Da", "ppm"):
            raise ValueError(f"unit must be 'Da' or 'ppm', got {self.unit!r}")
        if not self.value > 0:
            raise ValueError(f"tolerance must be > 0, got {self.value!r}")

    def half_width(self, mz: float) -> float:
        return self.value if self.unit == "Da" else mz * self.value * 1e-6

    def window(self, mz: float) -> tuple[float, float]:
        h = self.half_width(mz)
        return mz - h, mz + h


@dataclass(frozen=True)
class RtWindow:
    """A retention-time window in minutes, inclusive at both ends."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not self.end > self.start:
            raise ValueError(f"RT window end ({self.end}) must be after start ({self.start})")

    @classmethod
    def parse(cls, text: str) -> RtWindow:
        """Parse ``"18.3:19.0"`` as used on the command line."""
        start, sep, end = text.partition(":")
        if not sep:
            raise ValueError(f"expected START:END, got {text!r}")
        return cls(float(start), float(end))


@dataclass(frozen=True)
class QueryConfig:
    """How targets are matched and mobilograms are built, shared by every query."""

    tolerance: MzTolerance = MzTolerance()
    # Gaussian sigma (in mobility bins) applied to mobilograms; None = no smoothing.
    mobilogram_sigma: float | None = None
    baseline_correction: bool = True
    # A target is dropped for a sample unless every window's max intensity
    # reaches this. None disables the check.
    intensity_threshold: float | None = 100
    # Which maximum the threshold is compared with. The legacy code compared the
    # baseline-corrected maximum, so the threshold's meaning changed when
    # baseline_correction was toggled. "corrected" preserves the published
    # behaviour; "raw" compares the maximum before baseline subtraction.
    threshold_on: Literal["corrected", "raw"] = "corrected"

    def __post_init__(self) -> None:
        if self.threshold_on not in ("corrected", "raw"):
            raise ValueError(f"threshold_on must be 'corrected' or 'raw', got {self.threshold_on!r}")
        if self.mobilogram_sigma is not None and not self.mobilogram_sigma > 0:
            raise ValueError(f"mobilogram_sigma must be > 0 or None, got {self.mobilogram_sigma!r}")


@dataclass(frozen=True)
class StatsConfig:
    metric: Literal["mobility_difference", "peak_mobility_difference"] = "mobility_difference"
    correction: Literal["bonferroni", "holm", "tukey", "none"] = "bonferroni"
    alpha: float = 0.05
    min_samples: int = 3
    # The legacy code called this a Welch test but ran Student's t-test
    # (scipy's default equal_var=True). Kept as the default so published
    # p-values reproduce; set False for Welch.
    equal_var: bool = True

    def __post_init__(self) -> None:
        if self.correction not in ("bonferroni", "holm", "tukey", "none"):
            raise ValueError(f"unsupported correction {self.correction!r}")
