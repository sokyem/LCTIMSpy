"""Typed result objects returned by the core and serialized by the server."""

from __future__ import annotations

from dataclasses import asdict, dataclass


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
