"""Legacy location of the fragment math; the implementation lives in ``core.fragments``."""

from .core.fragments import calculate_fragment_mz, calculate_precursor_mz

__all__ = ["calculate_fragment_mz", "calculate_precursor_mz"]
