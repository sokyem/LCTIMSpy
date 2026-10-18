"""LCTIMSpy — processing and analysis of TIMSTOF LC-TIMS-MS data.

Public names are resolved lazily (PEP 562): ``import LCTIMSpy`` imports nothing
heavy, and in particular never the Bruker SDK, so the package works on machines
without it. A name's module is imported the first time it is accessed.
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "create_tdf_data": ".lctims",
    "extract_lcms_tdf_data": ".lctims",
    "extract_chromatograms": ".lctims",
    "extract_mobilogram": ".lctims",
    "calculate_fragment_mz": ".get_fragmentMz",
    "calculate_precursor_mz": ".get_fragmentMz",
    "normalize_series": ".plot_chromatograms_mobilograms",
    "plot_chromatograms": ".plot_chromatograms_mobilograms",
    "plot_interactive_chromatograms": ".plot_chromatograms_mobilograms",
    "plot_mobilogram_target_mz": ".plot_chromatograms_mobilograms",
    "plot_mobilogram_interactive": ".plot_chromatograms_mobilograms",
    "process_mobilogram_plots": ".plot_chromatograms_mobilograms",
    "process_mobilogram_interactive_plots": ".plot_chromatograms_mobilograms",
    "extract_and_calculate_mobility_difference": ".mobility_analysis",
    "extract_and_calculate_mobility_difference_multi": ".mobility_analysis",
    "compare_targetmz_differences_with_boxplot": ".mobility_analysis",
    # Phase 1 core API
    "ExtractionConfig": ".core.config",
    "MzTolerance": ".core.config",
    "RtWindow": ".core.config",
    "DatasetMeta": ".core.results",
    "probe_sdk": ".core.sdk",
    "Cache": ".core.io.cache",
    "read_dataset_meta": ".core.io.bruker",
    "extract_to_cache": ".core.io.extract",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
