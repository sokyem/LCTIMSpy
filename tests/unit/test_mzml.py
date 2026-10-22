"""mzML input: mobility representations, frames, precursors, and the full pipeline."""

import numpy as np
import pytest

from LCTIMSpy.core.chrom import chromatograms
from LCTIMSpy.core.config import MzTolerance, QueryConfig, RtWindow
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.core.io.mzml import index_mzml
from LCTIMSpy.core.mobility_diff import mobility_difference
from LCTIMSpy.discovery.capabilities import inspect
from tests.conftest import ISOMERS, TARGETS, WINDOWS, FakeFrameSource
from tests.mzml_fixtures import write_fake_mzml

N_MS1, N_MS2 = 96, 24
SHIFT = ISOMERS[1][1] - ISOMERS[0][1]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    root = tmp_path_factory.mktemp("mzml")
    return {style: write_fake_mzml(root / f"{style}.mzML", style)
            for style in ("per-scan", "combined", "drift", "none")}


@pytest.mark.parametrize("style,representation,coordinate", [
    ("per-scan", "per-spectrum scan parameter", "1/K0"),
    ("combined", "per-peak array", "1/K0"),
    ("drift", "per-spectrum scan parameter", "drift time"),
    ("none", "none", None),
])
def test_mobility_representation_is_detected_not_assumed(files, style, representation,
                                                         coordinate):
    index = index_mzml(files[style])
    assert index.mobility["representation"] == representation
    assert index.mobility["coordinate"] == coordinate
    assert len(index.frames) == N_MS1                       # per-scan spectra grouped by RT
    assert index.counts["MS2"] == N_MS2


def test_per_scan_spectra_are_grouped_into_frames(files):
    index = index_mzml(files["per-scan"])
    assert index.counts["MS1"] > N_MS1                     # many spectra per frame
    assert sum(len(v) for v in index.frame_spectra.values()) == index.counts["MS1"]


def test_precursors_from_ms2(files):
    pre = index_mzml(files["combined"]).precursors
    assert len(pre) == N_MS2
    assert (pre["charge"] == 2).all() and np.allclose(pre["isolation_width"], 2.0)
    targeted = pre[np.isclose(pre["mz_largest"], TARGETS[0])]
    assert len(targeted) > 0 and targeted["mobility"].notna().all()
    assert pre.loc[~np.isclose(pre["mz_largest"], TARGETS[0]), "mobility"].isna().all()


def test_capabilities_for_mzml(files):
    caps = inspect(files["combined"])
    assert caps.source_format == "mzml" and caps.mobility.present
    assert caps.mobility.coordinate == "1/K0" and caps.mobility.ccs == "unavailable"
    assert "calibration" in caps.mobility.ccs_detail
    none = inspect(files["none"])
    assert not none.mobility.present
    assert any("no ion mobility" in n for n in none.mobility.notes)


@pytest.mark.parametrize("style", ["combined", "per-scan"])
def test_full_pipeline_recovers_the_isomer_shift(files, cache, style):
    entry = extract_to_cache(files[style], cache=cache)
    assert entry.mobility["coordinate"] == "1/K0" and entry.mobility["binned"]
    result = mobility_difference(cache, {"run": entry}, TARGETS, [RtWindow(*w) for w in WINDOWS],
                                 QueryConfig(tolerance=MzTolerance(0.02)))
    first = result.differences[result.differences.target_mz == TARGETS[0]]
    assert len(first) == 1
    assert first.iloc[0].mobility_difference == pytest.approx(SHIFT, abs=0.002)


def test_msms_spectra_are_cached(files, cache):
    entry = extract_to_cache(files["combined"], cache=cache)
    assert entry.stats["pasef"]["msms_spectra"] == N_MS2
    spectra = cache.load_msms(entry, [1, 2])
    for pid in (1, 2):
        want_mz, want_int = FakeFrameSource.msms_spectrum(5 * pid)   # MS/MS frame 5, 10, ...
        np.testing.assert_allclose(spectra[pid][0], want_mz)
        np.testing.assert_allclose(spectra[pid][1], want_int)


def test_no_mobility_file_still_gives_chromatograms(files, cache):
    entry = extract_to_cache(files["none"], cache=cache)
    assert not entry.mobility["present"] and not entry.mobility["binned"]
    eics = chromatograms(cache, entry, TARGETS, MzTolerance(0.02))
    assert all(s.intensity.max() > 0 for s in eics)
    pts = cache.load_mz_window(entry, [(TARGETS[0] - 0.02, TARGETS[0] + 0.02)])
    assert pts["mobility"].isna().all()


def test_drift_time_is_kept_as_measured(files, cache):
    entry = extract_to_cache(files["drift"], cache=cache)
    assert entry.mobility["coordinate"] == "drift time" and not entry.mobility["binned"]
    pts = cache.load_mz_window(entry, [(TARGETS[0] - 0.02, TARGETS[0] + 0.02)])
    assert np.allclose(pts["mobility"], 20.0)               # not rounded to a 1/K0 bin grid
