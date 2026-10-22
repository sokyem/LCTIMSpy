"""DDA-PASEF precursors and MS/MS spectra: reading, mobility conversion, caching."""

import numpy as np
import pytest

from LCTIMSpy.core.errors import CancelledError
from LCTIMSpy.core.io import extract as extract_mod
from LCTIMSpy.core.io.bruker import read_frames
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.core.io.pasef import add_mobility, read_pasef_tables, summarize_precursors
from LCTIMSpy.core.progress import CancelToken
from tests.conftest import (
    ISOMERS,
    SECONDS_PER_FRAME,
    TARGET_MZ,
    FakeFrameSource,
    Ms1OnlyFrameSource,
    fake_scan_to_mobility,
    make_fake_d,
    pasef_precursor_plan,
)

N_PRECURSORS = len(pasef_precursor_plan()[0])


@pytest.fixture
def pasef_d(tmp_path):
    return make_fake_d(tmp_path / "data", "pasef.d", pasef=True)


def test_reads_tables_and_tolerates_non_pasef_runs(pasef_d, fake_d):
    precursors, pasef = read_pasef_tables(pasef_d)
    assert len(precursors) == N_PRECURSORS and len(pasef) == N_PRECURSORS
    empty_p, empty_f = read_pasef_tables(fake_d)
    assert empty_p.empty and empty_f.empty


def test_summary_keeps_unknown_charge_as_missing(pasef_d):
    precursors, pasef = read_pasef_tables(pasef_d)
    s = summarize_precursors(precursors, pasef, read_frames(pasef_d))
    assert s["charge"].isna().sum() == N_PRECURSORS // 2          # every second precursor
    assert s["mz_monoisotopic"].isna().sum() == N_PRECURSORS // 2
    first = s.iloc[0]
    assert first["rt"] == pytest.approx(first["parent_frame"] * SECONDS_PER_FRAME / 60)
    assert first["n_msms_frames"] == 1 and first["isolation_width"] == 2.0


def test_scan_numbers_become_mobility_with_swapped_bounds(pasef_d):
    precursors, pasef = read_pasef_tables(pasef_d)
    s = add_mobility(summarize_precursors(precursors, pasef, read_frames(pasef_d)),
                     lambda frame, scans: fake_scan_to_mobility(scans))
    targets = s[np.isclose(s["mz_monoisotopic"], TARGET_MZ)]
    assert set(np.round(targets["mobility"], 4)) <= {m for _, m in ISOMERS}
    # Higher scan number = lower 1/K0, so ScanNumBegin gives the *high* bound.
    assert (s["isolation_mobility_low"] < s["mobility"]).all()
    assert (s["mobility"] < s["isolation_mobility_high"]).all()
    assert np.allclose(s["isolation_mobility_high"] - s["isolation_mobility_low"], 0.04)


def test_extraction_caches_precursors_and_msms(pasef_d, cache):
    entry = extract_to_cache(pasef_d, cache=cache, frame_source=FakeFrameSource())
    stats = entry.stats["pasef"]
    assert stats["precursors"] == N_PRECURSORS
    assert stats["msms_spectra"] == N_PRECURSORS and stats["notes"] == []
    pre = cache.load_precursors(entry)
    assert len(pre) == N_PRECURSORS and pre["mobility"].notna().all()
    some = [int(pre["precursor_id"].iloc[i]) for i in (0, 3, 7)]
    spectra = cache.load_msms(entry, some)
    assert sorted(spectra) == sorted(some)
    for pid in some:
        want_mz, want_int = FakeFrameSource.msms_spectrum(pid)
        np.testing.assert_allclose(spectra[pid][0], want_mz)
        np.testing.assert_allclose(spectra[pid][1], want_int)
    assert cache.load_msms(entry, []) == {}


def test_non_pasef_run_has_no_precursor_files(fake_d, cache):
    entry = extract_to_cache(fake_d, cache=cache, frame_source=FakeFrameSource())
    assert entry.stats["pasef"]["precursors"] == 0
    assert cache.load_precursors(entry).empty and cache.load_msms(entry, [1]) == {}


def test_source_without_sdk_capabilities_records_why(pasef_d, cache):
    entry = extract_to_cache(pasef_d, cache=cache, frame_source=Ms1OnlyFrameSource())
    stats = entry.stats["pasef"]
    assert stats["msms_spectra"] == 0
    assert any("1/K0" in n for n in stats["notes"])
    assert any("MS/MS" in n for n in stats["notes"])
    pre = cache.load_precursors(entry)
    assert len(pre) == N_PRECURSORS and pre["mobility"].isna().all()


def test_cancel_during_msms_leaves_nothing(pasef_d, cache, monkeypatch):
    monkeypatch.setattr(extract_mod, "MSMS_BATCH", 4)
    token = CancelToken()

    class CancelInMsms(FakeFrameSource):
        def read_pasef_msms(self, ids):
            token.cancel()
            return super().read_pasef_msms(ids)

    with pytest.raises(CancelledError):
        extract_to_cache(pasef_d, cache=cache, cancel=token, frame_source=CancelInMsms())
    assert not cache.root.exists() or list(cache.root.iterdir()) == []
