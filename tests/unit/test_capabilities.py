"""Dataset capability reports: acquisition mode, mobility coordinate, CCS validity."""

from LCTIMSpy.discovery.capabilities import inspect_tdf
from tests.conftest import make_fake_d


def test_dda_pasef_with_calibration_is_ccs_convertible(tmp_path):
    d = make_fake_d(tmp_path, "std.d", pasef=True, calibration=True)
    caps = inspect_tdf(d, sdk_available=True)
    assert caps.source_format == "bruker-tdf"
    assert caps.acquisition == "DDA-PASEF"
    assert caps.has_msms and caps.msms_precursor_info
    assert caps.spectrum_counts == {"MS1": 96, "DDA-PASEF MS/MS": 24}
    m = caps.mobility
    assert m.present and m.coordinate == "1/K0" and m.unit == "V·s/cm²"
    assert m.acquisition_range == (0.6, 1.4)
    assert m.ccs == "convertible" and "derived" in m.ccs_detail
    assert caps.warnings == []


def test_ccs_is_never_claimed_without_sdk_or_calibration(tmp_path):
    d = make_fake_d(tmp_path, "nocal.d", pasef=True)
    assert inspect_tdf(d, sdk_available=True).mobility.ccs == "unreliable"
    caps = inspect_tdf(make_fake_d(tmp_path, "cal.d", pasef=True, calibration=True),
                       sdk_available=False)
    assert caps.mobility.ccs == "unavailable"
    assert caps.mobility.present        # cached 1/K0 stays usable


def test_custom_tims_range_is_flagged(tmp_path):
    d = make_fake_d(tmp_path, "cidtims.d", pasef=True, calibration=True, mobility_range=(0.3, 1.3))
    caps = inspect_tdf(d, sdk_available=True)
    assert caps.mobility.ccs == "unreliable"
    assert "CID-TIMS" in caps.mobility.ccs_detail
    assert any("non-standard" in w for w in caps.warnings)


def test_pasef_frames_without_precursor_tables_are_warned(tmp_path):
    caps = inspect_tdf(make_fake_d(tmp_path, "broken.d"), sdk_available=True)
    assert caps.acquisition == "DDA-PASEF" and not caps.msms_precursor_info
    assert any("cannot be linked" in w for w in caps.warnings)
