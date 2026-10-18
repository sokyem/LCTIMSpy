"""Lazy import, SDK probing, config objects, and SDK-free dataset metadata."""

import os
import subprocess
import sys

import pytest

from LCTIMSpy.core import sdk
from LCTIMSpy.core.config import ExtractionConfig, MzTolerance, RtWindow
from LCTIMSpy.core.errors import DatasetError, SdkUnavailableError
from LCTIMSpy.core.io.bruker import read_dataset_meta, read_frames
from LCTIMSpy.core.io.paths import is_network_path, validate_d_folder
from tests.conftest import N_FRAMES, SECONDS_PER_FRAME, is_ms1


def test_import_loads_nothing_heavy():
    """``import LCTIMSpy`` must not touch the SDK or the plotting stack.

    Run in a fresh interpreter: this process has already imported plenty.
    """
    code = (
        "import sys, LCTIMSpy\n"
        "heavy = [m for m in ('pyTDFSDK', 'plotly', 'matplotlib', 'pyarrow') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
        "assert callable(LCTIMSpy.calculate_fragment_mz)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_module_imports_without_sdk():
    import LCTIMSpy.lctims as legacy

    assert callable(legacy.extract_chromatograms)


def test_unknown_attribute_raises():
    import LCTIMSpy

    with pytest.raises(AttributeError):
        LCTIMSpy.does_not_exist  # noqa: B018


@pytest.fixture
def fresh_sdk(monkeypatch):
    monkeypatch.setattr(sdk, "_library", None)
    monkeypatch.delenv("LCTIMSPY_TIMSDATA", raising=False)
    return sdk


def test_probe_reports_missing_pytdfsdk(fresh_sdk, monkeypatch):
    monkeypatch.setattr(sdk.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sdk.importlib.util, "find_spec", lambda name: None)
    status = fresh_sdk.probe_sdk()
    assert not status.available
    assert "not installed" in status.error
    assert "[bruker]" in status.remediation


def test_probe_reports_missing_library(fresh_sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("LCTIMSPY_TIMSDATA", str(tmp_path / "nope.dll"))
    status = fresh_sdk.probe_sdk()
    assert not status.available
    assert "not found" in status.error
    with pytest.raises(SdkUnavailableError):
        fresh_sdk.load_sdk()


def test_probe_reports_unsupported_platform(fresh_sdk, monkeypatch):
    monkeypatch.setattr(sdk.platform, "system", lambda: "Darwin")
    status = fresh_sdk.probe_sdk()
    assert not status.available
    assert "Darwin" in status.error


def test_mz_tolerance_da_and_ppm():
    assert MzTolerance(0.02, "Da").window(500.0) == pytest.approx((499.98, 500.02))
    assert MzTolerance(10, "ppm").window(500.0) == pytest.approx((499.995, 500.005))
    with pytest.raises(ValueError):
        MzTolerance(0.02, "mDa")
    with pytest.raises(ValueError):
        MzTolerance(0, "Da")


def test_rt_window_parse_and_validation():
    assert RtWindow.parse("18.3:19.0") == RtWindow(18.3, 19.0)
    with pytest.raises(ValueError):
        RtWindow.parse("18.3")
    with pytest.raises(ValueError):
        RtWindow(19.0, 18.3)


def test_extraction_config_requires_binning():
    with pytest.raises(ValueError):
        ExtractionConfig(mobility_bin_width=0)
    cfg = ExtractionConfig()
    assert ExtractionConfig.from_dict(cfg.to_dict()) == cfg


def test_dataset_meta_without_sdk(fake_d):
    meta = read_dataset_meta(fake_d)
    n_ms1 = sum(is_ms1(i) for i in range(1, N_FRAMES + 1))
    assert meta.n_frames == N_FRAMES
    assert meta.n_ms1_frames == n_ms1
    assert meta.rt_min == pytest.approx(1 * SECONDS_PER_FRAME / 60)
    assert meta.rt_max == pytest.approx((N_FRAMES - 1) * SECONDS_PER_FRAME / 60)  # 120 is MS/MS
    assert (meta.mz_acq_lower, meta.mz_acq_upper) == (100.0, 1700.0)
    assert meta.instrument == "timsTOF Fake"
    assert meta.name == "sample"
    assert len(read_frames(fake_d)) == N_FRAMES


def test_validate_d_folder_errors(tmp_path, fake_d):
    with pytest.raises(DatasetError, match="does not exist"):
        validate_d_folder(tmp_path / "missing.d")
    (tmp_path / "file.d").write_text("")
    with pytest.raises(DatasetError, match="not a folder"):
        validate_d_folder(tmp_path / "file.d")
    tsf = tmp_path / "maldi.d"
    tsf.mkdir()
    (tsf / "analysis.tsf").write_text("")
    with pytest.raises(DatasetError, match="TSF"):
        validate_d_folder(tsf)
    os.remove(fake_d / "analysis.tdf_bin")
    with pytest.raises(DatasetError, match="analysis.tdf_bin"):
        validate_d_folder(fake_d)


def test_unc_paths_are_network():
    assert is_network_path(r"\\server\share\run.d")
    assert is_network_path("//server/share/run.d")
