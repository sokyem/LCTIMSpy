"""Write the synthetic isomer run as mzML, in the shapes real converters produce.

Styles:
    "per-scan"  one MS1 spectrum per TIMS scan, 1/K0 as a scan parameter
                (msconvert without --combineIonMobilitySpectra)
    "combined"  one MS1 spectrum per frame + "mean inverse reduced ion mobility array"
                (msconvert --combineIonMobilitySpectra)
    "drift"     combined, but drift time per spectrum (drift-tube style)
    "none"      no mobility at all (conventional LC-MS/MS)

Every 5th frame is an MS/MS spectrum of a precursor from the previous MS1
frame, as in the fake .d run.
"""

from pathlib import Path

import numpy as np

from tests.conftest import (
    ISOMERS,
    SECONDS_PER_FRAME,
    TARGET_MZ,
    TARGETS,
    FakeFrameSource,
    is_ms1,
)

SCAN_STEP = 0.001          # per-scan style: the TIMS scan grid in 1/K0


def write_fake_mzml(path: Path, style: str = "combined", n_frames: int = 120, seed: int = 0,
                    noise_points: int = 30) -> Path:
    from psims.mzml.writer import MzMLWriter

    source = FakeFrameSource(seed=seed)
    spectra = []                                    # (kwargs for write_spectrum)
    for f in range(1, n_frames + 1):
        rt_min = f * SECONDS_PER_FRAME / 60.0
        if is_ms1(f):
            mz, intensity, mobility = source.spectrum(f)
            # Keep every target-ion point, and only some noise, so files stay small.
            keep = np.zeros(len(mz), dtype=bool)
            for t in TARGETS:
                keep |= np.abs(mz - t) < 0.05
            keep[np.flatnonzero(~keep)[:noise_points]] = True
            mz, intensity, mobility = mz[keep], intensity[keep], mobility[keep]
            base = {"centroided": True, "scan_start_time": rt_min,
                    "params": [{"ms level": 1}, "MS1 spectrum", "positive scan"]}
            if style == "per-scan":
                scans = np.round(mobility / SCAN_STEP) * SCAN_STEP
                for value in np.unique(scans):
                    sel = scans == value
                    order = np.argsort(mz[sel])
                    spectra.append({**base, "mz": mz[sel][order], "intensity": intensity[sel][order],
                                    "scan_params": [{"inverse reduced ion mobility": float(value)}]})
            else:
                kw = {**base, "mz": mz, "intensity": intensity}
                if style == "combined":
                    kw["other_arrays"] = [("mean inverse reduced ion mobility array", mobility)]
                elif style == "drift":
                    kw["scan_params"] = [{"ion mobility drift time": 20.0}]
                spectra.append(kw)
        else:
            near = [mob for apex, mob in ISOMERS if abs(f - 1 - apex) <= 8]
            prec_mz = TARGET_MZ if near else 400.0 + f
            info = {"mz": prec_mz, "charge": 2, "intensity": 1e5,
                    "activation": ["collision-induced dissociation", {"collision energy": 30.0}],
                    "isolation_window": [1.0, prec_mz, 1.0]}
            if near and style in ("per-scan", "combined"):
                info["params"] = [{"inverse reduced ion mobility": near[0]}]
            frag_mz, frag_int = FakeFrameSource.msms_spectrum(f)
            spectra.append({"mz": frag_mz, "intensity": frag_int, "centroided": True,
                            "scan_start_time": rt_min,
                            "params": [{"ms level": 2}, "MSn spectrum", "positive scan"],
                            "precursor_information": info})

    with MzMLWriter(open(path, "wb"), close=True) as out:
        out.controlled_vocabularies()
        out.file_description(["MS1 spectrum", "MSn spectrum"])
        out.software_list([{"id": "psims-writer", "version": "1", "params": ["python-psims"]}])
        src = out.Source(1, ["electrospray ionization", "electrospray inlet"])
        ana = out.Analyzer(2, ["quadrupole", "time-of-flight"])
        det = out.Detector(3, ["microchannel plate detector"])
        out.instrument_configuration_list([out.InstrumentConfiguration(
            id="IC1", component_list=[src, ana, det], params=["timsTOF Pro 2"])])
        out.data_processing_list([out.DataProcessing(
            [out.ProcessingMethod(order=1, software_reference="psims-writer",
                                  params=["Conversion to mzML"])], id="DP1")])
        with out.run(id="run1", instrument_configuration="IC1"):
            with out.spectrum_list(count=len(spectra)):
                for i, kw in enumerate(spectra, start=1):
                    kw = dict(kw)
                    out.write_spectrum(kw.pop("mz"), kw.pop("intensity"), id=f"scan={i}", **kw)
    return path
