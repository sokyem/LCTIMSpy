"""Phase 5: discovery through the API, with the run in a real worker process."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from LCTIMSpy.core.io.cache import Cache
from LCTIMSpy.core.io.extract import extract_to_cache
from LCTIMSpy.discovery.synthetic import Scenario, SyntheticFrameSource, write_synthetic_d
from LCTIMSpy.server.app import create_app
from LCTIMSpy.server.settings import ServerSettings
from tests.unit.test_msms import SPECIES as ALL_SPECIES

REPO = Path(__file__).resolve().parents[2]
TOKEN = "t"
AUTH = {"X-LCTIMS-Token": TOKEN}
SPECIES = [s for s in ALL_SPECIES if s.name in ("H1", "H1'", "H3", "H3'")]


def write_peaks_csv(path: Path, species, run="disc.d", seed=0) -> Path:
    """PSMs in the PEAKS export layout PepSIFinder reads."""
    rng = np.random.default_rng(seed)
    rows = []
    for sp in species:
        for _ in range(3):
            mob = sp.mobility + rng.normal(0, 0.001)
            rows.append({"Peptide": sp.sequence, "-10lgP": 80.0, "Mass": sp.neutral_mass,
                         "ppm": 0.5, "m/z": sp.mono_mz, "z": sp.charge,
                         "RT": sp.rt + rng.normal(0, 0.02), "Scan": "F1:1", "Source File": run,
                         "1/k0 Range": f"{mob - 0.005:.4f}-{mob + 0.005:.4f}",
                         "Accession": "P_X"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def wait_job(c, job_id, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = c.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.2)
    pytest.fail("discovery job did not finish")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("discovery_api")
    mp = pytest.MonkeyPatch()
    mp.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    settings = ServerSettings(cache_dir=root / "cache", state_dir=root / "state",
                              scratch_dir=root / "scratch", token=TOKEN)
    scenario = Scenario(SPECIES, n_frames=500, seconds_per_frame=1.0, msms_every=3,
                        isolation_scans=30)
    d = write_synthetic_d(root / "data", "disc.d", scenario)
    extract_to_cache(d, cache=Cache(settings.cache_dir), frame_source=SyntheticFrameSource(scenario))
    undone = write_synthetic_d(root / "data", "fresh.d", scenario)
    psm_path = write_peaks_csv(root / "db.psms.csv", SPECIES)
    with TestClient(create_app(settings), headers=AUTH) as c:
        ds = c.post("/api/datasets", json={"path": str(d)}).json()
        fresh = c.post("/api/datasets", json={"path": str(undone)}).json()
        yield c, ds, fresh, psm_path
    mp.undo()


def test_psm_preview(client):
    c, _, _, psm_path = client
    r = c.post("/api/discovery/psm-preview", json={"path": str(psm_path)}).json()
    assert r["engine"] == "peaks" and r["total"] == 12 and r["with_mobility"] == 12
    assert r["runs"] == [{"run": "disc.d", "psms": 12}]


def test_run_requires_an_extracted_dataset(client):
    c, _, fresh, _ = client
    r = c.post("/api/discovery/runs", json={"dataset_id": fresh["id"]})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "not_extracted"


@pytest.fixture(scope="module")
def run(client):
    c, ds, _, psm_path = client
    job = c.post("/api/discovery/runs", json={
        "dataset_id": ds["id"], "psm_path": str(psm_path),
        "overrides": {"mobility_precision_rel": 0.003}}).json()
    done = wait_job(c, job["id"])
    assert done["state"] == "done", done
    return done["result"]["run_id"]


def test_discovery_run_end_to_end(client, run):
    c, ds, _, _ = client
    runs = c.get("/api/discovery/runs").json()
    assert [r["id"] for r in runs] == [run]
    body = c.get(f"/api/discovery/runs/{run}").json()
    meta = body["meta"]
    assert meta["psms"]["engine"] == "peaks" and meta["psms"]["used"] == 12
    assert "12 of 12 PSMs matched" in meta["notes"][0]
    # The user's precision replaced the placeholder.
    assert not meta["config"]["candidates"]["trace"]["mobility"]["precision_is_placeholder"]
    cands = body["candidates"]
    assert all("msms" not in c_ for c_ in cands)                    # heavy data left out
    by_mass = {round(x["neutral_mass"], 1): x for x in cands if x["species_b"]}
    assert by_mass[1200.6]["category"] == "High-confidence putative candidate"
    assert by_mass[1500.7]["classification"] == "B: IM-separated"
    assert meta["counts"]["candidates"] >= 2


def test_candidate_detail_for_the_viewer(client, run):
    c, *_ = client
    cands = c.get(f"/api/discovery/runs/{run}").json()["candidates"]
    target = next(x for x in cands if round(x["neutral_mass"], 1) == 1200.6 and x["species_b"])
    d = c.get(f"/api/discovery/runs/{run}/candidates/{target['candidate_id']}").json()
    detail = d["detail"]
    assert len(detail["chromatogram"]["rt"]) > 50
    assert [m["species"] for m in detail["mobilograms"]] == ["A", "B"]
    assert detail["map"] and len(detail["map"]["z"]) > 10
    assert [s["species"] for s in detail["ms1"]] == ["A", "B"]
    assert detail["mirror"]["matches"]
    assert "averagine isotope pattern" in detail["labels"]["calculated"]
    assert d["msms"]["similarity"]["cosine"] > 0.9


def test_reports(client, run):
    c, *_ = client
    csv = c.get(f"/api/discovery/runs/{run}/report", params={"format": "csv"})
    assert csv.status_code == 200 and csv.text.startswith("candidate_id,status,classification")
    html = c.get(f"/api/discovery/runs/{run}/report", params={"format": "html"}).text
    assert "Putative isomer candidates" in html and "not molar abundances" in html
    js = c.get(f"/api/discovery/runs/{run}/report", params={"format": "json"}).json()
    assert js["meta"]["counts"]["total"] == len(js["candidates"])


def test_delete_run(client, run):
    c, *_ = client
    assert c.delete(f"/api/discovery/runs/{run}").status_code == 204
    assert c.get(f"/api/discovery/runs/{run}").status_code == 404
