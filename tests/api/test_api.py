"""Phase 3: the FastAPI server, with real worker subprocesses on synthetic data.

Extraction jobs run ``python -m LCTIMSpy.server.worker.child`` exactly as in
production; the only test hook is a fake frame source, which the worker accepts
only when ``LCTIMSPY_TESTING=1``.
"""

import asyncio
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from LCTIMSpy.server.app import create_app
from LCTIMSpy.server.events import EventBus
from LCTIMSpy.server.runner import bind_loopback, make_server
from LCTIMSpy.server.settings import ServerSettings
from tests.conftest import TARGETS, WINDOWS, make_fake_d

REPO = Path(__file__).resolve().parents[2]
TOKEN = "test-token"
AUTH = {"X-LCTIMS-Token": TOKEN}
FAKE = "tests.conftest:FakeFrameSource"
SLOW = "tests.conftest:SlowFakeFrameSource"
CRASH = "tests.conftest:CrashingFrameSource"


@pytest.fixture(autouse=True)
def worker_env(monkeypatch):
    monkeypatch.setenv("LCTIMSPY_TESTING", "1")
    monkeypatch.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))


def make_settings(root: Path, source: str | None = FAKE, **kwargs) -> ServerSettings:
    extra = {"frame_source": source} if source else {}
    extra.update(kwargs.pop("extra", {}))
    return ServerSettings(cache_dir=root / "cache", state_dir=root / "state",
                          scratch_dir=root / "scratch", token=TOKEN, cancel_grace_s=5,
                          worker_spec_extra=extra, **kwargs)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path)), headers=AUTH) as c:
        yield c


def wait_job(client, job_id, timeout=60.0, states=("done", "failed", "cancelled")):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.1)
    pytest.fail(f"job {job_id} did not reach {states}; last state {job['state']}")


def add(client, path, **kw):
    r = client.post("/api/datasets", json={"path": str(path), **kw})
    assert r.status_code == 201, r.text
    return r.json()


def parse_sse(text: str) -> list[tuple[str, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields:
            events.append((fields["event"], fields.get("data", "")))
    return events


# ---- security ---------------------------------------------------------------

def test_token_host_and_cookie(tmp_path):
    app = create_app(make_settings(tmp_path))
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200                # health is open
        assert c.get("/api/datasets").status_code == 401
        assert c.get("/api/datasets", headers={"X-LCTIMS-Token": "nope"}).status_code == 401
        assert c.get("/api/datasets", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        assert c.get("/api/health", headers={"Host": "evil.example:8000"}).status_code == 403

        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303 and "token" not in r.headers["location"]
        assert "lctims_token" in r.headers["set-cookie"]
        c.cookies.set("lctims_token", TOKEN)
        assert c.get("/api/datasets").status_code == 200              # cookie alone suffices


def test_cross_origin_requests_are_refused(tmp_path):
    """Cookies ignore ports, so a page on another localhost port must not ride the session."""
    with TestClient(create_app(make_settings(tmp_path))) as c:
        c.cookies.set("lctims_token", TOKEN)
        ok = c.post("/api/peptides/fragments", json={"peptide": "PEPTIDE"},
                    headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"})
        assert ok.status_code == 200
        for headers in ({"Origin": "http://127.0.0.1:3000"},
                        {"Sec-Fetch-Site": "same-site"},
                        {"Sec-Fetch-Site": "cross-site"},
                        {"Origin": "null"}):
            r = c.post("/api/peptides/fragments", json={"peptide": "PEPTIDE"}, headers=headers)
            assert r.status_code == 403, headers
        # The bootstrap token is not accepted on the API itself.
        c.cookies.clear()
        assert c.get("/api/datasets", params={"token": TOKEN}).status_code == 401


def test_cookie_name_is_per_port(tmp_path):
    with TestClient(create_app(make_settings(tmp_path)), base_url="http://127.0.0.1:8123") as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.headers["set-cookie"].startswith("lctims_token_8123=")


def test_stale_cookie_is_replaced_by_url_token(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        c.cookies.set("lctims_token", "from-a-previous-run")
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert r.status_code == 303 and TOKEN in r.headers["set-cookie"]


# ---- system and datasets ----------------------------------------------------

def test_capabilities_and_fs(client, tmp_path):
    caps = client.get("/api/capabilities").json()
    assert {"available", "error", "remediation"} <= set(caps["sdk"])
    assert caps["browse_enabled"] is True
    assert client.get("/api/fs/roots").json()

    make_fake_d(tmp_path / "browse", "run1.d")
    (tmp_path / "browse" / "notes.txt").write_text("x")
    listing = client.get("/api/fs/list", params={"path": str(tmp_path / "browse")}).json()
    kinds = {e["name"]: e["kind"] for e in listing["entries"]}
    assert kinds == {"run1.d": "d_folder", "notes.txt": "file"}
    assert client.get("/api/fs/list", params={"path": str(tmp_path / "nope")}).status_code == 404


def test_browse_can_be_disabled(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, browse_enabled=False)), headers=AUTH) as c:
        assert c.get("/api/fs/roots").status_code == 403


def test_dataset_crud_and_persistence(tmp_path):
    d = make_fake_d(tmp_path / "data", "RH5956.d")
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings), headers=AUTH) as c:
        ds = add(c, d, group="iso")
        assert ds["name"] == "RH5956" and ds["cache"]["status"] == "missing"
        assert ds["meta"]["n_frames"] == 120
        assert add(c, d)["id"] == ds["id"]                             # same folder, same id
        assert c.patch(f"/api/datasets/{ds['id']}", json={"name": "rep1"}).json()["group"] == "iso"
        assert c.patch(f"/api/datasets/{ds['id']}", json={"group": None}).json()["group"] is None
        assert c.get("/api/datasets/unknown").status_code == 404
        r = c.post("/api/datasets", json={"path": str(tmp_path / "missing.d")})
        assert r.status_code == 422 and r.json()["error"] == "DatasetError"

    with TestClient(create_app(settings), headers=AUTH) as c:          # restart
        listed = c.get("/api/datasets").json()
        assert [x["name"] for x in listed] == ["rep1"]
        assert c.get("/api/jobs").json() == []                          # no phantom jobs
        assert c.delete(f"/api/datasets/{ds['id']}").status_code == 204
        assert c.get("/api/datasets").json() == []


def test_analysis_before_extraction_is_409(client, tmp_path):
    ds = add(client, make_fake_d(tmp_path / "data"))
    r = client.post("/api/signals/chromatograms",
                    json={"dataset_ids": [ds["id"]], "targets": [{"mz": TARGETS[0]}]})
    assert r.status_code == 409
    assert r.json()["detail"]["dataset_ids"] == [ds["id"]]


def test_fragments(client):
    r = client.post("/api/peptides/fragments", json={"peptide": "PEPTIDE", "charges": [1]})
    body = r.json()
    assert len(body["targets"]) == 12 and body["targets"][0]["label"] == "b1 (1+)"
    assert "M1+" in body["precursors"]
    assert client.post("/api/peptides/fragments", json={"peptide": "PEPTIDEB"}).status_code == 422


# ---- jobs -------------------------------------------------------------------

def test_extract_job_end_to_end(client, tmp_path):
    ds = add(client, make_fake_d(tmp_path / "data"))
    job = client.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()
    assert job["state"] in ("queued", "running")
    done = wait_job(client, job["id"])
    assert done["state"] == "done", done
    assert done["result"]["rows"] > 0
    assert client.get(f"/api/datasets/{ds['id']}").json()["cache"]["status"] == "cached"

    events = parse_sse(client.get("/api/events", params={"since": 0, "once": 1}).text)
    kinds = [k for k, _ in events]
    assert "progress" in kinds and "dataset" in kinds
    job_states = [data for k, data in events if k == "job"]
    assert any('"state":"running"' in s for s in job_states)
    assert any('"state":"done"' in s for s in job_states)

    # Already cached: a second extraction returns at once.
    again = wait_job(client, client.post("/api/jobs/extract",
                                         json={"dataset_id": ds["id"]}).json()["id"])
    assert again["state"] == "done" and again["result"]["key"] == done["result"]["key"]


def test_cancel_stops_worker_and_leaves_no_partial_cache(tmp_path):
    settings = make_settings(tmp_path, SLOW, extra={"frame_source_kwargs": {"delay": 0.05}})
    with TestClient(create_app(settings), headers=AUTH) as c:
        ds = add(c, make_fake_d(tmp_path / "data"))
        job = c.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()
        dup = c.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()
        assert dup["id"] == job["id"]                                   # deduplicated
        wait_job(c, job["id"], states=("running",))
        deadline = time.monotonic() + 30
        while c.get(f"/api/jobs/{job['id']}").json()["progress"] in (None,) or \
                c.get(f"/api/jobs/{job['id']}").json()["progress"]["done"] == 0:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert c.delete(f"/api/jobs/{job['id']}").json()["state"] == "cancelling"
        final = wait_job(c, job["id"], timeout=30)
        assert final["state"] == "cancelled"
        proc = c.app.state.lctims.jobs.get(job["id"]).proc
        assert proc.poll() is not None
        cache_dir = settings.cache_dir
        assert not cache_dir.exists() or list(cache_dir.iterdir()) == []


def test_native_crash_is_reported_with_output(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, CRASH)), headers=AUTH) as c:
        ds = add(c, make_fake_d(tmp_path / "data"))
        job = wait_job(c, c.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()["id"])
        assert job["state"] == "failed"
        assert job["error"]["error"] == "WorkerCrashed"
        assert "code 3" in job["error"]["message"]
        assert "timsdata: something went badly wrong" in "\n".join(job["log_tail"])
        assert not settings_cache_has_entries(tmp_path / "cache")


def settings_cache_has_entries(cache_dir: Path) -> bool:
    return cache_dir.exists() and any(p for p in cache_dir.iterdir() if ".tmp-" not in p.name)


def test_missing_sdk_fails_with_remediation(tmp_path, monkeypatch):
    monkeypatch.setenv("LCTIMSPY_TIMSDATA", str(tmp_path / "missing.dll"))
    with TestClient(create_app(make_settings(tmp_path, source=None)), headers=AUTH) as c:
        ds = add(c, make_fake_d(tmp_path / "data"))
        job = wait_job(c, c.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()["id"])
        assert job["state"] == "failed"
        assert job["error"]["error"] == "SdkUnavailableError"
        assert job["error"]["remediation"]


def test_copy_local_job(client, tmp_path):
    ds = add(client, make_fake_d(tmp_path / "share"))
    job = wait_job(client, client.post("/api/jobs/copy-local",
                                       json={"dataset_id": ds["id"]}).json()["id"])
    assert job["state"] == "done" and job["result"]["skipped"] is False
    local = client.get(f"/api/datasets/{ds['id']}").json()["local_path"]
    assert local and Path(local).is_dir() and (Path(local) / "analysis.tdf").is_file()
    assert str(tmp_path / "scratch") in local
    again = wait_job(client, client.post("/api/jobs/copy-local",
                                         json={"dataset_id": ds["id"]}).json()["id"])
    assert again["result"]["skipped"] is True


def test_shutdown_kills_running_workers(tmp_path):
    settings = make_settings(tmp_path, SLOW, extra={"frame_source_kwargs": {"delay": 0.5}})
    settings.cancel_grace_s = 1
    app = create_app(settings)
    with TestClient(app, headers=AUTH) as c:
        ds = add(c, make_fake_d(tmp_path / "data"))
        job_id = c.post("/api/jobs/extract", json={"dataset_id": ds["id"]}).json()["id"]
        wait_job(c, job_id, states=("running",))
        deadline = time.monotonic() + 20
        while app.state.lctims.jobs.get(job_id).proc is None:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    proc = app.state.lctims.jobs.get(job_id).proc
    proc.wait(15)
    assert proc.poll() is not None


# ---- analysis over extracted datasets --------------------------------------

@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    """Three replicates extracted through the API by real worker processes."""
    root = tmp_path_factory.mktemp("api")
    mp = pytest.MonkeyPatch()
    mp.setenv("LCTIMSPY_TESTING", "1")
    mp.setenv("PYTHONPATH", str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with TestClient(create_app(make_settings(root)), headers=AUTH) as c:
        ids = []
        for i in range(3):
            ds = add(c, make_fake_d(root / "data", f"rep{i}.d"))
            ids.append(ds["id"])
        # Different seeds per replicate so the stats have variance.
        for i, ds_id in enumerate(ids):
            c.app.state.lctims.settings.worker_spec_extra = {
                "frame_source": FAKE, "frame_source_kwargs": {"seed": i}}
            job = wait_job(c, c.post("/api/jobs/extract", json={"dataset_id": ds_id}).json()["id"])
            assert job["state"] == "done", job
        yield c, ids
    mp.undo()


def body(ids, **extra):
    return {"dataset_ids": ids, "targets": [{"mz": TARGETS[0], "label": "y1"},
                                             {"mz": TARGETS[1], "label": "y2"}], **extra}


def windows():
    return [{"start": a, "end": b} for a, b in WINDOWS]


def test_signals(extracted):
    c, ids = extracted
    r = c.post("/api/signals/chromatograms", json=body(ids[:1], max_points=40)).json()
    assert len(r["series"]) == 2 and len(r["series"][0]["x"]) == 40
    assert r["series"][0]["label"] == "y1"

    peaks = c.post("/api/chromatograms/peaks", json=body(ids[:1]) | {"dataset_id": ids[0]}).json()
    assert [round(p["apex_rt"], 1) for p in peaks["peaks"]] == [0.4, 0.8]

    mobs = c.post("/api/signals/mobilograms", json=body(ids[:1], windows=windows())).json()
    assert len(mobs["series"]) == 4
    assert {s["window"] for s in mobs["series"]} == {0, 1}


def test_mobility_difference_and_compare(extracted):
    c, ids = extracted
    diff = c.post("/api/analysis/mobility-difference", json=body(ids, windows=windows())).json()
    assert len(diff["differences"]) == 6 and diff["excluded"] == []
    row = diff["differences"][0]
    assert row["sample_name"] == "rep0" and row["label"] == "y1"
    assert row["mobility_difference"] == pytest.approx(0.0377, abs=0.001)

    cmp = c.post("/api/analysis/compare", json=body(ids, windows=windows())).json()
    assert len(cmp["pairs"]) == 1 and cmp["pairs"][0]["reject"] is True
    assert cmp["figure"]["data"]

    r = c.post("/api/analysis/mobility-difference",
               json=body(ids, windows=windows()[:1]))
    assert r.status_code == 422


def test_export_downloads(extracted):
    c, ids = extracted
    req = body(ids, windows=windows())
    r = c.post("/api/export", json={"kind": "mobility-difference", "format": "csv", "request": req})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.text.splitlines()[0].startswith("sample_id,sample_name")
    r = c.post("/api/export", json={"kind": "compare", "format": "html", "request": req})
    assert r.headers["content-type"].startswith("text/html") and "plotly" in r.text.lower()
    r = c.post("/api/export", json={"kind": "chromatograms", "format": "csv",
                                    "request": body(ids[:1])})
    assert "retention_time" in r.text.splitlines()[0]
    bad = c.post("/api/export", json={"kind": "mobilograms", "format": "csv", "request": {}})
    assert bad.status_code == 422


# ---- SSE --------------------------------------------------------------------

def test_event_bus_replay_and_live():
    async def run():
        bus = EventBus(maxlen=5)
        for i in range(3):
            bus.publish("x", {"i": i})
        sub = bus.subscribe(last_id=1)
        got = [await sub.next(0.1), await sub.next(0.1)]
        assert [e.id for e in got] == [2, 3]
        threading.Thread(target=lambda: bus.publish("x", {"i": 3})).start()
        live = await sub.next(2)
        assert live.id == 4
        assert await sub.next(0.05) is None                      # timeout, not end of stream
        sub.close()

        for i in range(10):
            bus.publish("x", {"i": i})
        gap = bus.subscribe(last_id=1)
        assert (await gap.next(0.1)).type == "resync"
        gap.close()

    asyncio.run(run())


def test_sse_live_stream_and_last_event_id(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    sock = bind_loopback(0)
    server = make_server(app)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{sock.getsockname()[1]}"
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        d = make_fake_d(tmp_path / "data")
        with httpx.Client(base_url=base, headers=AUTH, timeout=10) as http:
            with http.stream("GET", "/api/events") as stream:
                http.post("/api/datasets", json={"path": str(d)})
                seen = []
                for line in stream.iter_lines():
                    if line.startswith("event:"):
                        seen.append(line.split(": ", 1)[1])
                    if "dataset" in seen:
                        break
            last = app.state.lctims.bus.last_id
            http.patch(f"/api/datasets/{app.state.lctims.registry.list()[0].id}",
                       json={"group": "g"})
            replay = http.get("/api/events", params={"once": 1},
                              headers={"Last-Event-ID": str(last)}).text
            assert [k for k, _ in parse_sse(replay)] == ["dataset"]
    finally:
        server.should_exit = True
        thread.join(10)


def test_orphaned_worker_cancels_itself(tmp_path):
    """If the server dies, its worker notices and stops instead of running on."""
    import subprocess
    import sys

    from LCTIMSpy.server.jobs import worker_command
    from LCTIMSpy.server.worker.protocol import decode

    stand_in = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    d = make_fake_d(tmp_path / "data")
    spec = {"kind": "extract", "path": str(d), "cache_dir": str(tmp_path / "cache"),
            "frame_source": SLOW, "frame_source_kwargs": {"delay": 0.2},
            "parent_pid": stand_in.pid}
    worker = subprocess.Popen(worker_command(spec), stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, text=True, env=dict(os.environ))
    try:
        for line in worker.stdout:                      # wait until it is extracting
            msg = decode(line)
            if msg and msg["kind"] == "progress":
                break
        stand_in.kill()
        kinds = [m["kind"] for m in map(decode, worker.stdout) if m]
        assert worker.wait(30) == 0
        assert kinds[-1] == "cancelled"
        cache = tmp_path / "cache"
        assert not cache.exists() or list(cache.iterdir()) == []
    finally:
        stand_in.kill()
        worker.kill()


def test_built_frontend_is_served(tmp_path):
    from LCTIMSpy.server.app import STATIC_DIR

    if not (STATIC_DIR / "index.html").is_file():
        pytest.skip("web UI not built (cd web && npm run build)")
    with TestClient(create_app(make_settings(tmp_path)), headers=AUTH) as c:
        page = c.get("/")
        assert page.status_code == 200 and "<div id=\"root\">" in page.text
        asset = next((STATIC_DIR / "assets").glob("*.js")).name
        assert c.get(f"/assets/{asset}").status_code == 200
