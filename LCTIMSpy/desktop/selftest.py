"""``LCTIMSpy.exe --self-test``: prove a build works end to end, without real data.

A frozen app fails in ways tests of the source tree cannot see: a module or
data file PyInstaller did not collect only fails when that code path runs. So
this drives the real workflow through the real API on synthetic data:

1. three synthetic ``.d`` folders, extracted by **worker processes** -- under
   PyInstaller that is the exe re-launching itself with ``--lctims-worker``,
2. chromatograms, mobility difference, and Tukey statistics from the cache,
3. an HTML figure export (plotly.js must be bundled),
4. a probe of the bundled Bruker SDK.

Exit code 0 and ``SELF-TEST OK`` mean the build is good. Results go to stdout
and the log (a windowed build has no console).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger("lctimspy.selftest")

TARGETS = (500.25, 700.40)
ISOMERS = ((40, 0.9123), (80, 0.9500))       # (apex frame, mobility)
N_FRAMES = 120


class SyntheticFrameSource:
    """An isomer pair per target; the worker loads this by name (test hook)."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def read_frame(self, frame_id: int, num_scans: int):
        rng = np.random.default_rng([self.seed, frame_id])
        mz = list(rng.uniform(100, 1700, 200))
        intensity = list(rng.uniform(10, 100, 200).round())
        mobility = list(rng.uniform(0.6, 1.4, 200))
        for t, target in enumerate(TARGETS):
            for i, (apex, mob) in enumerate(ISOMERS):
                amp = 1000 * (1 + 0.5 * t) * np.exp(-0.5 * ((frame_id - apex) / 4) ** 2)
                if amp < 1:
                    continue
                centre = mob + (0.004 if t == 1 and i == 1 else 0.0)
                for k in range(-3, 4):
                    mz.append(target + rng.normal(0, 0.002))
                    intensity.append(round(amp * np.exp(-0.25 * k * k) * rng.uniform(0.8, 1.2)) + 1)
                    mobility.append(centre + 0.003 * k)
        order = np.argsort(mz)
        return tuple(np.asarray(a)[order] for a in (mz, intensity, mobility))


def make_synthetic_d(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "analysis.tdf_bin").write_bytes(b"\0" * 16)
    conn = sqlite3.connect(d / "analysis.tdf")
    conn.execute("CREATE TABLE Frames (Id INTEGER PRIMARY KEY, Time REAL, Polarity TEXT, "
                 "MsMsType INTEGER, NumScans INTEGER)")
    conn.executemany("INSERT INTO Frames VALUES (?, ?, ?, ?, ?)",
                     [(i, i * 0.6, "+", 0 if i % 5 else 8, 10) for i in range(1, N_FRAMES + 1)])
    conn.execute("CREATE TABLE GlobalMetadata (Key TEXT PRIMARY KEY, Value TEXT)")
    conn.executemany("INSERT INTO GlobalMetadata VALUES (?, ?)",
                     [("MzAcqRangeLower", "100"), ("MzAcqRangeUpper", "1700")])
    conn.commit()
    conn.close()
    return d


class Api:
    def __init__(self, base: str):
        self.base = base

    def call(self, method: str, path: str, body=None):
        req = urllib.request.Request(
            f"{self.base}/api{path}", method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"} if body is not None else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            ctype = r.headers.get("content-type", "")
        return json.loads(raw) if "json" in ctype else raw.decode("utf-8", "replace")


def run() -> int:
    from ..core.sdk import probe_sdk
    from ..server.settings import ServerSettings
    from .shell import ServerThread

    started = time.perf_counter()
    checks: list[str] = []

    def ok(msg: str) -> None:
        checks.append(msg)
        print(f"  ok  {msg}", flush=True)
        log.info("self-test ok: %s", msg)

    # The worker accepts a frame-source override only in testing mode.
    os.environ["LCTIMSPY_TESTING"] = "1"
    root = Path(tempfile.mkdtemp(prefix="lctims-selftest-"))
    settings = ServerSettings(cache_dir=root / "cache", state_dir=root / "state",
                              scratch_dir=root / "scratch", token=None)
    srv = ServerThread(settings)
    try:
        print(f"LCTIMSpy self-test in {root}", flush=True)
        srv.start()
        api = Api(srv.base_url)
        ok("server started")

        ids = []
        for i in range(3):
            d = make_synthetic_d(root / "data", f"SELFTEST{i}.d")
            ids.append(api.call("POST", "/datasets", {"path": str(d)})["id"])
        jobs = []
        for i, ds_id in enumerate(ids):
            settings.worker_spec_extra = {
                "frame_source": "LCTIMSpy.desktop.selftest:SyntheticFrameSource",
                "frame_source_kwargs": {"seed": i}}
            jobs.append(api.call("POST", "/jobs/extract", {"dataset_id": ds_id})["id"])
        deadline = time.monotonic() + 180
        for job_id in jobs:
            while True:
                job = api.call("GET", f"/jobs/{job_id}")
                if job["state"] in ("done", "failed", "cancelled"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"extraction job {job_id} did not finish")
                time.sleep(0.2)
            if job["state"] != "done":
                raise RuntimeError(f"extraction failed: {job.get('error')}\n"
                                   + "\n".join(job.get("log_tail") or []))
        ok(f"3 extractions in worker processes ({sum(1 for _ in jobs)} spawned)")

        targets = [{"mz": mz, "label": f"t{i}"} for i, mz in enumerate(TARGETS)]
        eic = api.call("POST", "/signals/chromatograms", {"dataset_ids": ids[:1], "targets": targets})
        assert len(eic["series"]) == 2 and max(eic["series"][0]["y"]) > 0
        peaks = api.call("POST", "/chromatograms/peaks", {"dataset_id": ids[0], "targets": targets})
        assert len(peaks["peaks"]) == 2, peaks
        ok("chromatograms and two-peak detection")

        windows = [{"start": 0.30, "end": 0.50}, {"start": 0.70, "end": 0.90}]
        body = {"dataset_ids": ids, "targets": targets, "windows": windows,
                "stats": {"correction": "tukey"}}
        cmp = api.call("POST", "/analysis/compare", body)
        diffs = cmp["differences"]["differences"]
        assert len(diffs) == 6, diffs
        shift = sum(r["mobility_difference"] for r in diffs if r["label"] == "t0") / 3
        assert abs(shift - 0.0377) < 0.002, shift
        assert cmp["pairs"] and cmp["pairs"][0]["reject"] is True, cmp["pairs"]
        ok(f"mobility difference {shift:.4f} and Tukey statistics")

        html = api.call("POST", "/export", {"kind": "compare", "format": "html", "request": body})
        assert "plotly" in html.lower() and len(html) > 1_000_000, len(html)
        ok(f"HTML figure export ({len(html) // 1024} KB, plotly.js embedded)")

        index = urllib.request.urlopen(f"{srv.base_url}/", timeout=10).read().decode()
        assert '<div id="root">' in index
        ok("web UI served")

        sdk = probe_sdk()
        ok(f"Bruker SDK {'loads' if sdk.available else 'unavailable: ' + str(sdk.error)}")

        print(f"SELF-TEST OK ({len(checks)} checks, {time.perf_counter() - started:.1f}s)",
              flush=True)
        return 0
    except Exception as exc:
        log.exception("self-test failed")
        print(f"SELF-TEST FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1
    finally:
        srv.stop()
        import shutil

        shutil.rmtree(root, ignore_errors=True)
