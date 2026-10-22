"""Phase 5: the desktop shell, without needing a window (except the opt-in GUI smoke)."""

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from LCTIMSpy.desktop.shell import IdleMonitor, ServerThread, StartupError, wait_until_healthy
from LCTIMSpy.server.settings import ServerSettings
from tests.conftest import make_fake_d

REPO = Path(__file__).resolve().parents[2]


def env_for(tmp_path, **extra):
    return dict(os.environ, LCTIMSPY_DATA_DIR=str(tmp_path / "data"),
                LCTIMSPY_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(REPO), **extra)


def test_worker_dispatch_never_loads_the_gui_stack(tmp_path):
    """Frozen workers re-run the desktop entry point; they must stay lightweight."""
    code = (
        "import sys\n"
        "import LCTIMSpy.desktop.main as m\n"
        "heavy = [x for x in ('webview', 'clr', 'fastapi', 'uvicorn', 'pandas') if x in sys.modules]\n"
        "assert not heavy, ('module import', heavy)\n"
        "rc = m.main(['--lctims-worker', '{not json'])\n"
        "heavy = [x for x in ('webview', 'clr', 'fastapi', 'uvicorn') if x in sys.modules]\n"
        "assert not heavy, ('worker', heavy)\n"
        "sys.exit(rc)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env_for(tmp_path), timeout=60)
    assert r.returncode == 2, r.stderr
    assert "__LCTIMS__:" in r.stdout and "BadSpec" in r.stdout


def make_settings(tmp_path, **kw) -> ServerSettings:
    return ServerSettings(cache_dir=tmp_path / "cache", state_dir=tmp_path / "state",
                          scratch_dir=tmp_path / "scratch", **kw)


def test_server_thread_start_and_clean_stop(tmp_path):
    srv = ServerThread(make_settings(tmp_path, token="t"))
    srv.start()
    port = srv.sock.getsockname()[1]
    with urllib.request.urlopen(f"{srv.base_url}/api/health") as r:
        assert r.status == 200
    assert srv.url.endswith("/?token=t")
    assert srv.stop(timeout=10) is True
    assert not srv.thread.is_alive()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)


def test_stop_kills_running_extraction(tmp_path, monkeypatch):
    monkeypatch.setenv("LCTIMSPY_TESTING", "1")
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    settings = make_settings(tmp_path, token=None, cancel_grace_s=2, worker_spec_extra={
        "frame_source": "tests.conftest:SlowFakeFrameSource", "frame_source_kwargs": {"delay": 0.5}})
    srv = ServerThread(settings)
    srv.start()
    ds = srv.state.registry.add(str(make_fake_d(tmp_path / "data")))
    job = srv.state.jobs.submit("extract", {
        "kind": "extract", "path": ds.path, "cache_dir": str(settings.cache_dir),
        **settings.worker_spec_extra}, ds.id)
    deadline = time.monotonic() + 30
    while job.proc is None or not job.progress:
        assert time.monotonic() < deadline, "worker never reported progress"
        time.sleep(0.05)
    assert srv.active_jobs() == 1
    srv.stop(timeout=15)
    assert job.proc.poll() is not None, "worker survived shutdown"
    leftovers = [p for p in settings.cache_dir.iterdir()] if settings.cache_dir.exists() else []
    assert leftovers == []


def test_wait_until_healthy_reports_dead_server():
    with pytest.raises(StartupError, match="exited"):
        wait_until_healthy("http://127.0.0.1:9", timeout=5, alive=lambda: False)
    with pytest.raises(StartupError, match="did not answer"):
        wait_until_healthy("http://127.0.0.1:9", timeout=0.3)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_idle_monitor_waits_for_first_tab_then_for_idle():
    clock, tabs = Clock(), {"n": 0}
    m = IdleMonitor(lambda: tabs["n"], idle_s=45, first_connect_s=180, clock=clock)
    clock.t = 100
    assert not m.should_exit()                 # still waiting for the browser to open
    tabs["n"] = 1
    clock.t = 120
    assert not m.should_exit()
    tabs["n"] = 0                              # tab closed (or reloading)
    clock.t = 130
    assert not m.should_exit()
    clock.t = 150
    tabs["n"] = 1                              # a reload reconnects in time
    assert not m.should_exit()
    tabs["n"] = 0
    clock.t = 160
    assert not m.should_exit()
    clock.t = 206
    assert m.should_exit()


def test_idle_monitor_gives_up_if_no_tab_ever_connects():
    clock = Clock()
    m = IdleMonitor(lambda: 0, idle_s=45, first_connect_s=180, clock=clock)
    clock.t = 179
    assert not m.should_exit()
    clock.t = 181
    assert m.should_exit()


def test_run_in_browser_smoke_mode(tmp_path):
    """The whole run() path -- logging, server, shutdown, os._exit -- without a window."""
    r = subprocess.run([sys.executable, "-m", "LCTIMSpy.desktop", "--browser", "--smoke"],
                       capture_output=True, text=True, env=env_for(tmp_path), timeout=90)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SMOKE OK (browser mode)" in r.stdout
    log = (tmp_path / "data" / "logs" / "lctimspy.log").read_text(encoding="utf-8")
    assert "shutdown clean" in log


@pytest.mark.skipif(os.environ.get("LCTIMSPY_GUI_TESTS") != "1",
                    reason="opens a real window; set LCTIMSPY_GUI_TESTS=1")
def test_desktop_window_smoke(tmp_path):
    r = subprocess.run([sys.executable, "-m", "LCTIMSpy.desktop", "--smoke"],
                       capture_output=True, text=True, env=env_for(tmp_path), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SMOKE OK: rendered 'Datasets', API 200" in r.stdout


def test_self_test_passes_from_source(tmp_path, monkeypatch):
    """The checks LCTIMSpy.exe --self-test runs, here against the source tree."""
    monkeypatch.setenv("LCTIMSPY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYTHONPATH", str(REPO))
    monkeypatch.setenv("LCTIMSPY_TESTING", "1")   # run() sets it; this makes pytest restore it
    from LCTIMSpy.desktop.selftest import run

    assert run() == 0
