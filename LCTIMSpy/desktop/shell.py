"""The desktop shell: the local server in a thread, the UI in a native window.

Startup:  bind 127.0.0.1:0 -> serve that socket in a thread -> wait for
          /api/health -> open the window at the token URL (sets the cookie).
          Opening the window first would show WebView2's connection-error
          page, which users report as a crash.
Closing:  if jobs are running, ask first. Then cancel every worker, stop the
          server, join it with a timeout, and ``os._exit`` as a backstop so a
          wedged native handle can never leave an invisible process holding the
          user's .d files open.
Fallback: if WebView2 cannot start (locked-down Windows 10/LTSC), open the
          default browser instead and exit once no tab has been connected for
          a while -- there is no console to Ctrl-C in a windowed build.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

from ..server.app import create_app
from ..server.runner import bind_loopback, make_server, url_for
from ..server.settings import ServerSettings, data_dir
from .logging_setup import setup_logging

log = logging.getLogger("lctimspy.desktop")

TITLE = "LCTIMSpy"
HEALTH_TIMEOUT_S = 30.0
SHUTDOWN_TIMEOUT_S = 10.0
IDLE_EXIT_S = 45.0              # browser fallback: exit after this long with no tab
FIRST_CONNECT_TIMEOUT_S = 180.0  # ...or if no tab ever connects
SMOKE_TIMEOUT_S = 30.0
BACKGROUND = "#f8fafc"           # matches --bg, so the window does not flash white


class StartupError(RuntimeError):
    pass


@dataclass
class ServerThread:
    """The FastAPI app served from a background thread on a pre-bound socket."""

    settings: ServerSettings
    app: object = field(init=False)
    server: object = field(init=False)
    sock: object = field(init=False)
    thread: threading.Thread | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.app = create_app(self.settings)
        self.server = make_server(self.app)
        self.sock = bind_loopback(0)

    @property
    def base_url(self) -> str:
        return url_for(self.sock, None).rstrip("/")

    @property
    def url(self) -> str:
        return url_for(self.sock, self.settings.token)

    @property
    def state(self):
        return self.app.state.lctims

    def start(self, timeout: float = HEALTH_TIMEOUT_S) -> None:
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]},
                                       name="uvicorn", daemon=True)
        self.thread.start()
        wait_until_healthy(self.base_url, timeout, alive=self.thread.is_alive)
        log.info("server ready at %s", self.base_url)

    def stop(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> bool:
        """Cancel workers, stop uvicorn, and wait. True if everything stopped in time."""
        deadline = time.monotonic() + timeout
        try:
            self.state.jobs.shutdown(timeout=max(1.0, timeout / 2))
        except Exception:
            log.exception("job shutdown failed")
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(max(0.0, deadline - time.monotonic()))
            stopped = not self.thread.is_alive()
        else:
            stopped = True
        try:
            self.sock.close()
        except OSError:
            pass
        return stopped

    def active_jobs(self) -> int:
        return sum(1 for j in self.state.jobs.list() if j.state in ("queued", "running", "cancelling"))


def wait_until_healthy(base_url: str, timeout: float, alive=lambda: True) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if not alive():
            raise StartupError("the server thread exited during startup; see the log")
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=2) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last = exc
        time.sleep(0.1)
    raise StartupError(f"the server did not answer {base_url}/api/health within {timeout:.0f}s: {last}")


class IdleMonitor:
    """Decides when the browser fallback should exit: no connected tab for a while."""

    def __init__(self, count, idle_s: float = IDLE_EXIT_S,
                 first_connect_s: float = FIRST_CONNECT_TIMEOUT_S, clock=time.monotonic):
        self.count = count
        self.idle_s = idle_s
        self.first_connect_s = first_connect_s
        self.clock = clock
        self.started = clock()
        self.seen = False
        self.idle_since: float | None = None

    def should_exit(self) -> bool:
        now = self.clock()
        if self.count() > 0:
            self.seen = True
            self.idle_since = None
            return False
        if not self.seen:
            return now - self.started > self.first_connect_s
        if self.idle_since is None:
            self.idle_since = now
        return now - self.idle_since > self.idle_s


# ---- window -----------------------------------------------------------------------


def run_window(srv: ServerThread, *, debug: bool, smoke: bool, storage: Path) -> int:
    """Show the UI in a WebView2 window; returns when it closes. Raises if WebView2 fails."""
    import webview

    webview.settings["ALLOW_DOWNLOADS"] = True       # CSV/HTML exports are downloads
    window = webview.create_window(
        TITLE, srv.url, width=1440, height=900, min_size=(1024, 680),
        text_select=True, background_color=BACKGROUND,
    )

    def on_closing():
        n = srv.active_jobs()
        if n == 0 or smoke:
            return True
        try:
            ok = window.create_confirmation_dialog(
                "Quit LCTIMSpy?",
                f"{n} job{'s are' if n > 1 else ' is'} still running and will be cancelled. "
                "Extractions in progress leave no partial cache, but will have to start over.")
        except Exception:
            log.exception("confirmation dialog failed; closing anyway")
            return True
        return bool(ok)                                  # False keeps the window open

    window.events.closing += on_closing
    result = {"code": 0}
    if smoke:
        window.events.loaded += lambda: threading.Thread(
            target=_smoke_check, args=(window, result), daemon=True).start()

    webview.start(debug=debug, private_mode=False, storage_path=str(storage))
    return result["code"]


def _smoke_check(window, result: dict) -> None:
    """Prove the UI rendered and the session cookie works inside WebView2, then close."""
    deadline = time.monotonic() + SMOKE_TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            if window.evaluate_js("!!document.querySelector('.sidebar')"):
                break
            time.sleep(0.25)
        else:
            raise StartupError("the UI did not render")
        # Synchronous XHR: evaluate_js does not await promises.
        status = window.evaluate_js(
            "(() => { const x = new XMLHttpRequest(); x.open('GET', '/api/capabilities', false);"
            " x.send(); return x.status; })()")
        heading = window.evaluate_js("document.querySelector('h1')?.textContent")
        if status != 200:
            raise StartupError(f"API call from the window returned {status} (session cookie not set?)")
        print(f"SMOKE OK: rendered {heading!r}, API {status}", flush=True)
        log.info("smoke OK: %r, API %s", heading, status)
    except Exception as exc:
        print(f"SMOKE FAILED: {exc}", flush=True)
        log.error("smoke failed: %s", exc)
        result["code"] = 1
    finally:
        window.destroy()


# ---- browser fallback -----------------------------------------------------------


def run_browser(srv: ServerThread, *, smoke: bool) -> int:
    print(f"{TITLE} is running at {srv.url}", flush=True)
    if smoke:
        wait_until_healthy(srv.base_url, 5)
        print("SMOKE OK (browser mode): server healthy", flush=True)
        return 0
    webbrowser.open(srv.url)
    monitor = IdleMonitor(lambda: srv.state.bus.subscriber_count)
    try:
        while not monitor.should_exit():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    log.info("browser fallback: no tab connected, exiting")
    return 0


# ---- entry ------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lctimspy desktop", description="LCTIMSpy desktop app")
    p.add_argument("--browser", action="store_true", help="use the default browser instead of a window")
    p.add_argument("--smoke", action="store_true", help="start, check the UI renders, and exit")
    p.add_argument("--debug", action="store_true", help="verbose logging and WebView2 dev tools")
    return p.parse_args(argv)


def run(argv: list[str]) -> int:
    args = parse_args(argv)
    root = data_dir()
    log_file = setup_logging(root / "logs", debug=args.debug)
    log.info("starting %s (smoke=%s, browser=%s)", TITLE, args.smoke, args.browser)

    try:
        srv = ServerThread(ServerSettings())
        srv.start()
    except Exception as exc:
        log.exception("startup failed")
        _fatal(f"{TITLE} could not start its local server:\n{exc}\n\nLog: {log_file}")
        return 1

    code = 0
    try:
        if args.browser:
            code = run_browser(srv, smoke=args.smoke)
        else:
            try:
                code = run_window(srv, debug=args.debug, smoke=args.smoke, storage=root / "webview")
            except Exception as exc:
                log.exception("WebView2 window failed; falling back to the browser")
                print(f"Could not open the {TITLE} window ({exc}); using the default browser.",
                      flush=True)
                if args.smoke:
                    print("SMOKE FAILED: WebView2 unavailable. Install the Microsoft Edge WebView2 "
                          "Runtime: https://go.microsoft.com/fwlink/p/?LinkId=2124703", flush=True)
                    code = 2
                else:
                    code = run_browser(srv, smoke=False)
    finally:
        clean = srv.stop()
        log.info("shutdown %s", "clean" if clean else "timed out; forcing exit")
        logging.shutdown()
    # Backstop: pythonnet/WebView2 and native SDK handles can hold non-daemon
    # threads; never leave an invisible process behind.
    os._exit(code)


def _fatal(message: str) -> None:
    print(message, flush=True)
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, TITLE, 0x10)
        except Exception:
            pass
