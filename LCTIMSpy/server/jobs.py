"""Background jobs, each in its own worker process.

Extraction drives ``timsdata.dll`` through ctypes, where an access violation
kills the whole interpreter. Running it in a subprocess keeps the server alive
when that happens, makes cancellation real (the process can be killed), and
guarantees native handles are released when the job ends. Post-cache queries
are fast pyarrow/numpy work and stay in the server's thread pool.

Disk-bound work does not parallelise on one spindle, so at most
``max_concurrent`` (default 2) jobs run; the rest queue.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .events import EventBus
from .worker.protocol import TERMINAL_KINDS, decode

LOG_TAIL_LINES = 200
CANCEL_GRACE_S = 10.0
ACTIVE_STATES = frozenset({"queued", "running", "cancelling"})


def worker_command(spec: dict) -> list[str]:
    """How to start a worker: re-invoke the frozen exe, or ``python -m`` otherwise."""
    payload = json.dumps(spec)
    if getattr(sys, "frozen", False):
        return [sys.executable, "--lctims-worker", payload]
    return [sys.executable, "-m", "LCTIMSpy.server.worker.child", payload]


@dataclass
class Job:
    id: str
    kind: str
    spec: dict
    dataset_id: str | None = None
    state: str = "queued"
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: dict | None = None
    result: dict | None = None
    error: dict | None = None
    log_tail: deque = field(default_factory=lambda: deque(maxlen=LOG_TAIL_LINES))
    # Partial-output folders the worker announced; removed if it dies uncleanly.
    temp_dirs: list[str] = field(default_factory=list)
    cancel_requested: bool = False
    proc: subprocess.Popen | None = field(default=None, repr=False)

    def snapshot(self, include_log: bool = False) -> dict:
        out = {
            "id": self.id, "kind": self.kind, "dataset_id": self.dataset_id, "state": self.state,
            "created": self.created, "started": self.started, "finished": self.finished,
            "progress": self.progress, "result": self.result, "error": self.error,
        }
        if include_log:
            out["log_tail"] = list(self.log_tail)
        return out


class JobManager:
    def __init__(self, bus: EventBus, max_concurrent: int = 2,
                 on_finished: Callable[[Job], None] | None = None,
                 cancel_grace_s: float = CANCEL_GRACE_S, control_dir: Path | None = None):
        self.bus = bus
        # Where per-job cancel flag files live.
        self.control_dir = Path(control_dir) if control_dir else Path(tempfile.gettempdir())
        self.max_concurrent = max_concurrent
        self.on_finished = on_finished
        self.cancel_grace_s = cancel_grace_s
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._running = 0
        self._lock = threading.RLock()
        self._closed = False

    # ---- queries ----------------------------------------------------------------

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def find_active(self, kind: str, dataset_id: str | None) -> Job | None:
        with self._lock:
            for job in self._jobs.values():
                if job.kind == kind and job.dataset_id == dataset_id and job.state in ACTIVE_STATES:
                    return job
        return None

    # ---- lifecycle ----------------------------------------------------------

    def submit(self, kind: str, spec: dict, dataset_id: str | None = None) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            job_id = uuid.uuid4().hex[:12]
            spec = {**spec, "cancel_file": str(self._cancel_file(job_id)),
                    "parent_pid": os.getpid()}
            job = Job(id=job_id, kind=kind, spec=spec, dataset_id=dataset_id)
            self._jobs[job.id] = job
            self._queue.append(job)
        self._publish(job)
        self._dispatch()
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        with self._lock:
            if job.state == "queued":
                self._queue.remove(job)
                job.state, job.finished = "cancelled", time.time()
                self._publish(job)
                return job
            if job.state != "running":
                return job
            job.cancel_requested = True
            job.state = "cancelling"
        self._publish(job)
        threading.Thread(target=self._cancel_ladder, args=(job,), daemon=True,
                         name=f"cancel-{job.id}").start()
        return job

    def shutdown(self, timeout: float = 10.0) -> None:
        """Cancel everything; kill whatever has not stopped by ``timeout``."""
        with self._lock:
            self._closed = True
            queued = list(self._queue)
            running = [j for j in self._jobs.values() if j.state in ("running", "cancelling")]
        for job in queued:
            self.cancel(job.id)
        for job in running:
            self.cancel(job.id)
        deadline = time.monotonic() + timeout
        for job in running:
            proc = job.proc
            if proc is None:
                continue
            try:
                proc.wait(max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()

    # ---- internals ----------------------------------------------------------

    def _publish(self, job: Job) -> None:
        self.bus.publish("job", job.snapshot())

    def _dispatch(self) -> None:
        started = []
        with self._lock:
            while self._queue and self._running < self.max_concurrent and not self._closed:
                job = self._queue.popleft()
                self._running += 1
                job.state, job.started = "running", time.time()
                started.append(job)
        for job in started:
            self._publish(job)
            threading.Thread(target=self._run, args=(job,), daemon=True,
                             name=f"job-{job.id}").start()

    def _run(self, job: Job) -> None:
        terminal = None
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            job.proc = subprocess.Popen(
                worker_command(job.spec), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, env=env, creationflags=flags,
            )
            if job.cancel_requested:          # cancelled before the process existed
                threading.Thread(target=self._cancel_ladder, args=(job,), daemon=True).start()
            for line in job.proc.stdout:
                msg = decode(line)
                if msg is None:
                    if line.strip():
                        job.log_tail.append(line.rstrip())
                    continue
                kind = msg.pop("kind")
                if kind == "progress":
                    job.progress = msg
                    self.bus.publish("progress", {"job_id": job.id, **msg})
                elif kind == "log":
                    if msg.get("temp_dir"):
                        job.temp_dirs.append(msg["temp_dir"])
                    job.log_tail.append(msg.get("message", ""))
                    self.bus.publish("log", {"job_id": job.id, **msg})
                elif kind in TERMINAL_KINDS:
                    terminal = (kind, msg)
            code = job.proc.wait()
        except OSError as exc:
            terminal, code = ("error", {"error": "SpawnError", "message": str(exc)}), -1
        finally:
            self._cancel_file(job.id).unlink(missing_ok=True)

        if terminal and terminal[0] == "result":
            job.state, job.result = "done", terminal[1]
        elif terminal and terminal[0] == "cancelled":
            job.state = "cancelled"
        elif terminal and terminal[0] == "error":
            job.state = "failed"
            job.error = {"remediation": None, **terminal[1]}
        elif job.cancel_requested:
            job.state = "cancelled"            # killed by the cancel ladder
        else:
            tail = "\n".join(list(job.log_tail)[-20:])
            job.state = "failed"
            job.error = {
                "error": "WorkerCrashed",
                "message": f"worker exited with code {code} without reporting a result; "
                           "this usually means a crash in the native Bruker library."
                           + (f"\nLast output:\n{tail}" if tail else ""),
                "remediation": None,
            }
        if not terminal or terminal[0] not in TERMINAL_KINDS:
            # Killed or crashed: the worker never ran its own cleanup.
            for d in job.temp_dirs:
                shutil.rmtree(d, ignore_errors=True)
        job.finished = time.time()
        with self._lock:
            self._running -= 1
        self._publish(job)
        if self.on_finished is not None:
            try:
                self.on_finished(job)
            except Exception as exc:  # a callback bug must not wedge the queue
                job.log_tail.append(f"on_finished failed: {exc!r}")
        self._dispatch()

    def _cancel_file(self, job_id: str) -> Path:
        return self.control_dir / f"lctims-cancel-{job_id}"

    def _cancel_ladder(self, job: Job) -> None:
        proc = job.proc
        if proc is None:
            return            # _run starts the ladder itself once the process exists
        try:
            self.control_dir.mkdir(parents=True, exist_ok=True)
            self._cancel_file(job.id).touch()
        except OSError:
            pass              # fall through to terminate()
        try:
            proc.wait(self.cancel_grace_s)
            return
        except subprocess.TimeoutExpired:
            proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
