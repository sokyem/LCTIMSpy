"""Progress reporting and cooperative cancellation for long-running jobs.

Extraction reports through a ``ProgressReporter`` and polls a ``CancelToken``
between frame batches. The CLI renders events as a progress bar; the server
(Phase 3) will forward the same events over SSE.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Protocol

from .errors import CancelledError


@dataclass(frozen=True)
class ProgressEvent:
    stage: str                     # "extract" | "finalize" | ...
    done: int
    total: int
    message: str = ""
    rate: float | None = None      # units per second over the stage so far
    eta_s: float | None = None     # estimated seconds remaining in the stage

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class ProgressReporter(Protocol):
    def __call__(self, event: ProgressEvent) -> None: ...


def null_reporter(event: ProgressEvent) -> None:
    """Reporter that discards every event."""


class StageTimer:
    """Turns ``(done, total)`` counts into events with a rate and an ETA."""

    def __init__(self, stage: str, total: int, reporter: ProgressReporter):
        self.stage = stage
        self.total = total
        self.reporter = reporter
        self._start = time.perf_counter()

    def update(self, done: int, message: str = "") -> None:
        elapsed = time.perf_counter() - self._start
        rate = done / elapsed if elapsed > 0 and done else None
        eta = (self.total - done) / rate if rate else None
        self.reporter(ProgressEvent(self.stage, done, self.total, message, rate, eta))


class CancelToken:
    """Thread-safe cancellation flag, polled by the worker between batches."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("Operation cancelled")
