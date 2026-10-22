"""Desktop entry point -- also the entry point of the frozen ``LCTIMSpy.exe``.

Keep this module's imports minimal. Under PyInstaller there is no python.exe, so
job workers re-launch this same executable with ``--lctims-worker``; anything
imported here at module scope would be imported again by every worker, and the
GUI stack must never load in a worker. So the worker dispatch and
``freeze_support()`` come first, and the heavy shell is imported only after.
"""

from __future__ import annotations

import sys

WORKER_FLAG = "--lctims-worker"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == WORKER_FLAG:
        from ..server.worker.child import main as worker_main

        return worker_main(argv[1:])

    import multiprocessing

    multiprocessing.freeze_support()

    from .shell import run

    return run(argv)
