"""Line protocol between a worker process and the server.

The worker writes one JSON object per line on stdout, prefixed with SENTINEL.
Any other line -- stray ``print``s from pyTDFSDK, pandas warnings, native
library chatter -- is not protocol and goes to a bounded log buffer that is only
surfaced if the job fails. Unfiltered high-rate stdout would otherwise flood the
event stream.

Worker -> server messages (``kind``):
    progress  stage, done, total, rate, eta_s, message
    log       message
    result    kind-specific payload; the job succeeded
    error     error, message, remediation; the job failed
    cancelled the job stopped on request and cleaned up

Server -> worker: cancellation is a file the server creates (``cancel_file`` in
the spec), and the worker also stops if the server process exits, so it never
runs on as an orphan. Not stdin: a pending pipe read on Windows can block DLL
loading in the worker (see ``child.watch_for_cancel``).
"""

from __future__ import annotations

import json

SENTINEL = "__LCTIMS__:"
TERMINAL_KINDS = frozenset({"result", "error", "cancelled"})


def encode(kind: str, **data) -> str:
    return SENTINEL + json.dumps({"kind": kind, **data}, separators=(",", ":"))


def decode(line: str) -> dict | None:
    """The message on a protocol line, or None for any other output."""
    line = line.rstrip("\r\n")
    if not line.startswith(SENTINEL):
        return None
    try:
        msg = json.loads(line[len(SENTINEL):])
    except ValueError:
        return None
    return msg if isinstance(msg, dict) and "kind" in msg else None
